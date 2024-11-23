import datetime
import dataclasses
import dotenv
import typing

import torch
import torch.distributed as dist
import torch.nn.parallel as torch_pl
import tqdm
import transformers as tr
import wandb

from team17.modeling.model import UltravoxConfig, UltravoxModel
from team17.modeling.processor import UltravoxProcessor
from team17.trainer import utils
from team17.trainer import lora
from team17.trainer.config import MyUltravoxTrainConfig
from team17.trainer.dataset import MyUltravoxDataset, DataCollatorForSeq2SeqWithAudio
from team17.trainer.test import test

dotenv.load_dotenv()

__all__ = ["train"]


class TrainState(typing.NamedTuple):
    step: int
    model: UltravoxModel
    optimizer: torch.optim.AdamW  # type: ignore
    scheduler: torch.optim.lr_scheduler.LRScheduler  # type: ignore
    train_dataset: MyUltravoxDataset


@dataclasses.dataclass
class LossConfig:
    loss_function: str = "kl"
    kl_temperature: float = 2.0

    @property
    def requires_alt_fields(self):
        return self.loss_function == "kl"


def init_model(config: MyUltravoxTrainConfig):
    device = utils.get_device()
    if config.ultravox_pretrained_path is not None:
        model = UltravoxModel.from_pretrained(config.ultravox_pretrained_path)
    else:
        model = UltravoxModel(
            UltravoxConfig(
                audio_model_id="openai/whisper-tiny",
                text_config=tr.LlamaConfig(
                    vocab_size=128128, hidden_size=64, num_hidden_layers=1
                ).to_dict(),
                audio_latency_block_size=None,
            )
        )

    model.set_loss_config(LossConfig(loss_function="ce"))

    # Apply LoRA to the model
    skip_modules = ["head", "emb", "cond_net"]  # Modules to skip for LoRA
    lora.add_lora(model, lora_rank=config.lora_r, skip=skip_modules)

    model = model.to(device, torch.bfloat16)
    return model


def init_train_state(config: MyUltravoxTrainConfig) -> TrainState:
    device = utils.get_device()

    model = init_model(config)

    # Filter trainable parameters for optimizer
    optimizer_params = []
    for name, param in model.named_parameters():
        if "lora" in name:  # Only train LoRA parameters
            optimizer_params.append(
                {"params": [param], "weight_decay": 0.0 if param.ndim == 1 else 0.1}
            )

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=config.lr,
        betas=(0.9, 0.95),
        eps=1e-11,
        fused=True if device.type == "cuda" else False,
    )

    scheduler = tr.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )

    # Initialize dataset and processor
    processor = UltravoxProcessor.from_pretrained("fixie-ai/ultravox-v0_3-llama-3_2-1b")
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    train_dataset = MyUltravoxDataset(processor)

    return TrainState(
        step=0,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset,
    )


def load_train_state(state: TrainState, config: MyUltravoxTrainConfig) -> TrainState:
    assert config.ckpt_path is not None
    checkpoint = utils.load_checkpoint(config.ckpt_path)
    state.model.load_state_dict(checkpoint["model"])
    state.optimizer.load_state_dict(checkpoint["optimizer"])
    state.scheduler.load_state_dict(checkpoint["scheduler"])
    state = state._replace(step=checkpoint["step"])
    return state


def save_train_state(state: TrainState, config: MyUltravoxTrainConfig) -> None:
    model = utils.unwrap_model(state.model)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": state.optimizer.state_dict(),
        "scheduler": state.scheduler.state_dict(),
        "step": state.step,
    }
    utils.save_checkpoint(checkpoint, step=state.step, run_id=config.run_id)


def prepare_batch(
    batch: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare batch by moving tensors to device"""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    audio_values = batch["audio_values"].to(device, dtype=torch.bfloat16)
    audio_token_start_idx = batch["audio_token_start_idx"].to(device)
    audio_token_len = batch["audio_token_len"].to(device)
    return input_ids, labels, audio_values, audio_token_start_idx, audio_token_len


def push_to_hub(
    model: UltravoxModel,
    processor: UltravoxProcessor,
    config: MyUltravoxTrainConfig,
    step: int,
) -> None:
    """Push model and processor to hub with merged LoRA weights"""
    # Create CPU copy of model and merge LoRA weights
    # # MEGA HACK CUZ DEEPCOPy DOESNT WORK AAAAA
    cpu_model = init_model(config)

    cpu_model.load_state_dict(model.state_dict())

    # cpu_model = copy.deepcopy(model).cpu()
    merged_model = lora.merge_lora(cpu_model)

    # Push both model and processor
    merged_model.push_to_hub(
        f"{config.run_name}",
        commit_message=f"step {step}, run_id {config.run_id}",
    )
    processor.push_to_hub(
        f"{config.run_name}",
        commit_message=f"step {step}, run_id {config.run_id}",
    )


def train_step(
    state: TrainState, batch: dict, config: MyUltravoxTrainConfig
) -> tuple[TrainState, float, float]:
    """Execute one training step"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare batch
    input_ids, labels, audio_values, audio_token_start_idx, audio_token_len = (
        prepare_batch(batch, device)
    )

    # Compute loss
    loss = compute_loss(
        state.model,  # type: ignore
        input_ids=input_ids,
        labels=labels,
        audio_values=audio_values,
        audio_token_start_idx=audio_token_start_idx,
        audio_token_len=audio_token_len,
    )

    # Optimization step
    state.optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(state.model.parameters(), 0.3)
    state.optimizer.step()
    state.scheduler.step()

    return state, loss.item(), grad_norm.item()


def calculate_throughput(duration: float, batch: dict, hz: float) -> float:
    total_tokens = batch["audio_emb"].size(1) * batch["audio_emb"].size(0)
    seconds_of_audio = total_tokens / hz
    return seconds_of_audio / duration


def cleanup():
    utils.rank_0_only(wandb.finish)()
    utils.distributed_only(dist.destroy_process_group)()  # type: ignore


def init_wandb(model: UltravoxModel, config: MyUltravoxTrainConfig) -> None:
    wandb.init(
        config=config.model_dump()
        | {"model_config": utils.unwrap_model(model).config.to_dict()},
        id=config.run_id,
        resume="allow",
        dir=utils.get_log_dir(),
        project=config.wandb_project_name,
        name=config.run_name,
    )
    if config.watch_every is not None:
        wandb.watch(model, log_freq=config.watch_every)


def log_metrics(state: TrainState, stats: dict[str, float]) -> None:
    stats_tensor = torch.tensor(list(stats.values()), device=utils.get_device())  # type: ignore
    utils.distributed_only(dist.all_reduce)(stats_tensor, op=dist.ReduceOp.SUM)  # type: ignore
    stats_tensor = stats_tensor / stats_tensor[-1]
    stats = {k: v for k, v in zip(stats.keys(), stats_tensor.tolist())}
    stats = {f"ultravox/{k}": v for k, v in stats.items() if k != "count"} | {
        "ultravox/current_lr": state.scheduler.get_last_lr()[0]
    }

    utils.rank_0_only(wandb.log)(stats, step=state.step)


def create_loader(
    dataset: MyUltravoxDataset, config: MyUltravoxTrainConfig, **kwargs
) -> torch.utils.data.DataLoader:  # type: ignore
    loader_args = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": True,
    }
    loader_args.update(kwargs)
    return torch.utils.data.DataLoader(dataset, **loader_args)  # type: ignore


def compute_loss(
    ultravox: UltravoxModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    audio_values: torch.Tensor,
    audio_token_start_idx: torch.Tensor,
    audio_token_len: torch.Tensor,
) -> torch.Tensor:
    """Compute loss for one training step"""
    loss = ultravox(
        input_ids=input_ids,
        labels=labels,
        audio_values=audio_values,
        audio_token_start_idx=audio_token_start_idx,
        audio_token_len=audio_token_len,
    ).loss
    return loss


def train(config: MyUltravoxTrainConfig) -> None:
    utils.pprint(config.model_dump(), json=True)

    utils.distributed_only(dist.init_process_group)(
        "nccl",
        rank=config.rank,
        world_size=config.world_size,
        timeout=datetime.timedelta(seconds=3600),
    )
    utils.distributed_only(dist.barrier)()

    utils.set_seed(config.seed)
    state = init_train_state(config)

    if config.ckpt_path:
        state = load_train_state(state, config=config)

    # Print trainable parameters (LoRA only)
    lora_params = list(lora.get_lora_params(state.model))
    num_params = sum(p.numel() for _, p in lora_params)
    gb = sum(p.numel() * p.element_size() for _, p in lora_params) / 1024**3
    utils.pprint(
        f"trainable LoRA parameters: {num_params / 1e6:.2f}M | {gb:.2f}GB",
        color="bold cyan",
    )

    # Rest of the training setup remains the same
    if config.gradient_checkpointing:
        state.model.gradient_checkpointing_enable()

    if config.world_size > 1:
        state = state._replace(
            model=torch_pl.DistributedDataParallel(
                state.model, device_ids=[config.rank]
            )
        )

    utils.backend_flags()

    if config.compile:
        torch._dynamo.config.optimize_ddp = False
        state = state._replace(model=torch.compile(state.model, mode="reduce-overhead"))

    utils.rank_0_only(init_wandb)(state.model, config=config)

    data_collator = DataCollatorForSeq2SeqWithAudio(
        state.train_dataset.processor.tokenizer
    )

    train_loader = create_loader(
        state.train_dataset,
        config=config,
        collate_fn=data_collator,
    )

    stats = {
        "train_loss": 0.0,
        "grad_norm": 0.0,
        "count": 0.0,
    }

    while True:
        for batch in tqdm.tqdm(train_loader, colour="blue"):
            state = state._replace(step=state.step + 1)
            state, loss, norm = train_step(state, batch=batch, config=config)

            stats["count"] += 1
            stats["train_loss"] += loss
            stats["grad_norm"] += norm

            print(f"loss: {loss}")

            if state.step % 5 == 0:
                log_metrics(state, stats=stats)
                stats = {k: 0.0 for k in stats}

            if config.save_every and state.step % config.save_every == 0:
                utils.rank_0_only(save_train_state)(state, config=config)

            if config.push_every and state.step % config.push_every == 0:
                model = utils.unwrap_model(state.model)
                utils.rank_0_only(push_to_hub)(
                    model,
                    state.train_dataset.processor,
                    config=config,
                    step=state.step,
                )

            if config.test_every and state.step % config.test_every == 0:
                test(
                    utils.unwrap_model(state.model),
                    step=state.step,
                    config=config,
                    description="evaluation_default",
                )

            if state.step >= config.max_steps:
                utils.pprint("\nmax steps reached, exiting...", color="bold red")
                test(
                    utils.unwrap_model(state.model),
                    step=state.step,
                    config=config,
                )
                break

        utils.distributed_only(dist.barrier)()

    cleanup()
