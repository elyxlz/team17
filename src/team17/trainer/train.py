import datetime
from enum import Enum
import dataclasses
import dotenv
import typing

import peft
import torch
import torch.distributed as dist
import torch.nn.parallel as torch_pl
import tqdm
import transformers as tr
import wandb

from team17.modeling.model import UltravoxConfig, UltravoxModel
from team17.modeling.processor import UltravoxProcessor
from team17.trainer import utils
from team17.trainer.config import MyUltravoxTrainConfig
from team17.trainer.dataset import MyUltravoxDataset, DataCollatorForSeq2SeqWithAudio
from team17.trainer.test import test

dotenv.load_dotenv()

__all__ = ["train"]


class TrainState(typing.NamedTuple):
    step: int
    model: UltravoxModel | peft.PeftModel  # type: ignore
    optimizer: torch.optim.AdamW  # type: ignore
    scheduler: torch.optim.lr_scheduler.LRScheduler  # type: ignore
    train_dataset: MyUltravoxDataset


@dataclasses.dataclass
class LoraConfigSimplified:
    """
    Low Rank Approximation (LoRA) configuration.

    Used for language and audio models separately.
    """

    # The rank of the approximation
    r: int = 0
    lora_alpha: float = 8
    target_modules: list[str] | None = dataclasses.field(
        default_factory=lambda: ["k_proj", "q_proj", "linear_k", "linear_q"]
    )


class LossFunction(str, Enum):
    CrossEntropy = "ce"
    KL_Divergence = "kl"


@dataclasses.dataclass
class LossConfig:
    loss_function: LossFunction = LossFunction.KL_Divergence
    kl_temperature: float = 2.0

    @property
    def requires_alt_fields(self):
        return self.loss_function == LossFunction.KL_Divergence


def create_lora_config(
    config: MyUltravoxTrainConfig,
    modules: list[str],  # , to_save: list[str]
) -> peft.LoraConfig:  # type: ignore
    return peft.LoraConfig(  # type: ignore
        task_type=peft.TaskType.CAUSAL_LM,  # type: ignore
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=modules,
        # modules_to_save=to_save,
        bias="none",
    )


def init_train_state(config: MyUltravoxTrainConfig) -> TrainState:
    device = utils.get_device()

    #
    # lora_config = LoraConfigSimplified(
    #     r=config.lora_r,
    #     lora_alpha=config.lora_alpha,
    #     target_modules=modules,
    # )

    if config.ultravox_pretrained_path is not None:
        model = UltravoxModel.from_pretrained(config.ultravox_pretrained_path)
        # model = UltravoxModel(UltravoxConfig(**config.ultravox_kwargs))
    else:
        model = UltravoxModel(
            UltravoxConfig(
                audio_model_id="openai/whisper-tiny",
                text_config=tr.LlamaConfig(
                    vocab_size=128128, hidden_size=64, num_hidden_layers=1
                ).to_dict(),  # **config.ultravox_kwargs),
                audio_latency_block_size=None,
            )
        )

    model.set_loss_config(LossConfig(LossFunction.CrossEntropy))
    # Prepare model for LoRA
    modules = [
        n
        for n, p in model.named_modules()
        if isinstance(p, torch.nn.Linear)
        and "head" not in n
        # and "cond_net" not in n
        and "emb" not in n
    ]
    # lora_config = create_lora_config(config, modules=modules)  # , to_save=to_save)
    # model = peft.get_peft_model(model, lora_config)  # type: ignore

    model = model.to(device, torch.bfloat16)

    # Filter trainable parameters for optimizer
    optimizer_params = [
        {
            "params": [p for n, p in model.named_parameters() if p.dim() > 1],
            "weight_decay": 0.1,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.dim() == 1],
            "weight_decay": 0.0,
        },
    ]

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(  # type: ignore
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

    # Initialize dataset
    processor = UltravoxProcessor.from_pretrained("fixie-ai/ultravox-v0_3-llama-3_2-1b")
    # processor = UltravoxProcessor.from_pretrained(
    #     "fixie-ai/ultravox-v0_4_1-llama-3_1-8b"
    # )
    processor.tokenizer.padding_side = "right"
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    train_dataset = MyUltravoxDataset(processor)

    breakpoint()
    processor.push_to_hub("banana1234")

    return TrainState(
        step=0,
        model=model,  # type: ignore
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


def cleanup():
    utils.rank_0_only(wandb.finish)()
    utils.distributed_only(dist.destroy_process_group)()  # type: ignore


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


def train(config: MyUltravoxTrainConfig) -> None:
    utils.pprint(config.model_dump(), json=True)

    utils.distributed_only(dist.init_process_group)(
        "nccl",
        rank=config.rank,
        world_size=config.world_size,
        timeout=datetime.timedelta(seconds=3600),
    )  # type: ignore
    utils.distributed_only(dist.barrier)()  # type: ignore

    utils.set_seed(config.seed)
    state = init_train_state(config)

    if config.ckpt_path:
        state = load_train_state(state, config=config)

    # print trainable parameters
    num_params, gb = utils.trainable_params(state.model)
    utils.pprint(
        f"trainable parameters: {num_params / 1e6:.2f}M | {gb:.2f}GB", color="bold cyan"
    )

    # gradient checkpointing
    if config.gradient_checkpointing:
        state.model.gradient_checkpointing_enable()

    # ddp
    if config.world_size > 1:
        state = state._replace(
            model=torch_pl.DistributedDataParallel(
                state.model, device_ids=[config.rank]
            )
        )

    # backend flags
    utils.backend_flags()

    # compile
    if config.compile:
        torch._dynamo.config.optimize_ddp = False  # type: ignore
        state = state._replace(model=torch.compile(state.model, mode="reduce-overhead"))  # type: ignore

    # wandb
    utils.rank_0_only(init_wandb)(state.model, config=config)

    data_collator = DataCollatorForSeq2SeqWithAudio(
        state.train_dataset.processor.tokenizer
    )

    # train loader
    train_loader = iter(
        create_loader(
            state.train_dataset,
            config=config,
            collate_fn=data_collator,
        )
    )

    # stats
    stats: dict[str, float] = {
        "train_loss": 0.0,
        "grad_norm": 0.0,
        "count": 0.0,
    }

    train_bar = tqdm.trange(
        len(state.train_dataset) - state.step,
        initial=state.step,
        total=config.max_steps,
        colour="blue",
        disable=config.rank > 0,
    )

    for _ in train_bar:
        try:
            batch = next(train_loader)
        except StopIteration:
            continue

        state = state._replace(step=state.step + 1)
        state, loss, norm = train_step(state, batch=batch, config=config)

        stats["count"] += 1
        stats["train_loss"] += loss
        stats["grad_norm"] += norm

        # log
        train_bar.set_description(f"loss: {loss:.2f}")
        if state.step % 50 == 0:
            log_metrics(state, stats=stats)
            stats_tensor = torch.tensor(list(stats.values()))
            stats = {
                k: v
                for k, v in zip(stats.keys(), torch.zeros_like(stats_tensor).tolist())
            }

        if config.save_every and state.step % config.save_every == 0:
            utils.rank_0_only(save_train_state)(state, config=config)

        if config.push_every and state.step % config.push_every == 0:
            model = utils.unwrap_model(state.model)
            utils.rank_0_only(model.push_to_hub)(
                f"Audiogen/{config.run_name}",
                commit_message=f"step {state.step}, run_id {config.run_id}",
                private=True,
            )

        if config.test_every and state.step % config.test_every == 0:
            test(
                utils.unwrap_model(state.model),
                step=state.step,
                config=config,
                description="evaluation_default",
            )
        if state.step >= config.max_steps:
            utils.pprint("\nmax steps or minutes reached, exiting...", color="bold red")
            test(
                utils.unwrap_model(state.model),
                step=state.step,
                config=config,
            )
            break

        utils.distributed_only(dist.barrier)()  # type: ignore

    cleanup()
