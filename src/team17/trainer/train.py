import datetime
from enum import Enum
import dataclasses
import dotenv
import typing
import copy

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


def init_train_state(config: MyUltravoxTrainConfig) -> TrainState:
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

    train_loader = iter(
        create_loader(
            state.train_dataset,
            config=config,
            collate_fn=data_collator,
        )
    )

    stats = {
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

        train_bar.set_description(f"loss: {loss:.2f}")
        if state.step % 50 == 0:
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
