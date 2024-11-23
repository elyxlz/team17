import itertools

import torch
import torch.distributed as dist
import tqdm
import transformers as tr

from team17.modeling.model import UltravoxModel
from team17.trainer import logging, utils
from team17.trainer.config import MyUltravoxTrainConfig
from team17.trainer.dataset import MyUltravoxDataset


def init_eval_environment(
    model: UltravoxModel, config: MyUltravoxTrainConfig, batch_size: int
) -> tuple:
    from team17.trainer.train import create_loader

    test_dataset = MyUltravoxDataset(config, model_config=model.config, test=True)  # type: ignore
    test_loader = create_loader(
        test_dataset,
        config,
        batch_size=batch_size,
        num_workers=min(4, config.num_workers),
        prefetch_factor=1 if config.num_workers > 0 else None,
        pin_memory=False,
    )
    device = utils.get_device()
    model = model.to(device)  # type: ignore

    tokenizer = tr.AutoTokenizer.from_pretrained(config.text_tokenizer_path)

    return test_loader, model, tokenizer, device


def log_eval_results(
    audio: torch.Tensor,
    real_texts: list[str],
    generated_texts: list[str],
    sample_rate: int,
    step: int,
    desc: str,
) -> None:
    random_idxs = torch.randperm(len(audio))[:32].tolist()
    log_dict = {
        "audio": [audio[i] for i in random_idxs],
        "real_text": [real_texts[i] for i in random_idxs],
        "generated_text": [generated_texts[i] for i in random_idxs],
    }

    utils.rank_0_only(logging.log_wandb_data)(
        elements=log_dict,
        flags={
            "audio": ["audio", "mel_spectrogram"],
            "real_text": ["string"],
            "generated_text": ["string"],
        },
        sample_rate=sample_rate,
        step=step,
        name=desc,
    )


@utils.general_exception_handler
@torch.no_grad()
def test(
    model: UltravoxModel,
    step: int,
    config: MyUltravoxTrainConfig,
) -> None:
    from team17.trainer.train import prepare_batch

    utils.distributed_only(dist.barrier)()  # type: ignore

    batch_size = min(config.batch_size, 16)  # avoid oom
    total_batches = max(config.test_size // batch_size // config.world_size, 1)
    step_for_audio_logging = torch.randint(0, total_batches, (1,)).item()

    test_loader, model, evaluator, tokenizer, device = init_eval_environment(
        model=model, config=config, batch_size=batch_size
    )

    for n, batch in enumerate(
        tqdm.tqdm(
            itertools.islice(test_loader, total_batches),
            total=total_batches,
            leave=False,
            colour="magenta",
            disable=config.rank > 0,
        )
    ):
        audio_emb = prepare_batch(batch, device=device)[2]
        batch = {
            k: v.to(device, audio_emb.dtype)
            if isinstance(v, torch.Tensor) and "raw" not in k
            else v
            for k, v in batch.items()
        }

        gen_tokens = model.generate_from_audio(
            audio_emb=audio_emb,
            **config.generate_kwargs,
        )
        gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        evaluator.fill(
            audio=batch["audio_raw"], text=gen_text, sample_rate=config.sample_rate
        )

        if n == step_for_audio_logging:
            log_eval_results(
                audio=batch["audio_raw"],
                real_texts=batch["text_raw"],  # type: ignore
                generated_texts=gen_text,
                sample_rate=48000,
                step=step,
            )
