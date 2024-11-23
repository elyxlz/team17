import random
import typing

import audiogen_io
import numpy as np
import torch
import torch.utils.data as td
import transformers as tr

from team17.trainer import utils
from team17.trainer.config import MyUltravoxTrainConfig

torch.multiprocessing.set_sharing_strategy("file_system")


def get_infos_from_query(query: dict, test: bool, local_mode: bool) -> list[dict]:
    if local_mode:
        audiogen_io.set_local_mode(True)

    query["test_set"] = test
    selector = audiogen_io.select(query).apply()
    out = [i.item for i in selector]  # type: ignore
    assert len(out) > 0
    return out


def get_fake_field(model_config: UltravoxConfig, test: bool) -> dict:
    out = {}
    out["audio_emb"] = torch.randn(154, model_config.audio_emb_dim)
    out["input_ids"] = torch.randint(0, 128, (64,))
    out["attention_mask"] = torch.randint(0, 2, (64,))
    out["text_raw"] = "banana"

    if test:
        out["audio_raw"] = torch.randn(2, 480_000)

    return out


def get_payload_from_info(info: dict) -> dict:
    ioitem = audiogen_io.io_item.IoItem(info)  # type: ignore
    payload = np.load(ioitem.download())
    return payload


def get_real_field(
    payload: dict,
    info: dict,
    audio_emb_type: str,
    tokenizer: tr.AutoTokenizer,
    text_max_length: int,
) -> dict:
    field = {}
    assert (
        audio_emb_type in payload.keys()
    ), f"{audio_emb_type} not in payload: {payload.keys()}"
    assert "text_raw" in info.keys()

    field.update(
        {"audio_emb": utils.misc.numpy_array_to_torch(payload[audio_emb_type])}
    )
    field.update(
        {k: v.unsqueeze(0) for k, v in field.items() if v.dim() == 1}
    )  # 2 dim always

    if "audio_raw" in payload.keys():
        field.update(
            {"audio_raw": utils.misc.numpy_array_to_torch(payload["audio_raw"])}
        )

    text = info["text_raw"]
    text = text + tokenizer.eos_token  # type: ignore
    r = tokenizer(
        text=text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=text_max_length,
        add_special_tokens=True,
    )  # type: ignore

    field["text_raw"] = info["text_raw"]
    field["input_ids"], field["attention_mask"] = (
        r["input_ids"][0],
        r["attention_mask"][0],
    )

    return field


def get_field(
    idx: int,
    config: MyUltravoxTrainConfig,
    model_config: MyUltravoxConfig,
    infos: list[dict],
    tokenizer: tr.AutoTokenizer,
    test: bool,
) -> dict[str, typing.Any]:
    try:
        if config.fake:
            item = get_fake_field(model_config, test=test)
        else:
            payload = get_payload_from_info(infos[idx])
            item = get_real_field(
                payload,
                info=infos[idx],
                audio_emb_type=config.audio_emb_type,
                tokenizer=tokenizer,
                text_max_length=config.text_max_length,
            )

        return item

    except Exception as e:
        utils.misc.pprint(f"error loading item at index {idx}: {e}", color="bold red")
        g = torch.Generator().manual_seed(idx)
        new_idx = int(torch.randint(0, len(infos or []), (1,), generator=g).item())
        return get_field(
            new_idx,
            config=config,
            model_config=model_config,
            infos=infos,
            tokenizer=tokenizer,
            test=test,
        )


class MyUltravoxDataset(td.IterableDataset):
    def __init__(
        self,
        config: MyUltravoxTrainConfig,
        model_config: MyUltravoxConfig,
        test: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.model_config = model_config
        self.test = test

        self.data_step = 0
        self.rank = config.rank
        self.world_size = config.world_size

        self.tokenizer = tr.AutoTokenizer.from_pretrained(
            config.text_tokenizer_path, add_eos_token=True
        )
        self.tokenizer.pad_token_id = 128004
        if config.fake:
            self.infos = [{}]

        self.infos = get_infos_from_query(
            config.query, test=test, local_mode=config.io_local_mode
        )

        if config.shuffle and not test:
            random.seed(config.seed)
            random.shuffle(self.infos)

        if config.io_blob_cache:
            audiogen_io.set_blob_cache(enabled=True)

    def __len__(self) -> int:
        if self.config.fake:
            return 100_000
        else:
            assert len(self.infos) > 0
            return len(self.infos)

    def __iter__(self) -> typing.Iterator[dict[str, torch.Tensor]]:
        worker_info = td.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        stride = num_workers * self.world_size
        offset = self.rank * num_workers + worker_id

        while True:
            self.data_step += stride
            idx = (offset + self.data_step) % len(self)
            yield get_field(
                idx,
                config=self.config,
                model_config=self.model_config,
                infos=self.infos,
                tokenizer=self.tokenizer,
                test=self.test,
            )


# import os
# import random
# import typing
#
# import numpy as np
# import torch
# import torch.utils.data as td
#
# from .config import VoxtralTrainConfig
#
#
# def get_npy_files(path: str) -> list[str]:
#     npy_files: list[str] = []
#     for root, _, files in os.walk(path):
#         for file in files:
#             if file.endswith(".npy"):
#                 npy_files.append(os.path.join(root, file))
#     return npy_files
#
#
# def get_fake_item() -> dict[str, torch.Tensor]:
#     return {"tokens": torch.randint(0, 1000, (220,))}
#
#
# def get_item(file_path: str) -> dict[str, torch.Tensor]:
#     try:
#         npy_data = np.load(file_path)
#         item: dict[str, torch.Tensor] = {}
#
#         item["tokens"] = torch.from_numpy(npy_data)
#
#         if item["tokens"].dim() == 2:
#             item["tokens"] = item["tokens"].squeeze()
#
#         return item
#     except Exception as e:
#         print(f"Error loading file {file_path}: {str(e)}")
#         # Generate a fake item as a fallback
#         return get_fake_item()
#
#
# class VoxtralDataset(td.IterableDataset):
#     config: VoxtralTrainConfig
#     data_step: int
#     rank: int
#     world_size: int
#     file_paths: list[str]
#
#     def __init__(self, config: VoxtralTrainConfig) -> None:
#         super().__init__()
#         self.config = config
#         self.data_step = 0
#         self.rank = config.rank
#         self.world_size = config.world_size
#         self.fake = config.fake
#         self.overfit = config.overfit
#
#         if self.fake:
#             self.file_paths = []
#         else:
#             self.file_paths = get_npy_files(config.data_path)
#
#         if not self.fake:
#             random.seed(config.seed)
#             random.shuffle(self.file_paths)
#             print(f"Total number of NPZ files: {len(self.file_paths)}")
#
#     def __len__(self) -> int:
#         if self.fake:
#             return 100_000
#         else:
#             assert len(self.file_paths) > 0
#             return len(self.file_paths)
#
#     def __iter__(self) -> typing.Iterator[dict[str, torch.Tensor]]:
#         worker_info = td.get_worker_info()
#         worker_id = worker_info.id if worker_info else 0
#         num_workers = worker_info.num_workers if worker_info else 1
#         stride = num_workers * self.world_size
#         offset = self.rank * num_workers + worker_id
#
#         while True:
#             self.data_step += stride
#             idx = (offset + self.data_step) % len(self)
#             if self.fake:
#                 yield get_fake_item()
#             else:
#                 if self.overfit is not None:
#                     idx = (idx % self.overfit) * 1_000
#                 yield get_item(self.file_paths[idx])
