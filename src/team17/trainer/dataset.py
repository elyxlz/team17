import os
import random
import typing

import numpy as np
import torch
import torch.utils.data as td
import transformers as tr


class MyUltravoxDataset(td.IterableDataset):
    def __init__(
        self,
        data_path: str,
        text_tokenizer_path: str,
        text_max_length: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        fake: bool = False,
        overfit: int | None = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.text_max_length = text_max_length
        self.data_step = 0
        self.rank = rank
        self.world_size = world_size
        self.fake = fake
        self.overfit = overfit

        # Initialize tokenizer
        self.tokenizer = tr.AutoTokenizer.from_pretrained(
            text_tokenizer_path, add_eos_token=True
        )
        self.tokenizer.pad_token_id = 128004

        if self.fake:
            self.file_paths = []
        else:
            self.file_paths = self._get_npy_files(data_path)
            random.seed(seed)
            random.shuffle(self.file_paths)
            print(f"Total number of NPY files: {len(self.file_paths)}")

    def _get_npy_files(self, path: str) -> list[str]:
        npy_files: list[str] = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".npy"):
                    npy_files.append(os.path.join(root, file))
        return npy_files

    def _get_fake_item(self) -> dict[str, torch.Tensor | str]:
        fake_text = "This is a fake sample text"
        text_tokens = self._tokenize_text(fake_text)

        return {
            "tokens": torch.randint(0, 1000, (220,)),
            "text_raw": fake_text,
            "input_ids": text_tokens["input_ids"],
            "attention_mask": text_tokens["attention_mask"],
        }

    def _tokenize_text(self, text: str) -> dict[str, torch.Tensor]:
        text = text + self.tokenizer.eos_token
        return self.tokenizer(
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
            add_special_tokens=True,
        )

    def _get_item(self, file_path: str) -> dict[str, torch.Tensor | str]:
        try:
            npy_data = np.load(file_path, allow_pickle=True)
            item: dict[str, torch.Tensor] = {}

            # Handle the base tokens
            tokens = torch.from_numpy(
                npy_data["tokens"] if isinstance(npy_data, np.ndarray) else npy_data
            )
            if tokens.dim() == 2:
                tokens = tokens.squeeze()
            item["tokens"] = tokens

            # Handle the text tokenization
            text = (
                npy_data["text"] if isinstance(npy_data, np.ndarray) else str(npy_data)
            )
            text_tokens = self._tokenize_text(text)

            item["text_raw"] = text
            item["input_ids"] = text_tokens["input_ids"][0]
            item["attention_mask"] = text_tokens["attention_mask"][0]

            return item

        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            return self._get_fake_item()

    def __len__(self) -> int:
        if self.fake:
            return 100_000
        else:
            assert len(self.file_paths) > 0
            return len(self.file_paths)

    def __iter__(self) -> typing.Iterator[dict[str, torch.Tensor]]:
        worker_info = td.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        stride = num_workers * self.world_size
        offset = self.rank * num_workers + worker_id

        while True:
            self.data_step += stride
            idx = (offset + self.data_step) % len(self)
            if self.fake:
                yield self._get_fake_item()
            else:
                if self.overfit is not None:
                    idx = (idx % self.overfit) * 1_000
                yield self._get_item(self.file_paths[idx])
