import os
import random
import typing
import re
from pathlib import Path

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

        # Validate inputs
        if not fake and not os.path.exists(data_path):
            raise ValueError(f"Data path does not exist: {data_path}")

        # Initialize tokenizer
        self.tokenizer = tr.AutoTokenizer.from_pretrained(
            text_tokenizer_path, add_eos_token=True
        )
        self.tokenizer.pad_token_id = 128004
        self.audio_token_id = self.tokenizer.pad_token_id

        if self.fake:
            self.file_paths = []
        else:
            self.file_paths = self._get_npy_files(data_path)
            if not self.file_paths:
                raise ValueError(f"No .npy files found in {data_path}")
            random.seed(seed)
            random.shuffle(self.file_paths)
            print(f"Total number of NPY files: {len(self.file_paths)}")

    def _get_npy_files(self, path: str) -> list[str]:
        """Get all .npy files from the directory."""
        npy_files = list(Path(path).rglob("*.npy"))
        return [str(f) for f in npy_files]

    def _extract_audio_segments(self, text: str) -> tuple[list[tuple[int, int]], str]:
        """Extract audio token indices and clean text from the input string."""
        pattern = r"<a(\d+)>"
        audio_segments = []
        current_pos = 0
        cleaned_text = ""

        # Find all audio token markers
        matches = list(re.finditer(pattern, text))

        if not matches:
            return [], text

        for i, match in enumerate(matches):
            start_idx = int(match.group(1))

            # Add text before the marker
            cleaned_text += text[current_pos : match.start()]
            # Add a special token placeholder
            cleaned_text += self.tokenizer.pad_token

            # Find the end index from next marker or estimate it
            if i < len(matches) - 1:
                end_idx = int(matches[i + 1].group(1))
            else:
                end_idx = start_idx + 1

            if start_idx < 0 or end_idx < start_idx:
                raise ValueError(
                    f"Invalid audio segment indices: {start_idx}, {end_idx}"
                )

            audio_segments.append((start_idx, end_idx))
            current_pos = match.end()

        # Add remaining text
        cleaned_text += text[current_pos:]

        return audio_segments, cleaned_text

    def _get_fake_item(self) -> dict[str, torch.Tensor | str]:
        """Generate fake data with proper audio embedding structure."""
        # Create fake text with audio token markers
        fake_text = "This is <a0>fake audio<a10> sample <a20>text<a30>"
        audio_segments, cleaned_text = self._extract_audio_segments(fake_text)

        # Tokenize cleaned text
        text_tokens = self._tokenize_text(cleaned_text)

        # Create fake audio embeddings (30 tokens, 768-dim)
        fake_audio_emb = torch.randn(30, 768, dtype=torch.float32)

        # Create audio token information
        audio_token_start_idx = torch.tensor(
            [seg[0] for seg in audio_segments], dtype=torch.long
        )
        audio_token_len = torch.tensor(
            [seg[1] - seg[0] for seg in audio_segments], dtype=torch.long
        )

        return {
            "input_ids": text_tokens["input_ids"][0],
            "attention_mask": text_tokens["attention_mask"][0],
            "text_raw": cleaned_text,
            "audio_emb": fake_audio_emb,
            "audio_token_start_idx": audio_token_start_idx,
            "audio_token_len": audio_token_len,
        }

    def _tokenize_text(self, text: str) -> dict[str, torch.Tensor]:
        """Tokenize text."""
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
        """Load and process a single data item."""
        # Load NPY file
        npy_data = np.load(file_path, allow_pickle=True)

        # Extract text and audio embeddings
        text = npy_data["text"] if isinstance(npy_data, np.ndarray) else str(npy_data)
        if not isinstance(text, str):
            text = str(text)

        audio_emb = npy_data["audio_emb"]
        if not isinstance(audio_emb, np.ndarray):
            raise ValueError("Expected numpy array for audio_emb")

        audio_emb = torch.from_numpy(audio_emb).to(torch.float32)
        if audio_emb.dim() != 2 or audio_emb.size(1) != 768:
            raise ValueError(f"Invalid audio embedding shape: {audio_emb.shape}")

        # Process text and extract audio segments
        audio_segments, cleaned_text = self._extract_audio_segments(text)

        # Tokenize cleaned text
        text_tokens = self._tokenize_text(cleaned_text)

        # Prepare audio information
        audio_token_start_idx = torch.tensor(
            [seg[0] for seg in audio_segments], dtype=torch.long
        )
        audio_token_len = torch.tensor(
            [seg[1] - seg[0] for seg in audio_segments], dtype=torch.long
        )

        # Validate output tensors
        if len(audio_segments) > 0:
            if audio_token_start_idx.max() >= audio_emb.size(0):
                raise ValueError("Audio index out of bounds")
            if (audio_token_start_idx + audio_token_len).max() > audio_emb.size(0):
                raise ValueError("Audio length out of bounds")

        return {
            "input_ids": text_tokens["input_ids"][0],
            "attention_mask": text_tokens["attention_mask"][0],
            "text_raw": cleaned_text,
            "audio_emb": audio_emb,
            "audio_token_start_idx": audio_token_start_idx,
            "audio_token_len": audio_token_len,
        }

    def __len__(self) -> int:
        if self.fake:
            return 100_000
        else:
            assert len(self.file_paths) > 0
            return len(self.file_paths)

    def __iter__(self) -> typing.Iterator[dict[str, torch.Tensor | str]]:
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
                continue

            if self.overfit is not None:
                idx = (idx % self.overfit) * 1_000

            # Keep trying files until we get a valid one
            while True:
                try:
                    item = self._get_item(self.file_paths[idx])
                    yield item
                    break
                except Exception as e:
                    print(f"Error loading file {self.file_paths[idx]}: {str(e)}")
                    self.data_step += stride
                    idx = (offset + self.data_step) % len(self)
