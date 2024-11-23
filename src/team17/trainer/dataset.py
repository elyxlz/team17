import os
import transformers
import dataclasses
import json
from typing import Dict, List, Optional, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from team17.trainer.voice_sample import VoiceSample
from team17.modeling.processor import UltravoxProcessor


class MyUltravoxDataset(Dataset):
    def __init__(
        self,
        processor: UltravoxProcessor,
        data_dir: str = "./data/processed",
        train_on_inputs: bool = False,
        inference_mode: bool = False,
    ):
        """
        Dataset for loading preprocessed data and preparing it for Ultravox using the processor.

        Args:
            data_dir: Directory containing processed data (with subdirs)
            processor: UltravoxProcessor for processing text and audio
            train_on_inputs: If True, include prompt tokens in loss calculation
            inference_mode: If True, only use input message (for generation)
        """
        super().__init__()
        self.processor = processor
        self.train_on_inputs = train_on_inputs
        self.inference_mode = inference_mode
        if self.inference_mode:
            self.train_on_inputs = True

        # Find all .npz files
        self.samples = []
        for subdir in os.listdir(data_dir):
            subdir_path = os.path.join(data_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            for filename in os.listdir(subdir_path):
                if filename.endswith(".npz"):
                    file_path = os.path.join(subdir_path, filename)
                    self.samples.append(file_path)

        print(f"Found {len(self.samples)} samples in {data_dir}")

    def _load_npz_as_voice_sample(self, npz_path: str) -> VoiceSample:
        """Load a .npz file into a VoiceSample object."""
        data = np.load(npz_path)

        # Load audio
        audio = data["audio"]

        # Convert conversation bytes back to dict
        conv_bytes = data["conversation"].tobytes()
        conv_data = json.loads(conv_bytes.decode("utf-8"))

        # Create voice sample
        return VoiceSample(
            messages=conv_data["messages"],
            audio=audio[0][0],
            sample_rate=conv_data["sample_rate"],
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Load the sample from .npz file
        sample = self._load_npz_as_voice_sample(self.samples[idx])

        # TODO: only get first 2 messages cuz i dont think the processor can even do multiturn????
        sample.messages = sample.messages[:2]

        # For inference, remove assistant message
        if self.inference_mode:
            sample.messages = sample.messages[:-1]

        # Get conversation text
        text = self.processor.tokenizer.apply_chat_template(
            sample.messages, tokenize=False
        )

        # Process using UltravoxProcessor
        # Audio needs to be [C x M] where C=1 for mono
        #
        audio = (
            np.expand_dims(sample.audio, axis=0) if sample.audio is not None else None
        )
        inputs = self.processor(
            text=text,
            audio=audio,
            return_tensors="pt",
            sampling_rate=sample.sample_rate,
        )

        # Squeeze extra dimensions from processor output
        input_ids = inputs["input_ids"].squeeze_(0)
        inputs["attention_mask"].squeeze_(0)
        if "audio_values" in inputs:
            inputs["audio_values"].squeeze_(0)
            inputs["audio_token_start_idx"].squeeze_(0)
            inputs["audio_token_len"].squeeze_(0)
            inputs["audio_len"].squeeze_(0)

        # Create labels (model handles shifting internally)
        labels = input_ids.clone()

        if not self.train_on_inputs:
            # Only compute loss on assistant responses
            input_text = self.processor.tokenizer.apply_chat_template(
                sample.messages[:-1], tokenize=False
            )
            input_token_len = self.processor(
                text=input_text,
                audio=audio,
                sampling_rate=sample.sample_rate,
            )["input_ids"].shape[-1]
            labels[:input_token_len] = -100

        return {
            **inputs,
            "labels": labels,
        }


def create_dataloaders(
    data_dir: str,
    processor: UltravoxProcessor,
    batch_size: int = 8,
    num_workers: int = 4,
    **dataset_kwargs,
) -> torch.utils.data.DataLoader:
    """Create a dataloader for the Ultravox dataset."""
    dataset = MyUltravoxDataset(
        data_dir=data_dir, processor=processor, **dataset_kwargs
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader


@dataclasses.dataclass
class DataCollatorForSeq2SeqWithAudio(transformers.DataCollatorForSeq2Seq):
    def __call__(self, features, *args, **kwargs):
        audio_values = [f.pop("audio_values", None) for f in features]
        input_ids_lens = torch.LongTensor([f["input_ids"].shape[-1] for f in features])
        batch = super().__call__(features, *args, **kwargs)
        # Pad the last dimension of all audio_values to the same length, with 0s on the right.
        if audio_values and audio_values[0] is not None:
            max_len = max([x.shape[-1] for x in audio_values])
            batch["audio_values"] = torch.stack(
                [
                    torch.nn.functional.pad(x, (0, max_len - x.shape[-1]))
                    for x in audio_values
                ]
            )
            if self.tokenizer.padding_side == "left":
                displacement = batch["input_ids"].shape[-1] - input_ids_lens
                batch["audio_token_start_idx"] += displacement.to(
                    batch["audio_token_start_idx"].device
                )

        return batch


# Usage example:
if __name__ == "__main__":
    import torch
    from ultravox.model import processor

    # Initialize processor
    processor = UltravoxProcessor.from_pretrained("path/to/processor")

    # Create dataloader
    dataloader = create_dataloaders(
        data_dir="path/to/processed/data",
        processor=processor,
        batch_size=4,
        train_on_inputs=False,
        include_alt_fields=False,
    )

    # Test one batch
    batch = next(iter(dataloader))
    print("Batch keys:", batch.keys())
    print("Input IDs shape:", batch["input_ids"].shape)
    print("Audio values shape:", batch["audio_values"].shape)
    print("Number of audio segments:", len(batch["audio_token_start_idx"]))
