import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import dotenv
import numpy as np
import pydantic_settings as pyds
import torch
import torchaudio
import transformers
import whisperx
from torch.utils.data import Dataset
from tqdm import tqdm

from team17 import utils
from team17.whisper_encoder import ModifiedWhisperEncoder

dotenv.load_dotenv()


class PreprocessingConfig(pyds.BaseSettings):
    input_path: str = "./data/chunks"
    output_path: str = "./data/processed"
    transcription_model: str = "medium"  # Model for transcription
    embedding_model: str = "small"  # Model for embeddings
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    compute_type: str = "float16"
    use_cuda: bool = torch.cuda.is_available()
    chunk_frames: int = 16_000 * 20
    sample_rate: int = 16_000
    max_save_workers: int = 16


class AudioChunkDataset(Dataset):
    def __init__(
        self, input_path: str, target_sample_rate: int, chunk_frames: int
    ) -> None:
        self.input_path = input_path
        self.target_sample_rate = target_sample_rate
        self.chunk_frames = chunk_frames
        self.file_list = []
        self._find_audio_files(input_path)

    def _find_audio_files(self, directory):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(
                    (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".mp4", ".webm")
                ):
                    self.file_list.append(os.path.join(root, file))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(file_path)

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, self.target_sample_rate
            )
            waveform = resampler(waveform)

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or cut to chunk_frames
        if waveform.shape[1] < self.chunk_frames:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.chunk_frames - waveform.shape[1])
            )
        elif waveform.shape[1] > self.chunk_frames:
            waveform = waveform[:, : self.chunk_frames]

        duration = waveform.shape[1] / self.target_sample_rate
        if duration > 30:
            raise ValueError(
                f"Audio duration {duration:.2f}s exceeds maximum allowed duration of 30s"
            )

        return waveform, os.path.basename(file_path)


def _save_processed_data(
    embeddings: torch.Tensor,
    speaker_segments: Dict[str, Dict],
    filename: str,
    output_path: str,
):
    subdir = filename[:2]
    full_output_path = os.path.join(output_path, subdir)
    os.makedirs(full_output_path, exist_ok=True)
    output_file = os.path.join(full_output_path, f"{os.path.splitext(filename)[0]}.npz")

    np.savez(output_file, embeddings=embeddings.cpu().numpy(), **speaker_segments)


@torch.no_grad()
def process_audio_chunks(config: PreprocessingConfig):
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    # Initialize models
    device = torch.device("cuda" if config.use_cuda else "cpu")

    # Load transcription model
    asr_model = whisperx.load_model(
        config.transcription_model,
        device="cuda" if config.use_cuda else "cpu",
        compute_type="float32" if not config.use_cuda else config.compute_type,
        asr_options={
            "multilingual": False,
            "hotwords": [],
        },
        language="en",
    )

    # Load alignment model (will be loaded based on detected language)
    diarize_model = whisperx.DiarizationPipeline(
        device=device, use_auth_token=os.getenv("HF_TOKEN")
    )

    # Load embedding model
    whisper_fe = transformers.WhisperFeatureExtractor.from_pretrained(
        f"openai/whisper-{config.embedding_model}",
        chunk_length=config.chunk_frames // config.sample_rate,
    )
    embedding_model = ModifiedWhisperEncoder.from_pretrained(
        f"openai/whisper-{config.embedding_model}"
    )
    embedding_model.to(device)

    dataset = AudioChunkDataset(
        config.input_path,
        target_sample_rate=config.sample_rate,
        chunk_frames=config.chunk_frames,
    )

    save_executor = ThreadPoolExecutor(max_workers=config.max_save_workers)

    for idx in tqdm(range(len(dataset)), desc="Processing audio chunks"):
        waveform, filename = dataset[idx]
        audio = waveform.squeeze().numpy()

        # Get transcription and alignment
        result = asr_model.transcribe(audio, batch_size=1)
        language = result["language"]

        if language != "en":
            print(f"skipping weird language {language}")
            continue

        # Load language-specific alignment model
        align_model, metadata = whisperx.load_align_model(
            language_code=language, device=device
        )
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, device
        )

        # Get speaker diarization
        diarize_segments = diarize_model(
            audio, min_speakers=config.min_speakers, max_speakers=config.max_speakers
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Get audio embeddings
        input_features = whisper_fe(
            audio, sampling_rate=config.sample_rate, return_tensors="pt"
        ).to(device)["input_features"]
        embeddings = embedding_model(input_features).last_hidden_state

        utils.play_audio(waveform)
        breakpoint()

        # Organize segments by speaker
        speaker_segments = {}
        for segment in result["segments"]:
            speaker = segment["speaker"]
            if speaker not in speaker_segments:
                speaker_segments[speaker] = {"transcripts": [], "timestamps": []}
            speaker_segments[speaker]["transcripts"].append(segment["text"])
            speaker_segments[speaker]["timestamps"].append(
                [segment["start"], segment["end"]]
            )

        save_executor.submit(
            _save_processed_data,
            embeddings,
            speaker_segments,
            filename,
            config.output_path,
        )

    save_executor.shutdown(wait=True)


if __name__ == "__main__":
    config = PreprocessingConfig()
    process_audio_chunks(config)
