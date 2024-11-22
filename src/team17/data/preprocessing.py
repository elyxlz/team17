import os
import random
from concurrent.futures import ThreadPoolExecutor

import dotenv
import numpy as np
import pydantic_settings as pyds
import torch
import torch.nn.functional as F
import torchaudio
import transformers
import whisperx
from torch.utils.data import IterableDataset
from tqdm import tqdm

from team17.whisper_encoder import ModifiedWhisperEncoder

dotenv.load_dotenv()


class PreprocessingConfig(pyds.BaseSettings):
    seed: int = 1338
    input_path: str = "./data/chunks"
    output_path: str = "./data/processed"
    transcription_model: str = "medium"  # Model for transcription
    embedding_model: str = "small"  # Model for embeddings
    compute_type: str = "float16"
    inner_batch_size: int = 4
    use_cuda: bool = torch.cuda.is_available()
    chunk_frames: int = 16_000 * 3
    sample_rate: int = 16_000
    num_workers: int = 0


class AudioChunkIterableDataset(IterableDataset):
    def __init__(
        self,
        input_path: str,
        target_sample_rate: int,
        chunk_frames: int,
        silence_threshold: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.input_path = input_path
        self.target_sample_rate = target_sample_rate
        self.chunk_frames = chunk_frames
        self.silence_threshold = silence_threshold
        self.file_list = []
        self._find_audio_files(input_path)
        random.seed(seed)
        random.shuffle(self.file_list)

    def _find_audio_files(self, directory):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(
                    (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".mp4", ".webm")
                ):
                    self.file_list.append(os.path.join(root, file))

    def _is_silent(self, waveform: torch.Tensor) -> bool:
        rms = torch.sqrt(torch.mean(waveform**2))
        return rms.item() < self.silence_threshold

    def __iter__(self):
        for file_path in self.file_list:
            try:
                waveform, sample_rate = torchaudio.load(file_path)

                if sample_rate != self.target_sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        sample_rate, self.target_sample_rate
                    )
                    waveform = resampler(waveform)

                # Convert to mono if needed
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # Skip if audio is silent
                if self._is_silent(waveform):
                    continue

                # Pad or cut to chunk_frames
                if waveform.shape[1] < self.chunk_frames:
                    waveform = F.pad(
                        waveform, (0, self.chunk_frames - waveform.shape[1])
                    )
                elif waveform.shape[1] > self.chunk_frames:
                    waveform = waveform[:, : self.chunk_frames]

                duration = waveform.shape[1] / self.target_sample_rate
                if duration > 30:
                    continue

                yield waveform, os.path.basename(file_path)

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue


def analyze_speakers_pronouns(transcript_data):
    PRONOUNS = {
        "first_person": {"i", "i'm", "i've", "i'll", "i'd", "me", "my", "mine"},
        "second_person": {
            "you",
            "you're",
            "you've",
            "you'll",
            "you'd",
            "your",
            "yours",
        },
    }

    def get_speakers():
        return {
            word["speaker"]: {"first_person": 0, "second_person": 0}
            for word in transcript_data["word_segments"]
            if word.get("speaker") is not None
        }

    def count_pronouns(stats):
        for word_data in transcript_data["word_segments"]:
            if (speaker := word_data.get("speaker")) is None:
                continue

            word = word_data["word"].lower().strip(".,!?")
            for pronoun_type, pronoun_set in PRONOUNS.items():
                if word in pronoun_set:
                    stats[speaker][pronoun_type] += 1
        return stats

    def find_best_speaker(stats):
        return max(
            stats.items(),
            key=lambda x: x[1]["first_person"] / (x[1]["second_person"] + 0.1),
            default=(None, {}),
        )[0]

    speaker_stats = count_pronouns(get_speakers())
    best_speaker = find_best_speaker(speaker_stats)

    return best_speaker, speaker_stats


def _save_processed_data(processed_data, filename, output_path):
    subdir = filename[:2]
    full_output_path = os.path.join(output_path, subdir)
    os.makedirs(full_output_path, exist_ok=True)
    output_file = os.path.join(full_output_path, f"{os.path.splitext(filename)[0]}.npz")
    np.savez(output_file, **processed_data)


@torch.no_grad()
def process_audio_chunks(config: PreprocessingConfig):
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    device = torch.device("cuda" if config.use_cuda else "cpu")

    # Load models
    asr_model = whisperx.load_model(
        config.transcription_model,
        device="cuda" if config.use_cuda else "cpu",
        compute_type="float32" if not config.use_cuda else config.compute_type,
        asr_options={"multilingual": False, "hotwords": []},
        language="en",
    )
    diarize_model = whisperx.DiarizationPipeline(
        device=device, use_auth_token=os.getenv("HF_TOKEN")
    )
    whisper_fe = transformers.WhisperFeatureExtractor.from_pretrained(
        f"openai/whisper-{config.embedding_model}",
        chunk_length=config.chunk_frames // config.sample_rate,
    )
    embedding_model = ModifiedWhisperEncoder.from_pretrained(
        f"openai/whisper-{config.embedding_model}"
    )
    embedding_model.to(device)  # type: ignore

    dataset = AudioChunkIterableDataset(
        config.input_path,
        target_sample_rate=config.sample_rate,
        chunk_frames=config.chunk_frames,
        seed=config.seed,
    )
    dl = iter(
        torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=config.num_workers
        )
    )

    save_executor = ThreadPoolExecutor(max_workers=config.num_workers + 1)

    def time_to_idx(time):
        return int(time * 50)

    while True:
        waveform, filename = next(dl)
        audio = waveform.squeeze().numpy()

        # Get transcription and alignment
        result = asr_model.transcribe(audio, batch_size=config.inner_batch_size)
        if result["language"] != "en":
            continue

        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device,  # type: ignore
        )

        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        num_speakers = len(set([i["speaker"] for i in result["segments"]]))
        if num_speakers != 2:
            continue

        input_features = whisper_fe(
            audio, sampling_rate=config.sample_rate, return_tensors="pt"
        ).to(device)["input_features"]
        embeddings = embedding_model(input_features).last_hidden_state.cpu().float16()

        best_speaker = analyze_speakers_pronouns(result)[0]
        if best_speaker is None:
            continue

        breakpoint()

        # Process segments
        sequence = []
        audio_emb_indices = []

        # Find first segment of best speaker
        start_idx = 0
        for i, segment in enumerate(result["segments"]):
            if segment["speaker"] == best_speaker:
                start_idx = i
                break

        segments = result["segments"][start_idx:]

        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]

            if segment["speaker"] == best_speaker:
                start_idx = time_to_idx(start_time)
                end_idx = time_to_idx(end_time)
                audio_emb_indices.extend(list(range(start_idx, end_idx)))
                for i in range(start_idx, end_idx):
                    sequence.append(f"<a{i}>")
            else:
                sequence.append(segment["text"])

        audio_emb_indices = torch.tensor(audio_emb_indices)
        selected_embeddings = embeddings[audio_emb_indices]

        processed_data = {
            "text": "".join(sequence),
            "audio_emb": selected_embeddings.cpu().numpy(),
        }

        save_executor.submit(
            _save_processed_data, processed_data, filename, config.output_path
        )

    save_executor.shutdown(wait=True)


if __name__ == "__main__":
    config = PreprocessingConfig()
    process_audio_chunks(config)
