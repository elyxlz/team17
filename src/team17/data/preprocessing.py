import os
import random
from concurrent.futures import ThreadPoolExecutor

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
    seed: int = 1338
    input_path: str = "./data/chunks"
    output_path: str = "./data/processed"
    transcription_model: str = "medium"  # Model for transcription
    embedding_model: str = "small"  # Model for embeddings
    compute_type: str = "float16"
    inner_batch_size: int = 4
    use_cuda: bool = torch.cuda.is_available()
    chunk_frames: int = 16_000 * 20
    sample_rate: int = 16_000
    max_save_workers: int = 16


class AudioChunkDataset(Dataset):
    def __init__(
        self, input_path: str, target_sample_rate: int, chunk_frames: int, seed: int
    ) -> None:
        self.input_path = input_path
        self.target_sample_rate = target_sample_rate
        self.chunk_frames = chunk_frames
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


def _save_processed_data(
    embeddings: torch.Tensor,
    speaker_segments: dict[str, dict],
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
        asr_options={"multilingual": False, "hotwords": []},
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
    embedding_model.to(device)  # type: ignore

    dataset = AudioChunkDataset(
        config.input_path,
        target_sample_rate=config.sample_rate,
        chunk_frames=config.chunk_frames,
        seed=config.seed,
    )

    save_executor = ThreadPoolExecutor(max_workers=config.max_save_workers)

    for idx in tqdm(range(len(dataset)), desc="Processing audio chunks"):
        waveform, filename = dataset[idx]
        audio = waveform.squeeze().numpy()

        # Get transcription and alignment
        result = asr_model.transcribe(audio, batch_size=config.inner_batch_size)
        language = result["language"]

        if language != "en":
            print(f"skipping weird language {language}")
            continue

        # Load language-specific alignment model
        align_model, metadata = whisperx.load_align_model(
            language_code=language, device=device
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device,  # type: ignore
        )

        # Get speaker diarization
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # keep only audios with 2 speakers
        num_speakers = len(set([i["speaker"] for i in result["segments"]]))
        if num_speakers != 2:
            print(f"audio had {num_speakers} speakers, skipping ...")
            continue

        # Get audio embeddings
        input_features = whisper_fe(
            audio, sampling_rate=config.sample_rate, return_tensors="pt"
        ).to(device)["input_features"]
        embeddings = embedding_model(input_features).last_hidden_state

        print([(i["text"], i["speaker"]) for i in result["segments"]])

        utils.play_audio(waveform)
        breakpoint()

        best_speaker = analyze_speakers_pronouns(result)

        if best_speaker is None:
            print("No obvious user speaker, skipping ...")
            continue

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
