import os
import random
import json

import numpy as np
import pydantic_settings as pyds
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

import whisperx
from team17.data import utils


class PreprocessingConfig(pyds.BaseSettings):
    seed: int = 1338
    input_path: str = "./data/chunks"
    output_path: str = "./data/processed"
    transcription_model: str = "medium"  # Model for transcription
    compute_type: str = "float16"
    inner_batch_size: int = 8
    use_cuda: bool = torch.cuda.is_available()
    chunk_frames: int = 16_000 * 30
    sample_rate: int = 16_000
    num_workers: int = 8


def _is_silent(waveform: torch.Tensor, thresh: float) -> bool:
    rms = torch.sqrt(torch.mean(waveform**2, dim=0))
    silent_samples = torch.sum(rms < thresh)
    return (silent_samples.item() / rms.shape[0]) >= 0.9


class AudioChunkDataset(Dataset):
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
        self.file_list = self._find_audio_files(input_path)
        random.seed(seed)
        random.shuffle(self.file_list)
        self.file_list = list(set(self.file_list))
        print(f"FOUND {len(self.file_list)} raw files")

    def _find_audio_files(self, directory):
        file_list = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(
                    (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".mp4", ".webm")
                ):
                    file_list.append(os.path.join(root, file))
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        try:
            file_path = self.file_list[idx]

            # Load audio
            waveform, sample_rate = torchaudio.load(file_path)

            # Resample if needed
            if sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, self.target_sample_rate
                )
                waveform = resampler(waveform)

            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Skip if audio is silent
            if _is_silent(waveform, thresh=self.silence_threshold):
                return self.__getitem__(idx + 1)

            # Pad or cut to chunk_frames
            if waveform.shape[1] < self.chunk_frames:
                waveform = F.pad(waveform, (0, self.chunk_frames - waveform.shape[1]))
            elif waveform.shape[1] > self.chunk_frames:
                waveform = waveform[:, : self.chunk_frames]

            return dict(
                waveform=waveform.numpy(),
                filename=os.path.basename(file_path),
            )

        except Exception as e:
            print(f"Error processing: {str(e)}")
            return self.__getitem__(idx + 1)


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

    def find_user_speaker(stats):
        return max(
            stats.items(),
            key=lambda x: x[1]["first_person"] / (x[1]["second_person"] + 0.1),
            default=(None, {}),
        )[0]

    speaker_stats = count_pronouns(get_speakers())
    user_speaker = find_user_speaker(speaker_stats)
    return user_speaker


def extract_conversation_turns(result, user_speaker):
    """Extract turns from diarized transcript, ensuring user start and assistant end."""
    messages = []
    current_speaker = None
    current_text = []
    current_start = None

    # Find first user segment
    user_start_idx = None
    for idx, segment in enumerate(result["segments"]):
        if segment.get("speaker") == user_speaker:
            user_start_idx = idx
            break

    if user_start_idx is None:
        return None  # No user turns found

    # Find last assistant segment
    assistant_end_idx = None
    for idx in range(len(result["segments"]) - 1, -1, -1):
        if result["segments"][idx].get("speaker") != user_speaker:
            assistant_end_idx = idx
            break

    if assistant_end_idx is None or assistant_end_idx <= user_start_idx:
        return None  # No valid conversation structure found

    # Process segments within the valid range
    segments = result["segments"][user_start_idx : assistant_end_idx + 1]

    for segment in segments:
        if segment.get("speaker") is None:
            continue

        if current_speaker is None:
            current_speaker = segment["speaker"]
            current_start = segment["start"]
            current_text = [segment["text"]]
        elif segment["speaker"] == current_speaker:
            current_text.append(segment["text"])
        else:
            # Save previous turn
            role = "user" if current_speaker == user_speaker else "assistant"
            if role == "user":
                message = {
                    "role": role,
                    "content": "<|audio|>",
                    "start_time": current_start,
                    "end_time": segment["start"],
                    "transcript": " ".join(current_text).strip(),
                }
            else:
                message = {"role": role, "content": " ".join(current_text).strip()}
            messages.append(message)

            # Start new turn
            current_speaker = segment["speaker"]
            current_start = segment["start"]
            current_text = [segment["text"]]

    # Add final turn (should be assistant based on our filtering)
    if current_text and current_speaker != user_speaker:
        message = {"role": "assistant", "content": " ".join(current_text).strip()}
        messages.append(message)

    # Validate turn structure
    if (
        not messages
        or messages[0]["role"] != "user"
        or messages[-1]["role"] != "assistant"
    ):
        return None

    return messages


def _save_processed_data(processed_data: dict, filename: str, output_path: str) -> None:
    """Save all data in a single .npz file."""
    subdir = filename[:2]
    full_output_path = os.path.join(output_path, subdir)
    os.makedirs(full_output_path, exist_ok=True)

    # Convert conversation dict to bytes for storage
    conv_bytes = json.dumps(processed_data["conversation"]).encode("utf-8")

    # Save everything in a single .npz file
    output_file = os.path.join(full_output_path, f"{os.path.splitext(filename)[0]}.npz")
    np.savez(
        output_file,
        audio=processed_data["audio"],
        conversation=np.frombuffer(conv_bytes, dtype=np.uint8),
    )


@torch.no_grad()
def process_audio_chunks(config: PreprocessingConfig):
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    device = torch.device("cuda" if config.use_cuda else "cpu")

    # Load models
    with utils.SuppressLogger():
        asr_model = whisperx.load_model(
            config.transcription_model,
            device="cuda" if config.use_cuda else "cpu",
            compute_type="float32" if not config.use_cuda else config.compute_type,
            language="en",
        )

    diarize_model = whisperx.DiarizationPipeline(
        device=device, use_auth_token=os.getenv("HF_TOKEN")
    )

    dataset = AudioChunkDataset(
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
    for batch in tqdm(dl):
        waveform = batch["waveform"]
        filename = batch["filename"][0]

        audio = waveform.squeeze().numpy()

        # Get transcription and alignment
        with utils.SuppressLogger(False):
            result = asr_model.transcribe(
                audio,
                batch_size=config.inner_batch_size,
                language="en",
            )

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

        # Diarization
        diarize_segments = diarize_model(audio, num_speakers=2)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        # Find user speaker
        user_speaker = analyze_speakers_pronouns(result)
        if user_speaker is None:
            print("Can't find user speaker, skipping...")
            continue

        # Extract conversation with proper turn ordering
        messages = extract_conversation_turns(result, user_speaker)
        if messages is None:
            print("Invalid conversation structure, skipping...")
            continue

        processed_data = {
            "audio": waveform.cpu().numpy().astype(np.float32),
            "conversation": {"messages": messages, "sample_rate": config.sample_rate},
        }

        _save_processed_data(processed_data, filename, config.output_path)


if __name__ == "__main__":
    config = PreprocessingConfig()
    process_audio_chunks(config)
