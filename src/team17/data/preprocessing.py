import os
import random
from concurrent.futures import ThreadPoolExecutor

import dotenv
import numpy as np
import pydantic_settings as pyds
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
from tqdm import tqdm

from team17 import utils

SUPPRESS = True

with utils.SuppressLogger(SUPPRESS):
    import transformers
    import whisperx

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
    chunk_frames: int = 16_000 * 24
    sample_rate: int = 16_000
    num_workers: int = 0


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

            # Check duration
            duration = waveform.shape[1] / self.target_sample_rate
            # if duration > 30:
            #     print(f"Duration above 30 seconds for {file_path}, skipping")
            #     return self.__getitem__(idx + 1)
            print(duration)

            return dict(
                waveform=waveform,
                filename=os.path.basename(file_path),
                duration=duration,
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

    return user_speaker, speaker_stats


def get_num_speakers(segments: list) -> int:
    num_speakers = len(set([i["speaker"] for i in segments if "speaker" in i]))
    return num_speakers


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
    with utils.SuppressLogger(SUPPRESS):
        asr_model = whisperx.load_model(
            config.transcription_model,
            device="cuda" if config.use_cuda else "cpu",
            compute_type="float32" if not config.use_cuda else config.compute_type,
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

    save_executor = ThreadPoolExecutor(max_workers=config.num_workers + 1)

    def time_to_idx(time):
        return int(time * 50)

    for batch in tqdm(dl):
        waveform = batch["waveform"]
        filename = batch["filename"]

        audio = waveform.squeeze().numpy()

        # Get transcription and alignment
        with utils.SuppressLogger(SUPPRESS):
            result = asr_model.transcribe(
                audio,
                batch_size=config.inner_batch_size,
                print_progress=True,
                combined_progress=True,
            )
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

        # make sure all segments have a speaker
        no_speaker = False
        for segment in result["segments"]:
            if segment.get("speaker") is None:
                no_speaker = True

        if no_speaker:
            print("a segment has no speaker")
            continue

        # make sure 2 speakers
        num_speakers = get_num_speakers(result["segments"])
        if num_speakers != 2:
            print(f"1: {num_speakers} speaker skipping ... ")
            continue

        # make sure identifiable user speaker
        user_speaker = analyze_speakers_pronouns(result)[0]
        if user_speaker is None:
            print("Cant find user speaker, skipping ... ")
            continue

        # Process segments
        segments = result["segments"]
        sequence = []
        audio_emb_indices = []

        # Find first segment of user speaker
        # start_idx = 0
        # for i, segment in enumerate(result["segments"]):
        #     if segment["speaker"] == user_speaker:
        #         start_idx = i
        #         break

        # Find last segment of non-user speaker
        # end_idx = len(result["segments"]) - 1
        # for i in range(len(result["segments"]) - 1, -1, -1):
        #     if result["segments"][i]["speaker"] != user_speaker:
        #         end_idx = i + 1
        #         break
        #     if result["segments"][i]["speaker"] == user_speaker:
        #         end_idx = i

        # Only use segments between start_idx and end_idx

        # Check again that audio still has 2 speakers
        num_speakers = get_num_speakers(segments)
        if num_speakers != 2:
            print("2. Only one speaker skipping ... ")
            continue

        # Verify that first speaker is user
        # if segments[0]["speaker"] != user_speaker: # TOOD:
        #     print("3. Invalid speaker sequence, skipping ...")
        #     continue

        input_features = whisper_fe(
            audio, sampling_rate=config.sample_rate, return_tensors="pt"
        ).to(device)["input_features"]
        chunks = torch.split(input_features, 3000, dim=-1)
        embeddings = torch.cat(
            [
                embedding_model(c).last_hidden_state.cpu().to(torch.float16)
                for c in chunks
            ],
            dim=1,
        )

        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]

            if segment["speaker"] == user_speaker:
                start_idx = time_to_idx(start_time)
                end_idx = time_to_idx(end_time)
                audio_emb_indices.extend(list(range(start_idx, end_idx)))
                for i in range(start_idx, end_idx):
                    sequence.append(f"<a{i}>")
            else:
                sequence.append(segment["text"])

        text = "".join(sequence)

        processed_data = {
            "text": text,
            "audio_emb": embeddings.cpu().numpy(),
        }

        breakpoint()

        save_executor.submit(
            _save_processed_data, processed_data, filename, config.output_path
        )

    save_executor.shutdown(wait=True)


if __name__ == "__main__":
    config = PreprocessingConfig()
    process_audio_chunks(config)
