import os
import subprocess
import tempfile

import soundfile as sf
import torch


def play_audio(x: torch.Tensor, sr: int = 16000) -> None:
    assert x.dim() == 2
    assert x.size(0) < 3

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_filename = temp_file.name

    audio_numpy = x.cpu().float().numpy().T  # Transpose to (num_samples, num_channels)
    sf.write(temp_filename, audio_numpy, samplerate=sr)  # Assuming 44.1kHz sample rate
    subprocess.run(["vlc", "--play-and-exit", temp_filename], check=True)
    os.remove(temp_filename)
