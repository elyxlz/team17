import logging
import os
import subprocess
import sys
import tempfile

import soundfile as sf
import torch


class SuppressLogger:
    def __init__(self, activate=True):
        self.activate = activate
        if self.activate:
            self.logger = logging.getLogger()
            self.level = self.logger.level
            self.handlers = self.logger.handlers.copy()

    def __enter__(self):
        if not self.activate:
            return self

        # Remove all handlers
        self.logger.handlers.clear()
        # Add null handler
        self.logger.addHandler(logging.NullHandler())
        self.logger.setLevel(logging.CRITICAL)
        # Suppress stdout/stderr
        self.devnull = open(os.devnull, "w")
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self.devnull
        sys.stderr = self.devnull
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.activate:
            return

        # Restore stdout/stderr
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self.devnull.close()
        # Restore logger state
        self.logger.removeHandler(self.logger.handlers[0])
        self.logger.setLevel(self.level)
        for handler in self.handlers:
            self.logger.addHandler(handler)


def play_audio(x: torch.Tensor, sr: int = 16000) -> None:
    assert x.dim() == 2
    assert x.size(0) < 3

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_filename = temp_file.name

    audio_numpy = x.cpu().float().numpy().T  # Transpose to (num_samples, num_channels)
    sf.write(temp_filename, audio_numpy, samplerate=sr)  # Assuming 44.1kHz sample rate
    try:
        subprocess.run(["vlc", "--play-and-exit", temp_filename], check=True)
    except Exception as e:
        print("no vlc")
    os.remove(temp_filename)
