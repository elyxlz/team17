from dataclasses import dataclass
import datetime
import logging
import tempfile

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from f5_tts_mlx.cfm import F5TTS
from f5_tts_mlx.utils import convert_char_to_pinyin


from fastapi.responses import FileResponse
import mlx.core as mx
from mlx import nn as nn

import soundfile as sf

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="F5 TTS MLX API", version="1.0")

logger.info("Initializing F5 Inference...")
f5tts = F5TTS.from_pretrained("lucasnewman/f5-tts-mlx")
# f5tts = nn.quantize(f5tts, group_size=64, bits=4)

logger.info("F5 Inference initialized successfully.")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_RATE = 24_000
HOP_LENGTH = 256
FRAMES_PER_SEC = SAMPLE_RATE / HOP_LENGTH
TARGET_RMS = 0.1


@dataclass
class F5GenerateRequest:
    voice: str
    input: str
    response_format: str
    model: str


@app.post("/v1/audio/speech")
async def f5_generate(
    request: F5GenerateRequest,
):
    generation_text = request.input

    ref_audio_path = "./voices/default.wav"
    ref_audio_text = "Morgan Freeman is an acclaimed American actor and narrator, known for his deep, distinctive voice and versatile roles in films such as The Shawshank Redemption. With the help of text-to-speech tools, it is now possible to generate voice and how to use best Morgan Freeman voice generators in three ways to create custom voiceovers, narrations, and speeches for your videos. Now, let's get started."
    steps = 10
    method = "euler"
    cfg_strength = 2.0
    sway_sampling_coef = -1.0
    speed = 1.0
    seed = None

    # load reference audio
    audio, sr = sf.read(ref_audio_path)
    if sr != SAMPLE_RATE:
        raise ValueError("Reference audio must have a sample rate of 24kHz")

    audio = mx.array(audio)
    ref_audio_duration = audio.shape[0] / SAMPLE_RATE
    print(f"Got reference audio with duration: {ref_audio_duration:.2f} seconds")

    rms = mx.sqrt(mx.mean(mx.square(audio)))
    if rms < TARGET_RMS:
        audio = audio * TARGET_RMS / rms

    # generate the audio for the given text
    text = convert_char_to_pinyin([ref_audio_text + " " + generation_text])

    start_date = datetime.datetime.now()

    duration = None

    wave, _ = f5tts.sample(
        mx.expand_dims(audio, axis=0),
        text=text,
        duration=duration,
        steps=steps,
        method=method,
        speed=speed,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        seed=seed,
    )

    # trim the reference audio
    wave = wave[audio.shape[0] :]
    generated_duration = wave.shape[0] / SAMPLE_RATE
    elapsed_time = datetime.datetime.now() - start_date

    print(f"Generated {generated_duration:.2f} seconds of audio in {elapsed_time}.")

    # Save the generated audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        sf.write(temp_file.name, wave, SAMPLE_RATE, format="WAV")
        temp_file_path = temp_file.name

    # Return the temporary file as the response
    return FileResponse(
        temp_file_path, media_type="audio/wav", filename="generated_audio.wav"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
