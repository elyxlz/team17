import time

import librosa
import transformers

pipe = transformers.pipeline(
    model="fixie-ai/ultravox-v0_3-llama-3_2-1b",
    trust_remote_code=True,
    device="mps",
)

path = "./sample.wav"  # TODO: pass the audio here
audio, sr = librosa.load(path, sr=16000)


turns = [
    {
        "role": "system",
        "content": "You are a friendly and helpful character. You love to answer questions for people.",
    },
]
start_time = time.time()

for k in range(10):
    # Benchmark the pipeline
    x = pipe({"audio": audio, "turns": turns, "sampling_rate": sr}, max_new_tokens=30)
end_time = time.time()

print(x)
print(f"Execution Time: {(end_time - start_time)/10:.2f} seconds")
