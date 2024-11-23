# Batman 🦇

<p align="center"><img src="./assets/batman.jpg" alt="Batman"></p>

Batman is a 100% local and custom speech2speech pipeline purpose built for emotional understanding and therapeutic conversations. 🗣️

# Architecture and Training

The system operates as a 2-stage pipeline:

1. **Audio Language Model (ALM)** 🧠: A LLaMA base model trained to directly process Whisper embeddings in its context, bypassing traditional transcription. By feeding emotional and tonal information directly from speech embeddings into the model, we maintain crucial affective signals that are typically lost in text intermediates - essential for therapeutic conversations.

2. **Text-to-Speech** 🔊: Powered by T5-TTS for voice synthesis.

The training data consisted of hand-curated therapeutic conversations, collected through a distributed yt-dlp + ffmpeg scraping setup. Our preprocessing pipeline, implemented with distributed PyTorch DataLoaders, performed:

- Whisper transcription
- Pyannote speaker diarization
- wav2vec 2.0 forced alignment timestamping
- Whisper embedding generation
- User/therapist identification based on first/second person pronoun ratios

The ALM was initialized from Ultravox weights and fine-tuned using LoRA and activation checkpointing via a custom trainer.

The TTS stage currently represents our primary latency bottleneck. Our planned evolution involves interleaving mimi (Moshi codec) and text tokens (thought tokens) in the assistant outputs to create a unified end-to-end speechLM. ⚡

# Inference

# Frontend

