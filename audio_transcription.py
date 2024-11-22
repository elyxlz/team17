import whisperx
import gc 

device = "cpu" 
audio_file = "/Users/hindy/Desktop/Learn English Conversation_ Weather and Small Talk 30 SECONDS (Subtitles).mp3"
batch_size = 2 # reduce if low on GPU mem
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type, asr_options={
            "multilingual": True,
            "hotwords": [],
        },)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)

result = model.transcribe(audio, batch_size=batch_size) 
print(result["segments"]) # before alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(type(result))

print(result["segments"]) # after alignmentdefault_asr_options

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

HF_TOK = "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" # Elio's token

# 3. Assign speaker labels
diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOK, device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs