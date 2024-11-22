from team17.data.preprocessing import PreprocessingConfig, process_audio_chunks

config = PreprocessingConfig(chunk_frames=16_000 * 10)
process_audio_chunks(config)
