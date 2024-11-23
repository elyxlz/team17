curl -X POST http://localhost:3000/v1/audio/speech \
-H "Content-Type: application/json" \
-d '{
  "voice": "default",
  "input": "Hello, this is an example of synthesized speech!",
  "response_format": "wav",
  "model": "fish-speech-1.4"
}' --output result.wav