export type AIResponse = {
  text: string;
  audioUrl?: string;
};

// Local backend URLs
const STT_URL = "http://0.0.0.0:8000/speech-to-text-file/";
const TTS_URL = "http://localhost:3000/v1/audio/speech";

// Speech-to-Text and Text-to-Speech function
export async function processMessageWithLocal(audioBlob: Blob): Promise<AIResponse> {
  try {
    // 1. Speech-to-Text with Ultravox
    const formData = new FormData();
    formData.append("audio_file", audioBlob, "audio.wav");

    const sttResponse = await fetch(STT_URL, {
      method: "POST",
      body: formData,
    });

    if (!sttResponse.ok) {
      throw new Error(`Speech-to-Text Error: ${sttResponse.statusText}`);
    }

    const sttData = await sttResponse.json();
    const transcribedText = sttData.text;

    if (!transcribedText) {
      throw new Error("No transcription available from the local backend");
    }

    // 2. Text-to-Speech with local backend
    const ttsRequestBody = {
      voice: "default", // Adjust the voice parameter as needed
      input: transcribedText,
      response_format: "wav",
      model: "fish-speech-1.4",
    };

    const ttsResponse = await fetch(TTS_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(ttsRequestBody),
    });

    if (!ttsResponse.ok) {
      throw new Error(`Text-to-Speech Error: ${ttsResponse.statusText}`);
    }

    const ttsAudioBlob = await ttsResponse.blob();
    const audioUrl = URL.createObjectURL(ttsAudioBlob);

    return {
      text: transcribedText,
      audioUrl: audioUrl,
    };
  } catch (error) {
    console.error("Error processing message locally:", error);
    throw error;
  }
}
