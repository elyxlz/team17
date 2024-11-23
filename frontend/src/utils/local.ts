export type AIResponse = {
  text: string;
  audioUrl?: string;
};

// Local backend URLs
const STT_URL = "/api/stt";
const TTS_URL = "/api/tts";

// Speech-to-Text and Text-to-Speech function
export async function processMessageWithLocal(audioBlob: Blob): Promise<AIResponse> {
  try {
    // 1. Speech-to-Text with Ultravox
    const formData = new FormData();
    formData.append("audio_file", audioBlob, "audio.wav");

    const sttResponse = await fetch(STT_URL, {
      method: "POST",
      body: formData,
      // mode: "no-cors",
    }).then((res) => res.text());
    // console.log(sttResponse);
    // if (!sttResponse.ok) {
    //   throw new Error(`Speech-to-Text Error: ${sttResponse.statusText}`);
    // }
    console.log(`STT Response: ${sttResponse}`);

    const transcribedText = sttResponse;

    // const transcribedText = "This is a test";

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
    console.log(`TTS Request Body: ${JSON.stringify(ttsRequestBody)}`);

    const ttsResponseBlob = await fetch(TTS_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(ttsRequestBody),
    }).then((response) => {
      if (!response.ok) {
        throw new Error(`Text-to-Speech Error: ${response.statusText}`);
      }
      return response.blob(); // Get the audio file as a Blob
    });

    // Create a URL for the Blob
    // const audioUrl = URL.createObjectURL(ttsResponseBlob);
    // Convert the Blob to Base64
    const audioBase64 = await blobToBase64(ttsResponseBlob);
    console.log(`Audio Base64: ${audioBase64}`);

    return {
      text: transcribedText,
      audioUrl: `data:audio/wav;base64,${audioBase64}`,
    };
  } catch (error) {
    console.error("Error processing message locally:", error);
    throw error;
  }
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64data = reader.result?.toString().split(",")[1]; // Remove the metadata
      resolve(base64data || "");
    };
    reader.onerror = (error) => reject(error);
    reader.readAsDataURL(blob);
  });
}