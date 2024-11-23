import OpenAI from 'openai';

export type AIResponse = {
  text: string;
  audioUrl?: string;
};

const apiKey = import.meta.env.VITE_OPENAI_API_KEY;
if (!apiKey) {
  throw new Error('OpenAI API key is not configured in environment variables');
}

const openai = new OpenAI({
  apiKey: apiKey,
  dangerouslyAllowBrowser: true
});

export async function processMessageWithOpenAI(inputAudioBlob: Blob): Promise<AIResponse> {
  try {
    // 1. Speech-to-Text
    const formData = new FormData();
    formData.append('file', inputAudioBlob, 'audio.webm');
    formData.append('model', 'whisper-1');

    const transcriptionResponse = await fetch('https://api.openai.com/v1/audio/transcriptions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${import.meta.env.VITE_OPENAI_API_KEY}`,
      },
      body: formData,
    });

    const transcriptionData = await transcriptionResponse.json();
    const transcribedText = transcriptionData.text;

    if (!transcribedText) {
      throw new Error('No transcription available');
    }

    // 2. Chat Completion
    const completion = await openai.chat.completions.create({
      messages: [{ role: "user", content: transcribedText }],
      model: "gpt-3.5-turbo",
    });

    const textResponse = completion.choices[0]?.message?.content;
    if (!textResponse) {
      throw new Error('No text response from AI');
    }

    // 3. Text-to-Speech
    const speechResponse = await openai.audio.speech.create({
      model: "tts-1",
      voice: "alloy",
      input: textResponse,
    });

    const outputAudioBlob = new Blob([await speechResponse.arrayBuffer()], { type: 'audio/mpeg' });
    const audioUrl = URL.createObjectURL(outputAudioBlob);

    return {
      text: textResponse,
      audioUrl: audioUrl,
    };
  } catch (error) {
    console.error('Error processing message:', error);
    throw error;
  }
}
