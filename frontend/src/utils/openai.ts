import OpenAI from 'openai';

export type AIResponse = {
  text: string;
  audioUrl?: string;
};

const openai = new OpenAI({
  apiKey: import.meta.env.VITE_OPENAI_API_KEY,
  dangerouslyAllowBrowser: true
});

export async function processMessageWithOpenAI(message: string): Promise<AIResponse> {
  try {
    // 1. First get the text response
    const completion = await openai.chat.completions.create({
      messages: [{ role: "user", content: message }],
      model: "gpt-3.5-turbo",
    });

    const textResponse = completion.choices[0]?.message?.content;
    if (!textResponse) {
      throw new Error('No response from AI');
    }

    // 2. Then convert that same text to speech
    const speechResponse = await openai.audio.speech.create({
      model: "tts-1",
      voice: "alloy",
      input: textResponse,  // Using the same text response
    });

    // 3. Create a playable URL for the audio
    const audioBlob = new Blob([await speechResponse.arrayBuffer()], { type: 'audio/mpeg' });
    const audioUrl = URL.createObjectURL(audioBlob);

    // 4. Return BOTH text and audio
    return {
      text: textResponse,    // The text response
      audioUrl: audioUrl,    // The speech version
    };
  } catch (error) {
    console.error('Error processing message:', error);
    throw error;
  }
}