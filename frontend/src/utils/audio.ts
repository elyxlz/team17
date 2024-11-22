export class AudioProcessingError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'AudioProcessingError';
  }
}

export async function createAudioFile(audioBlob: Blob): Promise<File> {
  try {
    const buffer = await audioBlob.arrayBuffer();
    return new File([buffer], 'audio.mp3', { type: 'audio/mp3' });
  } catch (error) {
    console.error('Audio processing error:', error);
    throw new AudioProcessingError('Failed to process audio data. Please try again.');
  }
}