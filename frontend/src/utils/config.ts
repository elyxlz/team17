export class ConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ConfigError';
  }
}

export function validateApiKey(apiKey: string | undefined): string {
  if (!apiKey) {
    throw new ConfigError(
      'OpenAI API key is missing. Please add your API key to the .env.local file with VITE_OPENAI_API_KEY.'
    );
  }

  if (apiKey === 'your_openai_api_key_here') {
    throw new ConfigError(
      'Please replace the placeholder API key in .env.local with your actual OpenAI API key.'
    );
  }

  return apiKey;
}