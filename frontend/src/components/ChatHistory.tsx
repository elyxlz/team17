import { AIResponse } from '../utils/openai';

interface HistoryItem {
  timestamp: string;
  transcription: string;
  response: AIResponse;
}

interface ChatHistoryProps {
  history: HistoryItem[];
}

export function ChatHistory({ history }: ChatHistoryProps) {
  if (history.length === 0) {
    return null;
  }

  const lastConversation = {
    startTime: new Date(history[0].timestamp).toLocaleString(),
    messages: history
  };

  const firstMessage = history[history.length - 1];

  return (
    <div className="sticky bottom-0 w-full flex justify-center pb-4 px-4 bg-gradient-to-t from-white via-white to-transparent">
      <div className="w-full max-w-2xl">
        <h3 className="align-left text-xl font-semibold text-gray-800 mb-4 text-left">Recent Conversation</h3>
        <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-200">
          <div className="flex justify-between items-center mb-2">
            <div className="text-xs text-gray-500">
              {lastConversation.startTime}
            </div>
            <div className="text-xs text-gray-500">
              {history.length} message{history.length > 1 ? 's' : ''}
            </div>
          </div>
          <div className="text-gray-800 text-left">
            <div className="mb-2">
              <span className="font-semibold">You:</span> {firstMessage.transcription}
            </div>
            <div className="mb-2">
              <span className="font-semibold">Therapist:</span> {firstMessage.response.text}
            </div>
            {firstMessage.response.audioUrl && (
              <div className="mt-2">
                <audio controls className="w-full">
                  <source src={firstMessage.response.audioUrl} type="audio/mpeg" />
                </audio>
              </div>
            )}
          </div>
          <div className="mt-4 text-center">
            <button
              onClick={() => window.location.href = '/conversations'}
              className="px-4 py-2 text-sm text-white bg-blue-500 rounded hover:bg-blue-600 transition-colors"
            >
              See All Conversations
            </button>
          </div>
        </div>
      </div>
    </div>
  );
} 