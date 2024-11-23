import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { AIResponse } from '../utils/openai';

interface HistoryItem {
  timestamp: string;
  transcription: string;
  response: AIResponse;
}


interface ChatSession {
  sessionId: string;
  startTime: string;
  messages: HistoryItem[];
}

export function Conversations() {
  const navigate = useNavigate();
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);

  useEffect(() => {
    const savedSessions = localStorage.getItem('chatSessions');
    if (savedSessions) {
      try {
        const parsedSessions: ChatSession[] = JSON.parse(savedSessions);
        const sortedSessions = parsedSessions
          .map(session => ({
            ...session,
            messages: session.messages.filter(msg => {
              const msgTime = new Date(msg.timestamp).getTime();
              const sessionTime = new Date(session.startTime).getTime();
              const nextSession = parsedSessions.find(s => 
                new Date(s.startTime).getTime() > sessionTime && 
                new Date(s.startTime).getTime() <= msgTime
              );
              return msgTime >= sessionTime && !nextSession;
            })
          }))
          .filter(session => session.messages.length > 0)
          .sort((a, b) => 
            new Date(b.startTime).getTime() - new Date(a.startTime).getTime()
          );
        setChatSessions(sortedSessions);
      } catch (error) {
        console.error('Error parsing chat sessions:', error);
        setChatSessions([]);
      }
    }
  }, []);

  if (chatSessions.length === 0) {
    return (
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold mb-6">All Conversations</h1>
        <p className="text-gray-600">No conversations yet.</p>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">All Conversations</h1>
        <button 
          onClick={() => navigate('/')}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
        >
          Return to Chat
        </button>
      </div>
      <div className="space-y-8">
        {chatSessions.map((session) => (
          <div key={session.sessionId} className="border rounded-lg p-4">
            <div className="text-sm text-gray-500 mb-4">
              Session started at {new Date(session.startTime).toLocaleString()}
            </div>
            <div className="space-y-4">
              {session.messages.map((item, index) => (
                <div 
                  key={index}
                  className="bg-white rounded-lg shadow-sm p-4 border border-gray-200"
                >
                  <div className="text-xs text-gray-500 mb-2">
                    {new Date(item.timestamp).toLocaleString()}
                  </div>
                  <div className="space-y-2">
                    <p className="text-gray-800">
                      <span className="font-semibold">You:</span> {item.transcription}
                    </p>
                    <p className="text-gray-800">
                      <span className="font-semibold">AI:</span> {item.response.text}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}