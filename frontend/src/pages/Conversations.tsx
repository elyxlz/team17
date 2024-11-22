import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { AIResponse } from '../utils/openai';

interface HistoryItem {
  timestamp: string;
  transcription: string;
  response: AIResponse;
}

interface GroupedConversation {
  startTime: string;
  messages: HistoryItem[];
}

export function Conversations() {
  const navigate = useNavigate();
  const [groupedConversations, setGroupedConversations] = useState<GroupedConversation[]>([]);

  useEffect(() => {
    const savedHistory = localStorage.getItem('chatHistory');
    if (savedHistory) {
      try {
        const parsedHistory: HistoryItem[] = JSON.parse(savedHistory);
        
        // Group conversations by hour
        const grouped = parsedHistory.reduce((acc: GroupedConversation[], item) => {
          const itemTime = new Date(item.timestamp);
          const lastGroup = acc[acc.length - 1];
          
          if (lastGroup && 
              new Date(lastGroup.startTime).getTime() + 3600000 > itemTime.getTime() && 
              lastGroup.messages.length < 10) {
            lastGroup.messages.push(item);
          } else {
            acc.push({
              startTime: item.timestamp,
              messages: [item]
            });
          }
          return acc;
        }, []);

        setGroupedConversations(grouped);
      } catch (error) {
        console.error('Error parsing chat history:', error);
        setGroupedConversations([]);
      }
    }
  }, []);

  if (groupedConversations.length === 0) {
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
        {groupedConversations.map((group, groupIndex) => (
          <div key={groupIndex} className="border rounded-lg p-4">
            <div className="text-sm text-gray-500 mb-4">
              Conversation started at {new Date(group.startTime).toLocaleString()}
            </div>
            <div className="space-y-4">
              {group.messages.map((item, index) => (
                <div 
                  key={index}
                  className="bg-white rounded-lg shadow-sm p-4 border border-gray-200"
                >
                  <div className="text-xs text-gray-500 mb-2">{item.timestamp}</div>
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