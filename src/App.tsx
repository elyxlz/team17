import { useState } from 'react';
import { Brain } from 'lucide-react';
import { AnimatePresence } from 'framer-motion';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import VoiceButton from './components/VoiceButton';
import ResponseDisplay from './components/ResponseDisplay';
import ErrorMessage from './components/ErrorMessage';
import { processMessageWithOpenAI, type AIResponse } from './utils/openai';
import { ConfigError } from './utils/config';
import Register from './components/Register.tsx';
import Login from './components/Login.tsx';
import ProtectedRoute from './components/ProtectedRoute.tsx';
import { ChatHistory } from './components/ChatHistory';
import { Conversations } from './pages/Conversations.tsx';

function VoiceAssistant() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [response, setResponse] = useState<AIResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  const [history, setHistory] = useState<Array<{
    timestamp: string;
    transcription: string;
    response: AIResponse;
  }>>(() => {
    const saved = localStorage.getItem('chatHistory');
    return saved ? JSON.parse(saved) : [];
  });

  const handleError = (message: string) => {
    setError(message);
    setIsProcessing(false);
  };

  const handleRecordingComplete = async (audioBlob: Blob) => {
    setError(null);
    setResponse(null);
    setIsProcessing(true);
    
    try {
      // 1. Speech to Text
      const formData = new FormData();
      formData.append('file', audioBlob, 'audio.webm');
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
      
      // 2. Get AI Response
      const result = await processMessageWithOpenAI(transcribedText);
      if (!result) return;
      
      // 3. Text to Speech
      const ttsResponse = await fetch('https://api.openai.com/v1/audio/speech', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${import.meta.env.VITE_OPENAI_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'tts-1',
          voice: 'alloy',
          input: result.text,
          response_format: 'aac',
        }),
      });

      const ttsAudioBlob = await ttsResponse.blob();
      const audioUrl = URL.createObjectURL(ttsAudioBlob);
      
      // Play the audio
      const audio = new Audio(audioUrl);
      audio.play();
      
      setResponse(result);
      
      const newHistoryItem = {
        timestamp: new Date().toLocaleString(),
        transcription: result.text,
        response: result
      };
      
      const updatedHistory = [newHistoryItem, ...history];
      setHistory(updatedHistory);
      localStorage.setItem('chatHistory', JSON.stringify(updatedHistory));
    } catch (error: any) {
      if (error instanceof ConfigError) {
        handleError(`Configuration Error: ${error.message}`);
      } else {
        handleError(error.message);
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('isAuthenticated');
    navigate('/login');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex flex-col items-center justify-center p-4">
      <button 
        onClick={handleLogout}
        className="absolute top-4 right-4 px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
      >
        Logout
      </button>

      <div className="text-center mb-12">
        <div className="flex items-center justify-center mb-4">
          <Brain className="w-12 h-12 text-blue-600 mr-3" />
          <h1 className="text-4xl font-bold text-gray-800">Voice Assistant</h1>
        </div>
        <p className="text-gray-600 text-lg">Speak your mind, and let AI assist you</p>
      </div>

      <AnimatePresence>
        {error && <ErrorMessage message={error} />}
      </AnimatePresence>

      <div className="relative">
        <VoiceButton
          onRecordingComplete={handleRecordingComplete}
          isProcessing={isProcessing}
          onError={handleError}
        />
        {isProcessing && (
          <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 whitespace-nowrap">
            <p className="text-sm text-gray-600">Processing your request...</p>
          </div>
        )}
      </div>
      <div className="mt-4 text-center text-gray-500 text-sm">
        Click the button and start speaking
      </div>

      <ResponseDisplay
        response={response}
        isVisible={!!response && !error}
      />

      {history.length > 0 && <ChatHistory history={history} />}

      <footer className="fixed bottom-4 text-center text-gray-500 text-sm">
        Your conversation is only stored locally on your device, it never leaves your control. None of what you will say can be accessed or used.
      </footer>
    </div>
  );
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/register" element={<Register />} />
        <Route path="/login" element={<Login />} />
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <VoiceAssistant />
            </ProtectedRoute>
          }
        />
        <Route path="/conversations" element={<Conversations />} />
      </Routes>
    </Router>
  );
}

export default App;