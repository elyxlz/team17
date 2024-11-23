import { useState, useEffect } from 'react';
import { Brain } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import { BrowserRouter as Router, Routes, Route, useNavigate, Navigate } from 'react-router-dom';
import VoiceButton from './components/VoiceButton';
import ResponseDisplay from './components/ResponseDisplay';
import ErrorMessage from './components/ErrorMessage';
import { processMessageWithOpenAI } from './utils/openai';
import { processMessageWithLocal } from './utils/local';
import { ConfigError } from './utils/config';
import Register from './components/Register.tsx';
import Login from './components/Login.tsx';
import ProtectedRoute from './components/ProtectedRoute.tsx';
import { ChatHistory } from './components/ChatHistory';
import { Conversations } from './pages/Conversations.tsx';
import "./styles.css";
import { Header } from './components/Header';
import { AnimatedText } from './components/AnimatedText';
import { AnimatedBlob } from './components/AnimatedBlob';
import { Footer } from './components/Footer';
import { ScrollIndicator } from './components/ScrollIndicator';

// Update this interface to match the one from openai.ts
interface AIResponse {
  text: string;
  audioUrl?: string;  // Optional property as defined in openai.ts
}

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
  const [scrollY, setScrollY] = useState(0);
  const text = "Your own private therapist to prevent depression.".split(" ");

  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const isAuthenticated = localStorage.getItem('isAuthenticated') === 'true';

  const handleError = (message: string) => {
    setError(message);
    setIsProcessing(false);
  };

  const handleRecordingComplete = async (audioBlob: Blob) => {
    setError(null);
    setResponse(null);
    setIsProcessing(true);

    try {
      // const aiResponse = await processMessageWithOpenAI(audioBlob);
      const aiResponse = await processMessageWithLocal(audioBlob);
      setResponse(aiResponse);

      const newHistoryItem = {
        timestamp: new Date().toLocaleString(),
        transcription: aiResponse.text,
        response: aiResponse
      };

      const updatedHistory = [newHistoryItem, ...history];
      setHistory(updatedHistory);
      localStorage.setItem('chatHistory', JSON.stringify(updatedHistory));
    } catch (error) {
      if (error instanceof ConfigError) {
        handleError(`Configuration Error: ${error.message}`);
      } else {
        handleError(error as string);
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const handleAuthAction = () => {
    if (isAuthenticated) {
      localStorage.removeItem('isAuthenticated');
      navigate('/login');
    } else {
      navigate('/login');
    }
  };

  return (
    <div className="App">
      <AnimatedBlob scrollY={scrollY} />
      <AnimatedText text={text} scrollY={scrollY} />
      <button 
        onClick={handleAuthAction}
        className={`fixed top-4 right-4 px-4 py-2 ${
          isAuthenticated ? 'text-black hover:italic' : 'hidden'
        } rounded transition-colors`}
      >
        {isAuthenticated ? 'Logout' : 'Login'}
      </button>
    
        <div className="grosse_div min-h-screen bg-gradient-to-br flex flex-col items-center justify-center p-4">
          
          <AnimatePresence>
            {error && <ErrorMessage message={error} />}
          </AnimatePresence>

            <ResponseDisplay
              response={response}
              isVisible={!!response && !error}
              />
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

            {history.length > 0 && <ChatHistory history={history} />}

          <Footer />
        </div>
      
    </div>
  );
}

function App() {
  return (
    <Router>
      <Routes>
        <Route 
          path="/register" 
          element={
            localStorage.getItem('isAuthenticated') === 'true' 
              ? <Navigate to="/" /> 
              : <Register />
          } 
        />
        <Route 
          path="/login" 
          element={
            localStorage.getItem('isAuthenticated') === 'true' 
              ? <Navigate to="/" /> 
              : <Login />
          } 
        />
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <VoiceAssistant />
            </ProtectedRoute>
          }
        />
        <Route
          path="/conversations"
          element={
            <ProtectedRoute>
              <Conversations />
            </ProtectedRoute>
          }
        />
      </Routes>
    </Router>
  );
}

export default App;
