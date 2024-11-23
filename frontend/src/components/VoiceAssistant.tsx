import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import VoiceButton from './VoiceButton';
import ResponseDisplay from './ResponseDisplay';
import ErrorMessage from './ErrorMessage';
import { ChatHistory } from './ChatHistory';
import { AnimatedText } from './AnimatedText';
import { AnimatedBlob } from './AnimatedBlob';
import { Footer } from './Footer';
import { processMessageWithOpenAI, getTranscription } from '../utils/openai';
import { ConfigError } from '../utils/config';

interface HistoryItem {
  timestamp: string;
  transcription: string;
  response: AIResponse;
}

interface AIResponse {
  text: string;
  audioUrl?: string;
}

interface ChatSession {
  sessionId: string;
  startTime: string;
  messages: HistoryItem[];
}

export function VoiceAssistant({ showLock = true }: { showLock?: boolean }) {
  const navigate = useNavigate();
  const [isProcessing, setIsProcessing] = useState(false);
  const [response, setResponse] = useState<AIResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [scrollY, setScrollY] = useState(0);
  const [isScrollLocked, setIsScrollLocked] = useState(false);
  
  const [history, setHistory] = useState<Array<{
    timestamp: string;
    transcription: string;
    response: AIResponse;
  }>>(() => {
    const saved = localStorage.getItem('chatHistory');
    return saved ? JSON.parse(saved) : [];
  });

  const welcomeText = "Your own private therapist to prevent depression.".split(" ");

  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY);
      
      if ((window.innerHeight + window.scrollY) >= document.documentElement.scrollHeight - 50) {
        if (!isScrollLocked) {
          setIsScrollLocked(true);
          document.body.style.overflow = 'hidden';
          window.scrollTo(0, document.documentElement.scrollHeight);
        }
      }
    };
    
    window.addEventListener("scroll", handleScroll);
    return () => {
      window.removeEventListener("scroll", handleScroll);
      document.body.style.overflow = 'auto';
    };
  }, [isScrollLocked]);

  const handleLogout = () => {
    localStorage.removeItem('isAuthenticated');
    navigate('/login');
  };

  const handleError = (message: string) => {
    setError(message);
    setIsProcessing(false);
  };

  const handleRecordingComplete = async (audioBlob: Blob) => {
    setError(null);
    setResponse(null);
    setIsProcessing(true);

    try {
      const transcription = await getTranscription(audioBlob);
      const aiResponse = await processMessageWithOpenAI(audioBlob);
      setResponse(aiResponse);
      handleNewMessage(transcription, aiResponse);
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

  const handleNewMessage = (transcription: string, response: AIResponse) => {
    const newMessage: HistoryItem = {
      timestamp: new Date().toISOString(),
      transcription,
      response,
    };

    // Get existing sessions or initialize empty array
    const savedSessions = localStorage.getItem('chatSessions');
    let sessions: ChatSession[] = savedSessions ? JSON.parse(savedSessions) : [];

    // Check if there's an active session for today
    const today = new Date().toISOString();
    let currentSession = sessions.find(s => 
      new Date(s.startTime).toDateString() === new Date(today).toDateString()
    );

    if (!currentSession) {
      // Create new session if none exists for today
      currentSession = {
        sessionId: crypto.randomUUID(),
        startTime: today,
        messages: []
      };
      sessions.push(currentSession);
    }

    // Add new message to current session
    currentSession.messages.push(newMessage);

    // Save back to localStorage
    localStorage.setItem('chatSessions', JSON.stringify(sessions));

    // Update local state if needed
    setHistory(prev => [...prev, newMessage]);
  };

  return (
    <div className="App">
      <AnimatedBlob scrollY={scrollY} />
      <AnimatedText text={welcomeText} scrollY={scrollY} />
      
      {showLock && localStorage.getItem('isAuthenticated') && (
        <button 
          onClick={handleLogout}
          className="fixed top-4 right-4 px-4 py-2 text-black hover:italic rounded transition-colors"
        >
          🔐 Lock
        </button>
      )}
    
      <div className="grosse_div min-h-screen bg-gradient-to-br flex flex-col items-center justify-center p-4">
        {error && (
          <ErrorMessage message={error} />
        )}

        <ResponseDisplay
          response={response}
          isVisible={!!response && !error}
        />

        <div className="relative sticky top-[380px]">
          <VoiceButton
            onRecordingComplete={handleRecordingComplete}
            isProcessing={isProcessing}
            onError={handleError}
          />
        </div>

        <div className="mt-4 text-center text-gray-500 text-sm sticky top-[550px]">
          Click the button and start speaking
        </div>

        {history.length > 0 && <ChatHistory history={history} />}

        <Footer />
      </div>
    </div>
  );
} 