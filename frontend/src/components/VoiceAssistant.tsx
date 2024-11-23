import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import VoiceButton from './VoiceButton';
import ResponseDisplay from './ResponseDisplay';
import ErrorMessage from './ErrorMessage';
import { ChatHistory } from './ChatHistory';
import { AnimatedText } from './AnimatedText';
import { AnimatedBlob } from './AnimatedBlob';
import { Footer } from './Footer';
import { processMessageWithOpenAI, getTranscription } from '../utils/openai';
import { processMessageWithLocal } from '../utils/local';
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
  const [currentState, setCurrentState] = useState<"idle" | "listening" | "thinking" | "speaking">("idle");
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
  const audioRef = useRef<HTMLAudioElement | null>(null); // Reference to the audio element

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

  useEffect(() => {
    if (response?.audioUrl && audioRef.current) {
      audioRef.current.play().catch((err) => {
        console.error("Autoplay failed:", err);
      });
    }
  }, [response?.audioUrl]);

  const handleLogout = () => {
    localStorage.removeItem('isAuthenticated');
    navigate('/login');
  };

  const handleError = (message: string) => {
    setError(message);
    setCurrentState("idle");
  };

  const handleRecordingComplete = async (audioBlob: Blob) => {
    setError(null);
    setResponse(null);
    setCurrentState("thinking");
    try {
      const aiResponse = await processMessageWithLocal(audioBlob);
      setResponse(aiResponse);
      const transcription = "";
      handleNewMessage(transcription, aiResponse);
    } catch (error) {
      if (error instanceof ConfigError) {
        handleError(`Configuration Error: ${error.message}`);
      } else {
        handleError(error as string);
      }
    } finally {
      setCurrentState("idle");
    }
  };

  const handleNewMessage = (transcription: string, response: AIResponse) => {
    const newMessage: HistoryItem = {
      timestamp: new Date().toISOString(),
      transcription,
      response,
    };

    const savedSessions = localStorage.getItem('chatSessions');
    const sessions: ChatSession[] = savedSessions ? JSON.parse(savedSessions) : [];

    const today = new Date().toISOString();
    let currentSession = sessions.find(s =>
      new Date(s.startTime).toDateString() === new Date(today).toDateString()
    );

    if (!currentSession) {
      currentSession = {
        sessionId: crypto.randomUUID(),
        startTime: today,
        messages: []
      };
      sessions.push(currentSession);
    }

    currentSession.messages.push(newMessage);
    localStorage.setItem('chatSessions', JSON.stringify(sessions));
    setHistory(prev => [...prev, newMessage]);
  };

  return (
    <div className="App">
      <AnimatedText text={welcomeText} scrollY={scrollY} />
      <AnimatedBlob scrollY={scrollY} currentState={currentState}>
        <VoiceButton
          onRecordingComplete={handleRecordingComplete}
          currentState={currentState}
          setCurrentState={setCurrentState}
          onError={handleError}
        />
      </AnimatedBlob>

      {showLock && localStorage.getItem('isAuthenticated') && (
        <button
          onClick={handleLogout}
          className="fixed top-4 right-4 px-4 py-2 text-black hover:italic rounded transition-colors"
        >
          🔐 Lock
        </button>
      )}

      <div className="grosse_div min-h-screen bg-gradient-to-br flex flex-col items-center justify-center p-4">
        {error && <ErrorMessage message={error} />}

        <div className="mt-4 text-center text-gray-500 text-sm sticky top-[550px]">
          Click the button and start speaking
        </div>

        {response && (
          <div className="w-full flex justify-center pb-4 px-4 bg-gradient-to-t from-white via-white to-transparent sticky top-[600px]">
            <div className="w-full max-w-[750px]">
              <h3 className="align-left text-xl font-semibold text-gray-800 mb-4 text-left">Recent Conversation</h3>
              <div className="bg-white rounded-lg shadow-sm p-4 border border-gray-200 mb-3">
                <div className="flex justify-between items-center mb-2">
                  <div className="text-xs text-gray-500">soijosd</div>
                  <div className="text-xs text-gray-500">
                    {history.length} message{history.length > 1 ? 's' : ''}
                  </div>
                </div>
                <div className="text-gray-800 text-left">
                  <div className="mb-2">
                    <span className="font-semibold">Therapist:</span> {response.text.replace(/^\s+|\s+$/g, '').replace(/["']/g, '')}
                  </div>
                  <div className="mt-2">
                    <audio controls className="w-full" ref={audioRef}>
                      <source src={response.audioUrl} type="audio/wav" />
                    </audio>
                  </div>
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
        )}

        <Footer scrollY={scrollY} />
      </div>
    </div>
  );
}
