import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { VoiceAssistant } from './components/VoiceAssistant';
import Register from './components/Register';
import Login from './components/Login';
import ProtectedRoute from './components/ProtectedRoute';
import { Conversations } from './pages/Conversations';
import NonRegisterHP from './pages/NonRegisterHP';
import "./styles.css";

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(localStorage.getItem('isAuthenticated') === 'true');

  useEffect(() => {
    const handleStorageChange = () => {
      setIsAuthenticated(localStorage.getItem('isAuthenticated') === 'true');
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
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
      const aiResponse = await processMessageWithOpenAI(audioBlob);
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
            isAuthenticated 
              ? <Navigate to="/" /> 
              : <NonRegisterHP />
          } 
        />
        <Route 
          path="/login" 
          element={
            isAuthenticated 
              ? <Navigate to="/" /> 
              : <Login setIsAuthenticated={setIsAuthenticated} />
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
