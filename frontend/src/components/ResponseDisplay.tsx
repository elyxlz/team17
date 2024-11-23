import { motion } from 'framer-motion';
import type { AIResponse } from '../utils/openai';

interface ResponseDisplayProps {
  response: AIResponse | null;
  isVisible: boolean;
}

const ResponseDisplay = ({ response, isVisible }: ResponseDisplayProps) => {
  if (!isVisible || !response) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="max-w-md w-full bg-white rounded-lg shadow-lg p-6 mb-8 max-w-[750px]"
    >
      <p className="text-gray-800 mb-4">{response.text}</p>
      
      {response.audioUrl && (
        <div className="mt-4">
          <audio 
            controls 
            className="w-full"
            // Cleanup the URL when component unmounts
            onEnded={() => URL.revokeObjectURL(response.audioUrl!)}
          >
            <source src={response.audioUrl} type="audio/wav" />
            Your browser does not support the audio element.
          </audio>
        </div>
      )}
    </motion.div>
  );
};

export default ResponseDisplay;