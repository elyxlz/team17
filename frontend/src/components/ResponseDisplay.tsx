import { motion } from 'framer-motion';

interface ResponseDisplayProps {
  response: {
    text: string;
    audioUrl?: string;
  } | null;
  isVisible: boolean;
}

export default function ResponseDisplay({ response, isVisible }: ResponseDisplayProps) {
  console.log('ResponseDisplay props:', { response, isVisible });
  
  if (!response || !isVisible) {
    console.log('Returning null because:', { noResponse: !response, notVisible: !isVisible });
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: isVisible ? 1 : 0, y: isVisible ? 0 : 20 }}
      transition={{ duration: 0.5 }}
      className="w-full max-w-2xl mt-8 p-6 bg-white rounded-lg shadow-lg"
    >
      <div className="flex items-start justify-between gap-4">
        <p className="text-gray-800 text-lg leading-relaxed flex-1">
          {response.text}
        </p>
      </div>
    </motion.div>
  );
}