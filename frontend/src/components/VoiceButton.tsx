import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { Mic, MicOff } from 'lucide-react';

interface VoiceButtonProps {
  onRecordingComplete: (audioBlob: Blob) => void;
  isProcessing: boolean;
  onError: (message: string) => void;
}

export default function VoiceButton({ onRecordingComplete, isProcessing, onError }: VoiceButtonProps) {
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: getMimeType()
      });
      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/mp3' });
        onRecordingComplete(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (err: any) {
      if (err.name === 'NotAllowedError') {
        onError('Microphone access denied. Please allow microphone access in your browser settings.');
      } else {
        onError('Could not access microphone. Please check your device settings.');
      }
      console.error('Error accessing microphone:', err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const buttonVariants = {
    idle: { scale: 1 },
    recording: { scale: 1.1, boxShadow: "0 0 0 0 rgba(239, 68, 68, 0.7)" },
    processing: { scale: 1, opacity: 0.7 }
  };

  const pulseAnimation = {
    recording: {
      scale: [1, 1.2, 1],
      boxShadow: [
        "0 0 0 0 rgba(239, 68, 68, 0.7)",
        "0 0 0 20px rgba(239, 68, 68, 0)",
        "0 0 0 0 rgba(239, 68, 68, 0)"
      ],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: "easeInOut"
      }
    }
  };

  return (
    <motion.button
      className={`w-32 h-32 rounded-full flex items-center justify-center text-white
        ${isRecording ? 'bg-red-500' : 'bg-blue-500'}
        ${isProcessing ? 'opacity-70 cursor-wait' : 'hover:bg-opacity-90'}
        transition-colors duration-200 ease-in-out`}
      variants={buttonVariants}
      animate={isProcessing ? "processing" : isRecording ? "recording" : "idle"}
      whileHover={!isProcessing && !isRecording ? { scale: 1.05 } : {}}
      onClick={!isProcessing ? (isRecording ? stopRecording : startRecording) : undefined}
      disabled={isProcessing}
    >
      <motion.div
        className="absolute w-full h-full rounded-full"
        variants={pulseAnimation}
        animate={isRecording ? "recording" : "idle"}
      />
      {isRecording ? (
        <MicOff className="w-12 h-12" />
      ) : (
        <Mic className="w-12 h-12" />
      )}
    </motion.button>
  );
}

// Check supported MIME types
const getMimeType = () => {
  const types = [
    'audio/wav',
    'audio/mp3',
    'audio/webm',
  ];
  
  for (const type of types) {
    if (MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }
  return 'audio/webm';  // fallback
};