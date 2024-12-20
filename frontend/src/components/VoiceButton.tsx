import { useRef } from "react";
import { motion } from "framer-motion";
import { Loader2, Mic, MicOff, Speaker } from "lucide-react";

interface VoiceButtonProps {
  onRecordingComplete: (audioBlob: Blob) => void;
  currentState: "idle" | "listening" | "thinking" | "speaking";
  setCurrentState: (state: "idle" | "listening" | "thinking" | "speaking") => void;
  onError: (message: string) => void;
}

export default function VoiceButton({
  onRecordingComplete,
  currentState,
  setCurrentState,
  onError,
}: VoiceButtonProps) {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const stateIcons: Record<string, { icon: JSX.Element }> = {
    idle: { icon: <Mic className="w-8 h-8 text-gray-700" /> },
    listening: { icon: <MicOff className="w-8 h-8 text-red-700 animate-pulse" /> },
    thinking: { icon: <Loader2 className="w-8 h-8 text-purple-700 animate-spin" /> },
    speaking: { icon: <Speaker className="w-8 h-8 text-yellow-700 animate-pulse" /> },
  };

  const { icon } = stateIcons[currentState] || stateIcons.idle;

  const startRecording = async () => {
    try {
      setCurrentState("listening");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: getMimeType(),
      });
      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(chunksRef.current, { type: "audio/mp3" });
        onRecordingComplete(audioBlob);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorderRef.current.start();
    } catch (err: unknown) {
      if (err instanceof Error && err.name === "NotAllowedError") {
        setCurrentState("idle");
        onError("Microphone access denied. Please allow microphone access in your browser settings.");
      } else {
        setCurrentState("idle");
        onError("Could not access microphone. Please check your device settings.");
      }
      console.error("Error accessing microphone:", err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }
    setCurrentState("idle");
  };

  return (
    <motion.button
      className={`w-full h-full rounded-full flex items-center justify-center transition-colors duration-200 ease-in-out disabled:pointer-events-none
        ${currentState === "thinking" ? "cursor-default" : ""}
        `}
      whileHover={currentState !== "thinking" ? { scale: 1.1} : {}}
      onClick={currentState !== "thinking" ? (currentState === "listening" ? stopRecording : startRecording) : undefined}
      disabled={currentState === "thinking"}
    >
      {icon}
    </motion.button>
  );
}

// Check supported MIME types
const getMimeType = () => {
  const types = ["audio/wav", "audio/mp3", "audio/webm"];
  for (const type of types) {
    if (MediaRecorder.isTypeSupported(type)) {
      return type;
    }
  }
  return "audio/webm"; // fallback
};
