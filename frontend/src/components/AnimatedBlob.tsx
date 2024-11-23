import React from "react";
import { motion } from "framer-motion";

interface AnimatedBlobProps {
  scrollY: number;
  children: React.ReactNode;
  isRecording: boolean;
}

export function AnimatedBlob({ scrollY, children, isRecording }: AnimatedBlobProps) {
  return (
    <div className="flex justify-center items-center h-full z-50 sticky top-0">
      <motion.div
        className={`h-[50svh] w-[50svh] fixed`}
        style={{
          background: isRecording
            ? "radial-gradient(circle at 50% 50%, red, darkred)" // Subtle red gradient
            : "radial-gradient(circle at 0% 0%, hotpink, slateblue)", // Default gradient
          boxShadow: isRecording
            ? "0 0 10px rgba(255, 0, 0, 0.8), 0 0 20px rgba(255, 0, 0, 0.6)" // Subtle red glow
            : "0 -2vmin 4vmin LightPink inset, 0 1vmin 4vmin MediumPurple inset, 0 -2vmin 7vmin purple inset, 0 0 3vmin Thistle, 0 5vmin 4vmin Orchid, 2vmin -2vmin 15vmin MediumSlateBlue, 0 0 7vmin MediumOrchid", // Default shadows
          borderRadius: "30% 70% 53% 47% / 26% 46% 54% 74%", // Keep morphing
        }}
        initial={{
          translateY: 0,
          scale: 0.7,
        }}
        animate={{
          translateY: -scrollY * 0.1, // Move based on scroll
          scale: isRecording ? 0.8 : 1 - scrollY * 0.0001, // Slightly grow during recording
          opacity: 1, // Keep opacity constant
        }}
        transition={{
          duration: isRecording ? 1.5 : 0.3, // Slower scale effect during recording
          ease: "easeInOut", // Smooth transition
          repeat: isRecording ? Infinity : 0, // Continuous scale effect when recording
          repeatType: "reverse", // Subtle back-and-forth pulse
        }}
      >
        {children}
      </motion.div>
    </div>
  );
}
