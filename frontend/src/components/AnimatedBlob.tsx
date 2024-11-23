import { motion } from 'framer-motion';

interface AnimatedBlobProps {
  scrollY: number;
}

export function AnimatedBlob({ scrollY }: AnimatedBlobProps) {
  return (
    <motion.div
      className="animated-blob"
      style={{
        transform: `translate(-50%, -50%) translateY(${scrollY * -0.05}px) scale(${1 - scrollY * 0.0007})`,
        opacity: 0.25 + scrollY * 0.0005,
      }}
    />
  );
} 