import { motion } from 'framer-motion';

export const Footer = ({ scrollY }: { scrollY: number }) => {
  return (
    <div className="fixed bottom-0 left-0 right-0 flex justify-center items-center p-1">
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: scrollY > 0 ? 0 : 50 }}
        className="bg-gray-800/80 text-white px-3 py-2 rounded-lg shadow-lg 
          backdrop-blur-sm max-w-[950px] w-11/12 text-center text-sm"
      >
        Your conversations are only stored locally on your device, it never leaves your control. 
        None of what you will say can be accessed or used. 🤫
      </motion.div>
    </div>
  );
}; 