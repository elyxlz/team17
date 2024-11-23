import { motion } from 'framer-motion';

export const Footer = () => {
  return (
    <div className="fixed bottom-0 left-0 right-0 flex justify-center items-center p-4">
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gray-800/80 text-white px-6 py-3 rounded-lg shadow-lg 
          backdrop-blur-sm max-w-[750px] w-11/12 text-center text-sm"
      >
        Your conversations are only stored locally on your device, it never leaves your control. 
        None of what you will say can be accessed or used. 🤫
      </motion.div>
    </div>
  );
}; 