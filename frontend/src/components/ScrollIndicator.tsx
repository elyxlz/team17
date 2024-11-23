import { motion } from 'framer-motion';

export const ScrollIndicator = () => {
  return (
    <motion.div 
      className="w-20 h-20 flex items-center justify-center rounded-full bg-white/10 backdrop-blur-sm"
      animate={{
        y: [0, 10, 0],
      }}
      transition={{
        duration: 1.5,
        repeat: Infinity,
        ease: "easeInOut",
      }}
    >
      <svg 
        width="60" 
        height="60" 
        viewBox="0 0 24 24" 
        fill="none" 
        className="text-black/80"
      >
        <path 
          d="M7 13L12 18L17 13" 
          stroke="currentColor" 
          strokeWidth="2" 
          strokeLinecap="round" 
          strokeLinejoin="round"
        />
        <path 
          d="M7 7L12 12L17 7" 
          stroke="currentColor" 
          strokeWidth="2" 
          strokeLinecap="round" 
          strokeLinejoin="round"
        />
      </svg>
    </motion.div>
  );
};