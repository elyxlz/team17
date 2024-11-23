import { motion } from 'framer-motion';
import { ScrollIndicator } from './ScrollIndicator';

interface AnimatedTextProps {
  text: string[];
  scrollY: number;
}

export function AnimatedText({ text, scrollY }: AnimatedTextProps) {
  return (
    <div className="relative flex flex-col items-center min-h-[200px]">
      <div className="text-container text-center"
        style={{ opacity: 1 - scrollY * 0.003 }}
      >
        {text.map((el, i) => (
          <motion.span
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{
              duration: 2,
              delay: i / 5,
            }}
            key={i}
          >
            {el.toLowerCase() === 'therapist' ? (
              <motion.i
                initial={{ fontStyle: "normal" }}
                animate={{ fontStyle: "italic" }}
                transition={{
                  duration: 1,
                  delay: (i / 5)  // Start italic animation after the fade-in
                }}
              >
                {el}
              </motion.i>
            ) : el}{" "}
          </motion.span>
        ))}
      </div>
      <div className="w-full flex justify-center mt-12">
        <ScrollIndicator />
      </div>
    </div>
  );
} 