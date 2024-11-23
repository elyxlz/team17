import { motion } from "framer-motion";

interface AnimatedBlobProps {
  scrollY: number;
  children: React.ReactNode;
  currentState: "idle" | "listening" | "thinking" | "speaking";
}

export function AnimatedBlob({ scrollY, children, currentState }: AnimatedBlobProps) {
  const stateStyles: Record<
    string,
    { gradient: string; boxShadow: string }
  > = {
    idle: {
      gradient: "radial-gradient(circle at 0% 0%, lightgray, darkgray)",
      boxShadow:
        "0 -2vmin 4vmin lightgray inset, 0 1vmin 4vmin gray inset, 0 -2vmin 7vmin darkgray inset, 0 0 3vmin silver, 0 5vmin 4vmin gainsboro, 2vmin -2vmin 15vmin dimgray, 0 0 7vmin lightsteelblue",
    },
    listening: {
      gradient: "radial-gradient(circle at 0% 0%, lightblue, dodgerblue)",
      boxShadow:
        "0 -2vmin 4vmin lightblue inset, 0 1vmin 4vmin dodgerblue inset, 0 -2vmin 7vmin steelblue inset, 0 0 3vmin skyblue, 0 5vmin 4vmin deepskyblue, 2vmin -2vmin 15vmin blue, 0 0 7vmin cornflowerblue",
    },
    thinking: {
      gradient: "radial-gradient(circle at 0% 0%, hotpink, slateblue)",
      boxShadow:
        "0 -2vmin 4vmin hotpink inset, 0 1vmin 4vmin mediumvioletred inset, 0 -2vmin 7vmin slateblue inset, 0 0 3vmin orchid, 0 5vmin 4vmin plum, 2vmin -2vmin 15vmin mediumorchid, 0 0 7vmin purple",
    },
    speaking: {
      gradient: "radial-gradient(circle at 0% 0%, gold, orange)",
      boxShadow:
        "0 -2vmin 4vmin gold inset, 0 1vmin 4vmin orange inset, 0 -2vmin 7vmin goldenrod inset, 0 0 3vmin yellow, 0 5vmin 4vmin darkorange, 2vmin -2vmin 15vmin orangered, 0 0 7vmin coral",
    },
  };

  const { gradient, boxShadow } = stateStyles[currentState] || stateStyles.idle;

  return (
    <div className="flex justify-center items-center h-full z-50 sticky top-0">
      <motion.div
        className="animate-morph h-[50svh] w-[50svh] fixed"
        style={{
          background: gradient,
          boxShadow: boxShadow,
          borderRadius: "30% 70% 53% 47% / 26% 46% 54% 74%",
        }}
        initial={{
          translateY: 0,
          scale: 1,
          opacity: 0.2,
        }}
        animate={{
          translateY: scrollY > 0 ? -scrollY * 0.24 : 0,
          scale: 1 - scrollY * 0.0001,
          opacity: Math.min(0.2 + scrollY * 0.0005, 1),
        }}
        transition={{
          duration: 0.3,
          ease: "easeOut",
        }}
      >
        {children}
      </motion.div>
    </div>
  );
}
