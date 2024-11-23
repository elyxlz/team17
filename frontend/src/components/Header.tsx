import { Brain } from 'lucide-react';

export function Header() {
  return (
    <div className="text-center mb-12">
      <div className="flex items-center justify-center mb-4">
        <Brain className="w-12 h-12 text-blue-600 mr-3" />
        <h1 className="text-4xl font-bold text-gray-800">Voice Assistant</h1>
      </div>
      <p className="text-gray-600 text-lg">Speak your mind, and let AI assist you</p>
    </div>
  );
} 