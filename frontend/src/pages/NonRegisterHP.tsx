import Register from '../components/Register';
import { VoiceAssistant } from '../components/VoiceAssistant';

function NonRegisterHP() {
  return (
    <div className="register-page">
      <div className="voice-assistant-container">
        <VoiceAssistant />
      </div>
      <div>
        Here to listen and guide you
      </div>
      <Register />
    </div>
  );
}

export default NonRegisterHP;