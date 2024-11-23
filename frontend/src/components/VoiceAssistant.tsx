import { useNavigate } from 'react-router-dom';
import { logout } from '../utils/auth';
// ... other existing imports ...

const VoiceAssistant = () => {
    const navigate = useNavigate();

    const handleLogout = async () => {
        await logout();
        navigate('/login');
    };

    return (
        <div style={{ position: 'relative', minHeight: '100vh' }}>
            <button 
                onClick={handleLogout}
                style={{
                    position: 'absolute',
                    top: '20px',
                    right: '20px',
                    padding: '8px 16px',
                    backgroundColor: '#f44336',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer'
                }}
            >
                Logout
            </button>

            {/* ... your existing VoiceAssistant content ... */}
        </div>
    );
};

export default VoiceAssistant; 