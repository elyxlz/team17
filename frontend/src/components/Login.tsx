import React, { useState } from "react";
import { comparePassword } from "../utils/auth";
import { getItem } from "../utils/storage";
import { useNavigate } from "react-router-dom";

interface User {
    password: string;
}

const Login = () => {
    const [password, setPassword] = useState("");
    const navigate = useNavigate();

    const handleLogin = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();

        const user = await getItem("user") as User;
        if (user) {
            const isValidPassword = await comparePassword(password, user.password);
            if (isValidPassword) {
                // Create a new chat session
                const newSessionId = Date.now().toString();
                const currentTime = new Date().toISOString();
                
                // Get existing sessions or initialize empty array
                const existingSessions = JSON.parse(localStorage.getItem('chatSessions') || '[]');
                
                // Add new session
                const newSession = {
                    sessionId: newSessionId,
                    startTime: currentTime,
                    messages: []
                };
                
                localStorage.setItem('chatSessions', JSON.stringify([...existingSessions, newSession]));
                localStorage.setItem('currentSessionId', newSessionId);
                localStorage.setItem('isAuthenticated', 'true');
                
                navigate('/');
            } else {
                alert("Invalid passcode.");
            }
        } else {
            alert("No passcode set. Please register first.");
            navigate('/register');
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
            <div className="bg-white p-8 rounded-lg shadow-md w-96">
                <h2 className="text-2xl font-bold mb-4">Enter Passcode</h2>
                <form onSubmit={handleLogin}>
                    <input
                        type="password"
                        placeholder="Enter passcode"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        className="w-full p-2 mb-4 border rounded"
                        required
                    />
                    <button 
                        type="submit"
                        className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600"
                    >
                        Unlock
                    </button>
                </form>
            </div>
        </div>
    );
};

export default Login;