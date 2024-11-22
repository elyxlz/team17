import React, { useState } from "react";
import { comparePassword } from "../utils/auth";
import { getItem } from "../utils/storage";
import { useNavigate } from "react-router-dom";

interface User {
    username: string;
    password: string;
}

const Login = () => {
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const navigate = useNavigate();

    const handleLogin = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();

        const user = await getItem("user") as User;
        if (user && user.username === username) {
            const isValidPassword = await comparePassword(password, user.password);
            if (isValidPassword) {
                alert("Login successful!");
                setUsername("");
                setPassword("");
                handleLoginSuccess();
            } else {
                alert("Invalid password.");
            }
        } else {
            alert("User not found.");
        }
    };

    const handleLoginSuccess = () => {
        // Set authentication flag
        localStorage.setItem('isAuthenticated', 'true');
        // Redirect to main page
        navigate('/');
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
            <div className="bg-white p-8 rounded-lg shadow-md w-96">
                <h2 className="text-2xl font-bold mb-4">Login</h2>
                <form onSubmit={handleLogin}>
                    <input
                        type="text"
                        placeholder="Username"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        required
                    />
                    <input
                        type="password"
                        placeholder="Password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                    />
                    <button type="submit">Login</button>
                </form>
            </div>
        </div>
    );
};

export default Login;