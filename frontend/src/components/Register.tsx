import React, { useState } from "react";
import { hashPassword } from "../utils/auth.ts";
import { setItem } from "../utils/storage.ts";
import { useNavigate } from 'react-router-dom';
import '../styles/Register.css';

const Register: React.FC = () => {
    const [password, setPassword] = useState("");
    const navigate = useNavigate();

    const handleRegister = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        
        try {
            const hashedPassword = await hashPassword(password);
            await setItem("user", { username: "user01", password: hashedPassword });
            
            localStorage.setItem('isAuthenticated', 'true');

            alert("Registration successful!");
            setPassword("");
            navigate('/');
        } catch (error) {
            console.error('Registration error:', error);
            alert("Registration failed. Please try again.");
        }
    };

    return (
        <form onSubmit={handleRegister} className="register-form">
            <h2>Set Passcode</h2>
            <div className="form-group">
                <input
                    type="password"
                    placeholder="Enter passcode"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    className="password-input"
                />
            </div>
            <button type="submit" className="submit-button">Set Passcode</button>
        </form>
    );
};

export default Register;