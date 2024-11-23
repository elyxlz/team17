import React, { useState } from "react";
import { hashPassword } from "../utils/auth.ts";
import { setItem } from "../utils/storage.ts";
import { useNavigate } from 'react-router-dom';

const Register: React.FC = () => {
    const [password, setPassword] = useState("");
    const navigate = useNavigate();

    const handleRegister = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();

        const hashedPassword = await hashPassword(password);
        await setItem("user", { username: "user01", password: hashedPassword });

        alert("Registration successful!");
        setPassword("");
        navigate('/');
    };

    return (
        <form onSubmit={handleRegister}>
            <h2>Set Passcode</h2>
            <input
                type="password"
                placeholder="Enter passcode"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
            />
            <button type="submit">Set Passcode</button>
        </form>
    );
};

export default Register;