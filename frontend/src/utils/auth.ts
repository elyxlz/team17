import { removeItem } from "./storage.ts";
import bcrypt from "bcryptjs";

const SALT_ROUNDS = 10;

export const hashPassword = async (password: string): Promise<string> => {
    const salt = await bcrypt.genSalt(SALT_ROUNDS);
    return bcrypt.hash(password, salt);
};

export const comparePassword = async (password: string, hashedPassword: string): Promise<boolean> => {
    return bcrypt.compare(password, hashedPassword);
};

export const logout = async (): Promise<void> => {
    await removeItem("user");
    localStorage.removeItem('isAuthenticated');
    alert("Logged out successfully.");
};


