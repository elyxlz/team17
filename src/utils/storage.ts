import localforage from "localforage";

export const setItem = async (key: string, value: any) => {
    await localforage.setItem(key, value);
};

export const getItem = async (key: string) => {
    return localforage.getItem(key);
};

export const removeItem = async (key: string) => {
    await localforage.removeItem(key);
};