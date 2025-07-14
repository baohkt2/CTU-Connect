import { STORAGE_KEYS } from '@/shared/constants';

/**
 * Local storage utilities with type safety
 */
export const storage = {
  // Generic get method
  get: <T>(key: string): T | null => {
    if (typeof window === 'undefined') return null;

    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : null;
    } catch (error) {
      console.error(`Error getting ${key} from localStorage:`, error);
      return null;
    }
  },

  // Generic set method
  set: <T>(key: string, value: T): void => {
    if (typeof window === 'undefined') return;

    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error(`Error setting ${key} in localStorage:`, error);
    }
  },

  // Remove item
  remove: (key: string): void => {
    if (typeof window === 'undefined') return;

    try {
      localStorage.removeItem(key);
    } catch (error) {
      console.error(`Error removing ${key} from localStorage:`, error);
    }
  },

  // Clear all storage
  clear: (): void => {
    if (typeof window === 'undefined') return;

    try {
      localStorage.clear();
    } catch (error) {
      console.error('Error clearing localStorage:', error);
    }
  },

  // Auth token methods
  getAuthToken: (): string | null => {
    return storage.get<string>(STORAGE_KEYS.AUTH_TOKEN);
  },

  setAuthToken: (token: string): void => {
    storage.set(STORAGE_KEYS.AUTH_TOKEN, token);
  },

  removeAuthToken: (): void => {
    storage.remove(STORAGE_KEYS.AUTH_TOKEN);
  },

  // User data methods
  getUserData: (): any | null => {
    return storage.get(STORAGE_KEYS.USER_DATA);
  },

  setUserData: (user: any): void => {
    storage.set(STORAGE_KEYS.USER_DATA, user);
  },

  removeUserData: (): void => {
    storage.remove(STORAGE_KEYS.USER_DATA);
  },
};

/**
 * Session storage utilities
 */
export const sessionStorage = {
  get: <T>(key: string): T | null => {
    if (typeof window === 'undefined') return null;

    try {
      const item = window.sessionStorage.getItem(key);
      return item ? JSON.parse(item) : null;
    } catch (error) {
      console.error(`Error getting ${key} from sessionStorage:`, error);
      return null;
    }
  },

  set: <T>(key: string, value: T): void => {
    if (typeof window === 'undefined') return;

    try {
      window.sessionStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error(`Error setting ${key} in sessionStorage:`, error);
    }
  },

  remove: (key: string): void => {
    if (typeof window === 'undefined') return;

    try {
      window.sessionStorage.removeItem(key);
    } catch (error) {
      console.error(`Error removing ${key} from sessionStorage:`, error);
    }
  },

  clear: (): void => {
    if (typeof window === 'undefined') return;

    try {
      window.sessionStorage.clear();
    } catch (error) {
      console.error('Error clearing sessionStorage:', error);
    }
  },
};
