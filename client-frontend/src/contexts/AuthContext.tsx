'use client';

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { User, LoginRequest, RegisterRequest } from '@/types';
import { authService } from '@/services/authService';

interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (credentials: {
    identifier: string;
    password: string;
    recaptchaToken: string
  }) => Promise<void>;
  register: (userData: RegisterRequest) => Promise<void>;
  logout: () => void;
  updateUser: (userData: Partial<User>) => void;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [isHydrated, setIsHydrated] = useState(false);

  useEffect(() => {
    setIsHydrated(true);
  }, []);

  useEffect(() => {
    if (!isHydrated) return;

    const checkAuth = async () => {
      try {
        // Check if we have tokens
        const hasAccessToken = authService.getAccessTokenFromCookie();
        const hasRefreshToken = authService.getRefreshTokenFromCookie();

        if (!hasAccessToken && !hasRefreshToken) {
          // No tokens at all, user is not authenticated
          setLoading(false);
          return;
        }

        // Try to get current user
        if (hasAccessToken && !authService.isTokenExpired()) {
          // Access token is valid, try to get user
          try {
            const response = await authService.getCurrentUser();
            if (response.success && response.data) {
              setUser(response.data);
              setLoading(false);
              return;
            }
          } catch (error) {
            console.log('Failed to get user with current access token, trying refresh...');
          }
        }

        // Access token is expired or invalid, try refresh
        if (hasRefreshToken) {
          try {
            const refreshResponse = await authService.refreshToken();
            if (refreshResponse.user) {
              setUser(refreshResponse.user);
            } else {
              // Try to get user after refresh
              const userResponse = await authService.getCurrentUser();
              if (userResponse.success && userResponse.data) {
                setUser(userResponse.data);
              }
            }
          } catch (refreshError) {
            console.error('Token refresh failed:', refreshError);
            // Clear any stale tokens and user state
            setUser(null);
            // Clear cookies by calling logout (but don't redirect)
            try {
              await authService.logout();
            } catch (logoutError) {
              console.error('Logout error during refresh failure:', logoutError);
            }
          }
        }
      } catch (error) {
        console.error('Auth check error:', error);
        setUser(null);
      }

      setLoading(false);
    };

    checkAuth();
  }, [isHydrated]);

  const login = async (credentials: LoginRequest) => {
    try {
      const response = await authService.login(credentials);
      // Backend trả về user trong response, tokens được set trong cookies
      console.log('Login response:', response);
      if (response.user) {
        setUser(response.user);
      }
    } catch (error) {
      throw error;
    }
  };

  const register = async (userData: RegisterRequest) => {
    try {
      // Chỉ gọi API đăng ký, không auto-login
      await authService.register(userData);
      // Không set user vì cần xác thực email trước
    } catch (error) {
      throw error;
    }
  };

  const logout = async () => {
    try {
      await authService.logout();
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear user state
      setUser(null);
      // Redirect to login page
      if (typeof window !== 'undefined') {
        window.location.href = '/login';
      }
    }
  };

  const updateUser = (userData: Partial<User>) => {
    if (user) {
      setUser({ ...user, ...userData });
    }
  };

  const value = {
    user,
    loading,
    login,
    register,
    logout,
    updateUser,
    isAuthenticated: !!user,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
