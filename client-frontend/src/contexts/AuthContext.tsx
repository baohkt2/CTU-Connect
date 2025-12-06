'use client';
import React, { createContext, useContext, useState, useEffect, ReactNode, useCallback } from 'react';
import { User, LoginRequest, RegisterRequest } from '@/types';
import { authService } from '@/services/authService';
import { usePathname } from 'next/navigation';
import {userService} from "@/services/userService";

interface AuthContextType {
  user: User | null;
  token: string | null;
  loading: boolean;
  login: (credentials: LoginRequest) => Promise<void>;
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

// Public routes that don't require authentication
const PUBLIC_ROUTES = [
    '/login',
  '/register',
  '/change-password',
    '/forgot-password',
  '/verify-email',
  '/resend-verification',
    '/reset-password',
];

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [isHydrated, setIsHydrated] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    setIsHydrated(true);
  }, []);

  useEffect(() => {
    if (!isHydrated) return;

    // Check if current route is public
    const isPublicRoute = PUBLIC_ROUTES.some(route => pathname.startsWith(route));

    const checkAuth = async () => {
      console.log('DEBUG: ==> Starting checkAuth for route:', pathname);
      console.log('DEBUG: ==> Is public route:', isPublicRoute);
      console.log('DEBUG: ==> PUBLIC_ROUTES:', PUBLIC_ROUTES);

      try {
        // For public routes, skip auth check and set loading to false
        if (isPublicRoute) {
          console.log('DEBUG: Public route detected, skipping auth check:', pathname);
          setLoading(false);
          return;
        }

        console.log('DEBUG: Checking authentication for protected route:', pathname);

        // Use the enhanced authentication method that returns both auth status and user data
        const authResult = await authService.checkAuthenticationWithUser();

        console.log('DEBUG: Auth result:', authResult);

        if (authResult.isAuthenticated) {
          console.log('DEBUG: User authenticated, setting user data');
          const myProfile = await userService.getMyProfile()
          setUser(myProfile);

          console.log('DEBUG: myProfile :', myProfile);
          if (!await userService.checkProfileCompletion()) {
            console.log('DEBUG: User profile not completed, redirecting to profile update');
            // Redirect to profile update page if profile is not completed
           if (typeof window !== 'undefined' && window.location.pathname !== '/profile/update') {
              window.location.replace('/profile/update');
            }
          }
        } else {
          console.log('DEBUG: User not authenticated, clearing user data');
          setUser(null);

          // If we have an error, log it
          if (authResult.error) {
            console.log('DEBUG: Authentication error:', authResult.error);
          }
        }
      } catch (error) {
        console.log('DEBUG: Auth check failed with exception:', error);
        setUser(null);
      } finally {
        // Always set loading to false when auth check is complete
        console.log('DEBUG: Setting loading to false');
        setLoading(false);
      }
    };

    checkAuth();
  }, [isHydrated, pathname]);

  // Function to get WebSocket token when needed
  const getWebSocketToken = useCallback(async () => {
    if (!user) return null;

    try {
      const response = await authService.getWebSocketToken();
      setToken(response.token);
      return response.token;
    } catch (error) {
      console.error('Failed to get WebSocket token:', error);
      return null;
    }
  }, [user]);

  // Get WebSocket token after user login
  useEffect(() => {
    if (user && !token) {
      getWebSocketToken();
    }
  }, [user, token, getWebSocketToken]);

  const login = async (credentials: LoginRequest) => {
    try {
      const response = await authService.login(credentials);
      console.log('Login response:', response);
      if (response.user) {
        setUser(response.user);
        // Get WebSocket token after successful login
        const wsToken = await authService.getWebSocketToken();
        setToken(wsToken.token);
      }
    } catch (error) {
      throw error;
    }
  };

  const register = async (userData: RegisterRequest) => {
    try {
      await authService.register(userData);
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
      // Clear user state and token
      setUser(null);
      setToken(null);
      // Redirect to login page
      if (typeof window !== 'undefined') {
        window.location.replace('/');
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
    token,
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
