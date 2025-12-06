/* eslint-disable @typescript-eslint/no-explicit-any */
import api from '@/lib/api';
import { LoginRequest, RegisterRequest, AuthResponse, ApiResponse, User } from '@/types';

export const authService = {
  async login(credentials: LoginRequest): Promise<AuthResponse> {
    const response = await api.post('/auth/login', credentials, {
      withCredentials: true, // Để nhận cookies
    });
    return response.data;
  },

  async register(userData: RegisterRequest): Promise<{ message: string }> {
    const response = await api.post('/auth/register', userData, {
      withCredentials: true, // Để nhận cookies nếu cần
    });
    return { message: response.data };
  },

  async logout(): Promise<void> {
    // Simply call logout endpoint - no need to manually handle tokens
    await api.post('/auth/logout', {}, {
      withCredentials: true,
    });
  },

  async refreshToken(): Promise<AuthResponse> {
    // API Gateway will automatically handle refresh token from HttpOnly cookies
    const response = await api.post('/auth/refresh-token', {}, {
      withCredentials: true,
    });
    return response.data;
  },

  async verifyEmail(token: string): Promise<{ message: string }> {
    const response = await api.post(`/auth/verify-email?token=${token}`);
    return { message: response.data };
  },

  async resendVerificationEmail(token: string): Promise<{ message: string }> {
    const response = await api.post('/auth/resend-verification', { token });
    return { message: response.data };
  },

  async forgotPassword(email: string): Promise<{ message: string }> {
    const response = await api.post('/auth/forgot-password', { email });
    return { message: response.data };
  },

  async resetPassword(token: string, newPassword: string): Promise<{ message: string }> {
    const response = await api.post('/auth/reset-password', { token, newPassword });
    return { message: response.data };
  },

  async getCurrentUser(): Promise<ApiResponse<User>> {
    const response = await api.get('/auth/me', {
      withCredentials: true,
    });
    return response.data;
  },

  async getWebSocketToken(): Promise<{ token: string }> {
    // const response = await api.get('/auth/websocket-token', {
    //   withCredentials: true,
    // });
    // return response.data;
    return { token: '' }; // Placeholder until backend is implemented
  },

  /**
   * Kiểm tra xác thực người dùng và trả về thông tin người dùng nếu đã xác thực
   * @returns {Promise<{ isAuthenticated: boolean; user: User | null; error?: any }>}
   */
  checkAuthenticationWithUser: async (): Promise<{
    isAuthenticated: boolean;
    user: User | null;
    error?: any;
  }> => {
    console.log('DEBUG: ========== Starting checkAuthenticationWithUser ==========');
    console.log('DEBUG: Current URL:', typeof window !== 'undefined' ? window.location.href : 'server-side');

    try {
      console.log('DEBUG: Making API call to /auth/me');
      const response = await api.get('/auth/me', {
        withCredentials: true,
      });

      console.log('DEBUG: API response status:', response.status);
      console.log('DEBUG: API response data:', response.data.user);

      // Check if the response indicates authentication success
      if (response.data && response.status === 200) {
        console.log('DEBUG: Authentication successful, user data found');
        return {
          isAuthenticated: true,
          user: response.data.user,
        };
      } else if (response.data && !(response.status === 200)) {
        console.log('DEBUG: Authentication failed - response indicates failure');
        return {
          isAuthenticated: false,
          user: null,
        };
      } else {
        console.log('DEBUG: Unexpected response format:', response.data);
        return {
          isAuthenticated: false,
          user: null,
        };
      }
    } catch (error: any) {
      console.error('DEBUG: checkAuthenticationWithUser error:', error);
      console.error('DEBUG: Error status:', error.response?.status);
      console.error('DEBUG: Error data:', error.response?.data);

      return {
        isAuthenticated: false,
        user: null,
        error,
      };
    } finally {
      console.log('DEBUG: ========== checkAuthenticationWithUser completed ==========');
    }
  }
};
