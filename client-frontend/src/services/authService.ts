import api from '@/lib/api';
import { LoginRequest, RegisterRequest, AuthResponse, ApiResponse, User } from '@/types';

export const authService = {
  async login(credentials: LoginRequest): Promise<AuthResponse> {
    const response = await api.post('/api/auth/login', credentials, {
      withCredentials: true, // Để nhận cookies
    });
    return response.data;
  },

  async register(userData: RegisterRequest): Promise<{ message: string }> {
    const response = await api.post('/api/auth/register', userData, {
      withCredentials: true, // Để nhận cookies nếu cần
    });
    // Backend trả về message thông báo đã gửi email
    return { message: response.data };
  },

  async logout(): Promise<void> {
    // Backend expects refresh token in request body
    const refreshToken = this.getRefreshTokenFromCookie();
    await api.post('/api/auth/logout', { refreshToken }, {
      withCredentials: true,
    });
  },

  async refreshToken(): Promise<AuthResponse> {
    const refreshToken = this.getRefreshTokenFromCookie();
    const response = await api.post('/api/auth/refresh-token', { refreshToken }, {
      withCredentials: true,
    });
    return response.data;
  },

  async verifyEmail(token: string): Promise<{ message: string }> {
    const response = await api.post(`/api/auth/verify-email?token=${token}`);
    return { message: response.data };
  },

  async resendVerificationEmail(token: string): Promise<{ message: string }> {
    const response = await api.post('/api/auth/resend-verification', { token });
    return { message: response.data };
  },

  async forgotPassword(email: string): Promise<{ message: string }> {
    const response = await api.post('/api/auth/forgot-password', { email });
    return { message: response.data };
  },

  async resetPassword(token: string, newPassword: string): Promise<{ message: string }> {
    const response = await api.post('/api/auth/reset-password', { token, newPassword });
    return { message: response.data };
  },

  async getCurrentUser(): Promise<ApiResponse<User>> {
    const response = await api.get('/api/auth/me', {
      withCredentials: true,
    });
    return response.data;
  },

  // Helper methods for cookie management
  getAccessTokenFromCookie(): string | null {
    if (typeof document === 'undefined') return null;
    const cookies = document.cookie.split(';');
    const tokenCookie = cookies.find(cookie => cookie.trim().startsWith('accessToken='));
    return tokenCookie ? tokenCookie.split('=')[1] : null;
  },

  getRefreshTokenFromCookie(): string | null {
    if (typeof document === 'undefined') return null;
    const cookies = document.cookie.split(';');
    const tokenCookie = cookies.find(cookie => cookie.trim().startsWith('refreshToken='));
    return tokenCookie ? tokenCookie.split('=')[1] : null;
  },

  isTokenExpired(): boolean {
    const token = this.getAccessTokenFromCookie();
    if (!token) return true;

    try {
      const payload = JSON.parse(atob(token.split('.')[1]));
      return payload.exp * 1000 < Date.now();
    } catch {
      return true;
    }
  }
};
