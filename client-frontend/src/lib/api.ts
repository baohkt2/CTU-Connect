import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8090/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true, // Quan trọng: để gửi và nhận cookies
});

// Request interceptor - không cần thêm Authorization header vì dùng cookies
api.interceptors.request.use(
  (config) => {
    // Có thể thêm CSRF token nếu cần
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor để handle errors và auto-refresh token
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        // Thử refresh token
        await api.post('/api/auth/refresh-token', {
          refreshToken: getRefreshTokenFromCookie()
        });

        // Retry original request
        return api(originalRequest);
      } catch (refreshError) {
        // Refresh token failed, redirect to login
        if (typeof window !== 'undefined') {
          window.location.href = '/login';
        }
        return Promise.reject(refreshError);
      }
    }

    return Promise.reject(error);
  }
);

// Helper function to get refresh token from cookie
function getRefreshTokenFromCookie(): string | null {
  if (typeof document === 'undefined') return null;
  const cookies = document.cookie.split(';');
  const tokenCookie = cookies.find(cookie => cookie.trim().startsWith('refreshToken='));
  return tokenCookie ? tokenCookie.split('=')[1] : null;
}

export default api;
