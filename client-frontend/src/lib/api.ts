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
    // API Gateway sẽ tự động đọc tokens từ HttpOnly cookies
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor để handle 401 errors từ API Gateway
api.interceptors.response.use(
  (response) => {
    if (response.status === 201) {
      console.log('DEBUG: Created resource:', response.config.url);
    } else {
      console.log('DEBUG: API Response successful:', response.config.url, response.data);
    }
    return response;
  },

  async (error) => {
    const originalRequest = error.config;
    console.log('DEBUG: API Error interceptor triggered:', originalRequest.url, error.response?.status);

    // Nếu nhận 401 từ API Gateway
    if (error.response?.status === 401) {
      console.log('DEBUG: 401 Unauthorized received for:', originalRequest.url);

      // IMPORTANT: Don't intercept logout requests - let them fail naturally
      if (originalRequest.url?.includes('/auth/logout')) {
        console.log('DEBUG: Logout request failed, ignoring');
        return Promise.reject(error);
      }

      // Nếu đây là request refresh-token thất bại, logout user
      if (originalRequest.url?.includes('/auth/refresh-token')) {
        console.log('DEBUG: Refresh token failed, redirecting to login');
        // Clear any auth state and redirect WITHOUT calling logout API
        if (typeof window !== 'undefined') {
          localStorage.removeItem('user');
          sessionStorage.clear();
          // Redirect to login
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }

      // Nếu chưa thử refresh token, thử refresh token
      if (!originalRequest._retry) {
        originalRequest._retry = true;
        console.log('DEBUG: Attempting token refresh for failed request:', originalRequest.url);

        try {
          // Thử refresh token thông qua API Gateway
          const refreshResponse = await api.post('/auth/refresh-token', {}, {
            withCredentials: true
          });
          console.log('DEBUG: Token refresh successful, retrying original request');

          // Retry original request sau khi refresh thành công
          return api(originalRequest);
        } catch (refreshError) {
          console.log('DEBUG: Token refresh failed, redirecting to login');
          // Refresh thất bại, clear local data and redirect WITHOUT calling logout API
          if (typeof window !== 'undefined') {
            localStorage.removeItem('user');
            sessionStorage.clear();
            window.location.href = '/login';
          }
          return Promise.reject(refreshError);
        }
      } else {
        console.log('DEBUG: Already retried request, not attempting refresh again');
      }
    }

    console.log('DEBUG: Non-401 error or already handled:', error.response?.status);
    return Promise.reject(error);
  }
);

export default api;
