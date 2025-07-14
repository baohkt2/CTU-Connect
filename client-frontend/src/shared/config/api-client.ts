import axios, {
  AxiosInstance,
  AxiosResponse,
  AxiosError,
  AxiosRequestConfig,
} from 'axios';
import { API_CONFIG, ERROR_MESSAGES } from '@/shared/constants';

import { storage } from '@/shared/utils';
import {ApiError} from "@/shared/types/common";
import {ApiResponse} from "@/types";

interface ErrorResponseData {
  message?: string;
  code?: string;
  [key: string]: unknown;
}

class ApiClient {
  private instance: AxiosInstance;
  private isRefreshing = false;
  private failedQueue: Array<{
    resolve: (value?: AxiosResponse | void) => void;
    reject: (reason?: unknown) => void;
  }> = [];

  constructor() {
    this.instance = axios.create({
      baseURL: API_CONFIG.BASE_URL,
      timeout: API_CONFIG.TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    this.instance.interceptors.request.use(
        (config) => {
          const token = storage.getAuthToken();
          if (token) {
            config.headers.Authorization = `Bearer ${token}`;
          }
          return config;
        },
        (error) => Promise.reject(error)
    );

    this.instance.interceptors.response.use(
        (response) => response,
        async (error: AxiosError) => {
          const originalRequest = error.config as AxiosRequestConfig & {
            _retry?: boolean;
          };

          if (error.response?.status === 401 && !originalRequest._retry) {
            if (this.isRefreshing) {
              return new Promise((resolve, reject) => {
                this.failedQueue.push({ resolve, reject });
              }).then(() => {
                return this.instance(originalRequest);
              });
            }

            originalRequest._retry = true;
            this.isRefreshing = true;

            try {
              const refreshToken = storage.get('refresh_token');
              if (refreshToken) {
                const response = await this.instance.post('/auth/refresh', {
                  refreshToken,
                });

                const { accessToken } = response.data;
                storage.setAuthToken(accessToken);

                this.failedQueue.forEach(({ resolve }) => resolve());
                this.failedQueue = [];

                return this.instance(originalRequest);
              }
            } catch (refreshError: unknown) {
              storage.removeAuthToken();
              storage.removeUserData();
              storage.remove('refresh_token');

              if (typeof window !== 'undefined') {
                window.location.href = '/login';
              }
            } finally {
              this.isRefreshing = false;
            }
          }

          return Promise.reject(this.handleError(error));
        }
    );
  }

  private handleError(error: AxiosError): ApiError {
    const response = error.response;

    if (!response) {
      return {
        status: 0,
        message: ERROR_MESSAGES.NETWORK_ERROR,
        code: 'NETWORK_ERROR',
      };
    }

    const data = response.data as ErrorResponseData;
    const status = response.status;
    let message: string = ERROR_MESSAGES.SERVER_ERROR;

    switch (status) {
      case 400:
        message = data.message || ERROR_MESSAGES.VALIDATION_ERROR;
        break;
      case 401:
        message = ERROR_MESSAGES.UNAUTHORIZED;
        break;
      case 403:
        message = ERROR_MESSAGES.FORBIDDEN;
        break;
      case 404:
        message = ERROR_MESSAGES.NOT_FOUND;
        break;
      case 500:
        message = ERROR_MESSAGES.SERVER_ERROR;
        break;
      default:
        message = data.message || ERROR_MESSAGES.SERVER_ERROR;
    }

    return {
      status,
      message,
      code: data.code,
      details: data,
    };
  }

  async get<T>(url: string, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
    const response = await this.instance.get<ApiResponse<T>>(url, config);
    return response.data;
  }


  async post<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.instance.post<T>(url, data, config);
    return response.data;
  }

  async put<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.instance.put<T>(url, data, config);
    return response.data;
  }

  async patch<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.instance.patch<T>(url, data, config);
    return response.data;
  }

  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.instance.delete<T>(url, config);
    return response.data;
  }

  async uploadFile<T>(
      url: string,
      file: File,
      onProgress?: (progress: number) => void
  ): Promise<T> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.instance.post<T>(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (event) => {
        if (onProgress && event.total) {
          const progress = (event.loaded / event.total) * 100;
          onProgress(progress);
        }
      },
    });

    return response.data;
  }

  async downloadFile(url: string, filename: string): Promise<void> {
    const response = await this.instance.get(url, {
      responseType: 'blob',
    });

    const blob = new Blob([response.data]);
    const link = document.createElement('a');
    link.href = window.URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    window.URL.revokeObjectURL(link.href);
  }
}

export const apiClient = new ApiClient();
export default apiClient;
