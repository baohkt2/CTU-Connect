import api from '@/lib/api';
import {ApiResponse, FacultyProfileUpdateRequest, PaginatedResponse, StudentProfileUpdateRequest, User} from '@/types';
import {categoryService} from './categoryService';

export const userService = {
  async getProfile(userId: string): Promise<User> {
    const response = await api.get(`/users/${userId}`);
    return response.data;
  },

  async updateProfile(userData: Partial<User>): Promise<User> {
    const response = await api.put('/users/profile', userData);
    return response.data;
  },

  async uploadAvatar(file: File): Promise<ApiResponse<string>> {
    const formData = new FormData();
    formData.append('avatar', file);
    const response = await api.post('/users/avatar', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  async searchUsers(query: string, page = 0, size = 10): Promise<PaginatedResponse<User>> {
    const response = await api.get(`/users/search?q=${query}&page=${page}&size=${size}`);
    return response.data;
  },

  async followUser(userId: string): Promise<ApiResponse<null>> {
    const response = await api.post(`/users/${userId}/follow`);
    return response.data;
  },

  async unfollowUser(userId: string): Promise<ApiResponse<null>> {
    const response = await api.delete(`/users/${userId}/follow`);
    return response.data;
  },

  async getFollowers(userId: string, page = 0, size = 10): Promise<PaginatedResponse<User>> {
    const response = await api.get(`/users/${userId}/followers?page=${page}&size=${size}`);
    return response.data;
  },

  async getFollowing(userId: string, page = 0, size = 10): Promise<PaginatedResponse<User>> {
    const response = await api.get(`/users/${userId}/following?page=${page}&size=${size}`);
    return response.data;
  },

  async getFriends(userId: string, page = 0, size = 10): Promise<PaginatedResponse<User>> {
    const response = await api.get(`/users/${userId}/friends?page=${page}&size=${size}`);
    return response.data;
  },

  async getMyProfile(): Promise<User> {
    const response = await api.get('/users/me/profile');
    return response.data;
  },

  async updateMyProfile(userData: StudentProfileUpdateRequest | FacultyProfileUpdateRequest): Promise<User> {
    const response = await api.put('/users/me/profile', userData);
    return response.data;
  },

  async checkProfileCompletion(): Promise<boolean> {
    const response = await api.get('/users/checkMyInfo');
    return response.data;
  },

  // Delegate category operations to categoryService
  ...categoryService,
};
