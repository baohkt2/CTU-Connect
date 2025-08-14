import api from '@/lib/api';
import {ApiResponse, LecturerProfileUpdateRequest, PaginatedResponse, StudentProfileUpdateRequest, User} from '@/types';
import {categoryService} from './categoryService';

export const userService = {
  async getProfile(userId: string): Promise<User> {
    const response = await api.get(`/users/${userId}/profile`);
    return response.data;
  },

  async updateProfile(userData: Partial<User>): Promise<User> {
    console.log('Updating user profile with data:', userData);
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

  async updateMyProfile(userData: StudentProfileUpdateRequest | LecturerProfileUpdateRequest): Promise<User> {
    const response = await api.put('/users/me/profile', userData);
    return response.data;
  },

  async checkProfileCompletion(): Promise<boolean> {
    const response = await api.get('/users/checkMyInfo');
    return response.data;
  },

  // ========================= FRIENDS MANAGEMENT =========================

  // Get current user's friends
  async getMyFriends(): Promise<PaginatedResponse<User>> {
    const response = await api.get('/users/me/friends');
    return response.data;
  },

  // Get friend requests received
  async getFriendRequests(): Promise<User[]> {
    const response = await api.get('/users/me/friend-requests');
    return response.data;
  },

  // Get friend requests sent
  async getSentFriendRequests(): Promise<User[]> {
    const response = await api.get('/users/me/friend-requested');
    return response.data;
  },

  // Get friend suggestions
  async getFriendSuggestions(): Promise<PaginatedResponse<User>> {
    const response = await api.get('/users/me/friend-suggestions');
    return response.data;
  },

  // Get mutual friends with another user
  async getMutualFriends(otherUserId: string): Promise<PaginatedResponse<User>> {
    const response = await api.get(`/users/me/mutual-friends/${otherUserId}`);
    return response.data;
  },

  // Send friend request
  async sendFriendRequest(friendId: string): Promise<ApiResponse<string>> {
    const response = await api.post(`/users/me/invite/${friendId}`);
    return response.data;
  },

  // Accept friend request
  async acceptFriendRequest(friendId: string): Promise<ApiResponse<string>> {
    const response = await api.post(`/users/me/accept-invite/${friendId}`);
    return response.data;
  },

  // Reject friend request
  async rejectFriendRequest(friendId: string): Promise<ApiResponse<string>> {
    const response = await api.post(`/users/me/reject-invite/${friendId}`);
    return response.data;
  },

  // Remove friend
  async removeFriend(friendId: string): Promise<ApiResponse<string>> {
    const response = await api.delete(`/users/me/friends/${friendId}`);
    return response.data;
  }

};
