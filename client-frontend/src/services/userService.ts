/* eslint-disable @typescript-eslint/no-explicit-any */
import api from '@/lib/api';
import {ApiResponse, LecturerProfileUpdateRequest, PaginatedResponse, StudentProfileUpdateRequest, User} from '@/types';


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
    // Transform frontend data to backend format
    const backendData: any = {
      fullName: userData.fullName,
      bio: userData.bio,
      avatarUrl: userData.avatarUrl,
      backgroundUrl: userData.backgroundUrl,
    };
    
    // For students
    if ('studentId' in userData) {
      backendData.studentId = userData.studentId;
      backendData.majorCode = userData.majorCode;
      backendData.batchYear = userData.batchYear;
      backendData.genderName = userData.genderCode; // Map genderCode (e.g., "M") to genderName field
    }
    
    // For lecturers (if needed in future)
    if ('staffCode' in userData) {
      // Add lecturer specific fields if needed
    }
    
    console.log('Sending update profile request:', backendData);
    const response = await api.put('/users/me/profile', backendData);
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

  // Get friend suggestions (old endpoint - deprecated)
  async getFriendSuggestions(): Promise<PaginatedResponse<User>> {
    const response = await api.get('/users/me/friend-suggestions');
    return response.data;
  },

  // Get friend suggestions with filters (NEW - enhanced version)
  async searchFriendSuggestions(params?: {
    query?: string;
    college?: string;
    faculty?: string;
    batch?: string;
    limit?: number;
  }): Promise<User[]> {
    const queryParams = new URLSearchParams();
    if (params?.query) queryParams.append('query', params.query);
    if (params?.college) queryParams.append('college', params.college);
    if (params?.faculty) queryParams.append('faculty', params.faculty);
    if (params?.batch) queryParams.append('batch', params.batch);
    if (params?.limit) queryParams.append('limit', params.limit.toString());
    
    const response = await api.get(`/users/friend-suggestions/search?${queryParams.toString()}`);
    return response.data;
  },

  // Search user by email
  async searchUserByEmail(email: string): Promise<User> {
    const response = await api.get(`/users/search/email?email=${encodeURIComponent(email)}`);
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
  },

  // Get friendship status with another user
  async getFriendshipStatus(targetUserId: string): Promise<{ status: 'none' | 'friends' | 'sent' | 'received' | 'self' }> {
    const response = await api.get(`/users/${targetUserId}/friendship-status`);
    return response.data;
  },

  // Get mutual friends with another user
  async getMutualFriendsWithUser(targetUserId: string, page = 0, size = 20): Promise<PaginatedResponse<User>> {
    const response = await api.get(`/users/${targetUserId}/mutual-friends?page=${page}&size=${size}`);
    return response.data;
  },

  // Get mutual friends count
  async getMutualFriendsCount(targetUserId: string): Promise<number> {
    const response = await api.get(`/users/${targetUserId}/mutual-friends-count`);
    return response.data.count;
  }

};
