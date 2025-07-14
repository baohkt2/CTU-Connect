/*
import { apiClient } from '@/shared/config/api-client';
import { API_ENDPOINTS } from '@/shared/constants';
import { createApiUrl } from '@/shared/utils';
import {
  User,
  UpdateUserRequest,
  PaginatedResponse,
  ApiResponse,
  UserStats,
  FileUploadResponse,
} from '@/shared/types';

/!**
 * User Service
 * Handles all user-related API calls
 *!/
export class UserService {
  /!**
   * Get user profile by ID
   *!/
  async getUserProfile(userId: string): Promise<User> {
    const url = createApiUrl(API_ENDPOINTS.USERS.PROFILE, { id: userId });
    return apiClient.get<User>(url);
  }

  /!**
   * Update current user profile
   *!/
  async updateProfile(userData: UpdateUserRequest): Promise<User> {
    return apiClient.put<User>(API_ENDPOINTS.USERS.PROFILE, userData);
  }

  /!**
   * Upload user avatar
   *!/
  async uploadAvatar(
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<FileUploadResponse> {
    return apiClient.uploadFile<FileUploadResponse>(
      API_ENDPOINTS.USERS.AVATAR,
      file,
      onProgress
    );
  }

  /!**
   * Search users
   *!/
  async searchUsers(
    query: string,
    page = 0,
    size = 10
  ): Promise<PaginatedResponse<User>> {
    const url = createApiUrl(API_ENDPOINTS.USERS.SEARCH, undefined, {
      q: query,
      page,
      size,
    });
    return apiClient.get<PaginatedResponse<User>>(url);
  }

  /!**
   * Follow user
   *!/
  async followUser(userId: string): Promise<ApiResponse<null>> {
    const url = createApiUrl(API_ENDPOINTS.USERS.FOLLOW, { id: userId });
    return apiClient.post<ApiResponse<null>>(url);
  }

  /!**
   * Unfollow user
   *!/
  async unfollowUser(userId: string): Promise<ApiResponse<null>> {
    const url = createApiUrl(API_ENDPOINTS.USERS.FOLLOW, { id: userId });
    return apiClient.delete<ApiResponse<null>>(url);
  }

  /!**
   * Get user followers
   *!/
  async getFollowers(
    userId: string,
    page = 0,
    size = 10
  ): Promise<PaginatedResponse<User>> {
    const url = createApiUrl(
      API_ENDPOINTS.USERS.FOLLOWERS,
      { id: userId },
      { page, size }
    );
    return apiClient.get<PaginatedResponse<User>>(url);
  }

  /!**
   * Get users that user is following
   *!/
  async getFollowing(
    userId: string,
    page = 0,
    size = 10
  ): Promise<PaginatedResponse<User>> {
    const url = createApiUrl(
      API_ENDPOINTS.USERS.FOLLOWING,
      { id: userId },
      { page, size }
    );
    return apiClient.get<PaginatedResponse<User>>(url);
  }

  /!**
   * Get user friends
   *!/
  async getFriends(
    userId: string,
    page = 0,
    size = 10
  ): Promise<PaginatedResponse<User>> {
    const url = createApiUrl(
      API_ENDPOINTS.USERS.FRIENDS,
      { id: userId },
      { page, size }
    );
    return apiClient.get<PaginatedResponse<User>>(url);
  }

  /!**
   * Get user statistics
   *!/
  async getUserStats(userId: string): Promise<UserStats> {
    const url = createApiUrl('/users/:id/stats', { id: userId });
    return apiClient.get<UserStats>(url);
  }

  /!**
   * Check if user is following another user
   *!/
  async isFollowing(userId: string): Promise<boolean> {
    const url = createApiUrl('/users/:id/following/check', { id: userId });
    const response = await apiClient.get<{ isFollowing: boolean }>(url);
    return response.isFollowing;
  }
}

// Export singleton instance
export const userService = new UserService();
*/
