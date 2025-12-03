import { apiClient } from '@/shared/config/api-client';
import { API_ENDPOINTS } from '@/shared/constants';
import { createApiUrl } from '@/shared/utils';
import {
  User,
  UserProfile,
  UpdateProfileRequest,
  PaginatedResponse,
  ApiResponse,
} from '@/shared/types';

/**
 * User Service - Updated to sync with backend APIs
 */
export class UserService {
  /**
   * Get current user profile
   */
  async getCurrentUser(): Promise<User> {
    return apiClient.get<User>(API_ENDPOINTS.USERS.PROFILE);
  }

  /**
   * Get user by ID
   */
  async getUser(userId: string): Promise<User> {
    const url = createApiUrl(API_ENDPOINTS.USERS.BY_ID, { id: userId });
    return apiClient.get<User>(url);
  }

  /**
   * Update user profile
   */
  async updateProfile(updateData: UpdateProfileRequest): Promise<User> {
    return apiClient.put<User>(API_ENDPOINTS.USERS.UPDATE_PROFILE, updateData);
  }

  /**
   * Search users - Updated to match EnhancedUserController
   */
  async searchUsers(
    query: string,
    faculty?: string,
    major?: string,
    batch?: string,
    page = 0,
    size = 20
  ): Promise<User[]> {
    const url = createApiUrl(API_ENDPOINTS.USERS.SEARCH, undefined, {
      query,
      faculty,
      major,
      batch,
      page,
      size,
    });
    return apiClient.get<User[]>(url);
  }

  /**
   * Get friend suggestions - Updated to match backend
   */
  async getFriendSuggestions(limit = 10): Promise<any[]> {
    const url = createApiUrl(API_ENDPOINTS.USERS.FRIEND_SUGGESTIONS, undefined, { limit });
    return apiClient.get<any[]>(url);
  }

  /**
   * Send friend request - Updated to match backend
   */
  async sendFriendRequest(targetUserId: string): Promise<ApiResponse<null>> {
    const url = createApiUrl(API_ENDPOINTS.USERS.SEND_FRIEND_REQUEST, { id: targetUserId });
    return apiClient.post<ApiResponse<null>>(url);
  }

  /**
   * Accept friend request - Updated to match backend
   */
  async acceptFriendRequest(requesterId: string): Promise<ApiResponse<null>> {
    const url = createApiUrl(API_ENDPOINTS.USERS.ACCEPT_FRIEND_REQUEST, { id: requesterId });
    return apiClient.post<ApiResponse<null>>(url);
  }

  /**
   * Get user's friends
   */
  async getFriends(userId: string): Promise<User[]> {
    const url = createApiUrl(API_ENDPOINTS.USERS.FRIENDS, { id: userId });
    return apiClient.get<User[]>(url);
  }

  /**
   * Get mutual friends count - Updated to match backend
   */
  async getMutualFriendsCount(targetUserId: string): Promise<number> {
    const url = createApiUrl(API_ENDPOINTS.USERS.MUTUAL_FRIENDS, { id: targetUserId });
    return apiClient.get<number>(url);
  }

  /**
   * Get user timeline - Updated to match backend
   */
  async getUserTimeline(userId: string, page = 0, size = 10): Promise<any[]> {
    const url = createApiUrl(API_ENDPOINTS.USERS.TIMELINE, { id: userId }, { page, size });
    return apiClient.get<any[]>(url);
  }

  /**
   * Get user activities - Updated to match backend
   */
  async getUserActivities(userId: string, page = 0, size = 10): Promise<any[]> {
    const url = createApiUrl(API_ENDPOINTS.USERS.ACTIVITIES, { id: userId }, { page, size });
    return apiClient.get<any[]>(url);
  }
}

// Export singleton instance
export const userService = new UserService();
