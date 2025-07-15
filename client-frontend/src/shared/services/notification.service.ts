import { apiClient } from '@/shared/config/api-client';
import { ApiResponse } from '@/shared/types';

/**
 * Notification Service
 * Handles push notifications and in-app notifications (cross-domain)
 */
export class NotificationService {
  /**
   * Get user notifications
   */
  async getNotifications(page = 0, size = 20): Promise<any> {
    const url = `/notifications?page=${page}&size=${size}`;
    return apiClient.get(url);
  }

  /**
   * Mark notification as read
   */
  async markAsRead(notificationId: string): Promise<ApiResponse<null>> {
    return apiClient.put<ApiResponse<null>>(`/notifications/${notificationId}/read`);
  }

  /**
   * Mark all notifications as read
   */
  async markAllAsRead(): Promise<ApiResponse<null>> {
    return apiClient.put<ApiResponse<null>>('/notifications/read-all');
  }

  /**
   * Get unread notifications count
   */
  async getUnreadCount(): Promise<number> {
    const response = await apiClient.get<{ count: number }>('/notifications/unread-count');
    return response.count;
  }

  /**
   * Delete notification
   */
  async deleteNotification(notificationId: string): Promise<ApiResponse<null>> {
    return apiClient.delete<ApiResponse<null>>(`/notifications/${notificationId}`);
  }

  /**
   * Update notification preferences
   */
  async updatePreferences(preferences: any): Promise<ApiResponse<null>> {
    return apiClient.put<ApiResponse<null>>('/notifications/preferences', preferences);
  }

  /**
   * Register push token
   */
  async registerPushToken(token: string, platform: 'web' | 'ios' | 'android'): Promise<ApiResponse<null>> {
    return apiClient.post<ApiResponse<null>>('/notifications/push-token', { token, platform });
  }
}

// Export singleton instance
export const notificationService = new NotificationService();
