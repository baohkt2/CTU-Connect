import { apiClient } from '@/shared/config/api-client';

/**
 * Analytics Service
 * Handles user analytics and tracking (cross-domain)
 */
export class AnalyticsService {
  /**
   * Track user event
   * @param eventName Tên sự kiện (e.g., 'login', 'click_button')
   * @param properties Thuộc tính bổ sung của sự kiện
   */
  async trackEvent(eventName: string, properties?: Record<string, unknown>): Promise<void> {
    try {
      await apiClient.post('/analytics/event', {
        eventName,
        properties,
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      console.error(`Failed to track event ${eventName}:`, error);
      throw error;
    }
  }

  /**
   * Track page view
   * @param pageName Tên trang (e.g., 'login_page', 'dashboard')
   * @param properties Thuộc tính bổ sung của page view
   */
  async trackPageView(pageName: string, properties?: Record<string, unknown>): Promise<void> {
    try {
      await apiClient.post('/analytics/page-view', {
        pageName,
        properties,
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      console.error(`Failed to track page view ${pageName}:`, error);
      throw error;
    }
  }

  /**
   * Track user action
   * @param action Hành động (e.g., 'submit_form', 'download_file')
   * @param target Mục tiêu của hành động (e.g., 'login_form', 'export_button')
   * @param properties Thuộc tính bổ sung
   */
  async trackUserAction(action: string, target: string, properties?: Record<string, unknown>): Promise<void> {
    try {
      await apiClient.post('/analytics/user-action', {
        action,
        target,
        properties,
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      console.error(`Failed to track user action ${action} on ${target}:`, error);
      throw error;
    }
  }

  /**
   * Get user analytics data
   * @param timeRange Khoảng thời gian (default: '30d')
   * @returns Dữ liệu phân tích của người dùng
   */
  async getUserAnalytics(timeRange: '7d' | '30d' | '90d' = '30d'): Promise<unknown> {
    try {
      const response = await apiClient.get(`/analytics/user?timeRange=${timeRange}`);
      return response.data;
    } catch (error) {
      console.error(`Failed to fetch user analytics for ${timeRange}:`, error);
      throw error;
    }
  }

  /**
   * Get app analytics data
   * @param timeRange Khoảng thời gian (default: '30d')
   * @returns Dữ liệu phân tích của ứng dụng
   */
  async getAppAnalytics(timeRange: '7d' | '30d' | '90d' = '30d'): Promise<unknown> {
    try {
      const response = await apiClient.get(`/analytics/app?timeRange=${timeRange}`);
      return response.data;
    } catch (error) {
      console.error(`Failed to fetch app analytics for ${timeRange}:`, error);
      throw error;
    }
  }
}

// Export singleton instance
export const analyticsService = new AnalyticsService();