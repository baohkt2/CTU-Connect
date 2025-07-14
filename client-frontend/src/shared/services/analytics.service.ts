import { apiClient } from '@/shared/config/api-client';


/**
 * Analytics Service
 * Handles user analytics and tracking (cross-domain)
 */
export class AnalyticsService {
  /**
   * Track user event
   */
  async trackEvent(eventName: string, properties?: Record<string, unknown>): Promise<void> { ... }

  async trackPageView(pageName: string, properties?: Record<string, unknown>): Promise<void> { ... }

  async trackUserAction(action: string, target: string, properties?: Record<string, unknown>): Promise<void> { ... }

  async getUserAnalytics(timeRange: '7d' | '30d' | '90d' = '30d'): Promise<unknown> {
    return apiClient.get(`/analytics/user?timeRange=${timeRange}`);
  }

  async getAppAnalytics(timeRange: '7d' | '30d' | '90d' = '30d'): Promise<unknown> {
    return apiClient.get(`/analytics/app?timeRange=${timeRange}`);
  }

}

// Export singleton instance
export const analyticsService = new AnalyticsService();
