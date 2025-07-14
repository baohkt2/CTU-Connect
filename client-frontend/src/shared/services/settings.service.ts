import { apiClient } from '@/shared/config/api-client';
import { storage } from '@/shared/utils';
import { ApiResponse } from '@/shared/types';

/**
 * Settings Service
 * Handles user preferences and app settings (cross-domain)
 */
export class SettingsService {
  /**
   * Get user settings
   */
  async getUserSettings(): Promise<any> {
    return apiClient.get('/settings/user');
  }

  /**
   * Update user settings
   */
  async updateUserSettings(settings: Record<string, any>): Promise<ApiResponse<null>> {
    return apiClient.put<ApiResponse<null>>('/settings/user', settings);
  }

  /**
   * Get privacy settings
   */
  async getPrivacySettings(): Promise<any> {
    return apiClient.get('/settings/privacy');
  }

  /**
   * Update privacy settings
   */
  async updatePrivacySettings(settings: Record<string, any>): Promise<ApiResponse<null>> {
    return apiClient.put<ApiResponse<null>>('/settings/privacy', settings);
  }

  /**
   * Get notification settings
   */
  async getNotificationSettings(): Promise<any> {
    return apiClient.get('/settings/notifications');
  }

  /**
   * Update notification settings
   */
  async updateNotificationSettings(settings: Record<string, any>): Promise<ApiResponse<null>> {
    return apiClient.put<ApiResponse<null>>('/settings/notifications', settings);
  }

  /**
   * Export user data
   */
  async exportUserData(): Promise<ApiResponse<{ downloadUrl: string }>> {
    return apiClient.post<ApiResponse<{ downloadUrl: string }>>('/settings/export-data');
  }

  /**
   * Delete account
   */
  async deleteAccount(password: string): Promise<ApiResponse<null>> {
    return apiClient.post<ApiResponse<null>>('/settings/delete-account', { password });
  }

  /**
   * Get app configuration
   */
  async getAppConfig(): Promise<any> {
    return apiClient.get('/settings/app-config');
  }

  // Local settings (stored in localStorage)
  /**
   * Get local theme preference
   */
  getTheme(): 'light' | 'dark' | 'system' {
    return storage.get('theme') || 'system';
  }

  /**
   * Set local theme preference
   */
  setTheme(theme: 'light' | 'dark' | 'system'): void {
    storage.set('theme', theme);
    this.applyTheme(theme);
  }

  /**
   * Apply theme to document
   */
  private applyTheme(theme: 'light' | 'dark' | 'system'): void {
    if (typeof window === 'undefined') return;

    const root = document.documentElement;

    if (theme === 'system') {
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      root.classList.toggle('dark', prefersDark);
    } else {
      root.classList.toggle('dark', theme === 'dark');
    }
  }

  /**
   * Get local language preference
   */
  getLanguage(): string {
    return storage.get('language') || 'vi';
  }

  /**
   * Set local language preference
   */
  setLanguage(language: string): void {
    storage.set('language', language);
  }
}

// Export singleton instance
export const settingsService = new SettingsService();
