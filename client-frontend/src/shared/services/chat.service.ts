import { apiClient } from '@/shared/config/api-client';
import { API_ENDPOINTS } from '@/shared/constants';
import { createApiUrl } from '@/shared/utils';
import {
  ChatRoom,
  ChatMessage,
  CreateChatRoomRequest,
  SendMessageRequest,
  UpdateMessageRequest,
  PaginatedResponse,
  ApiResponse,
  OnlineStatus,
} from '@/shared/types';

/**
 * Chat Service
 * Handles all chat-related API calls
 */
export class ChatService {
  /**
   * Get chat rooms
   */
  async getChatRooms(page = 0, size = 20): Promise<PaginatedResponse<ChatRoom>> {
    const url = createApiUrl(API_ENDPOINTS.CHAT.ROOMS, undefined, { page, size });
    return apiClient.get<PaginatedResponse<ChatRoom>>(url);
  }

  /**
   * Get single chat room
   */
  async getChatRoom(roomId: string): Promise<ChatRoom> {
    const url = createApiUrl(API_ENDPOINTS.CHAT.ROOMS + '/:id', { id: roomId });
    return apiClient.get<ChatRoom>(url);
  }

  /**
   * Create new chat room
   */
  async createChatRoom(roomData: CreateChatRoomRequest): Promise<ChatRoom> {
    return apiClient.post<ChatRoom>(API_ENDPOINTS.CHAT.ROOMS, roomData);
  }

  /**
   * Get messages for a chat room
   */
  async getMessages(
    roomId: string,
    page = 0,
    size = 50
  ): Promise<PaginatedResponse<ChatMessage>> {
    const url = createApiUrl(
      API_ENDPOINTS.CHAT.MESSAGES,
      { roomId },
      { page, size }
    );
    return apiClient.get<PaginatedResponse<ChatMessage>>(url);
  }

  /**
   * Send message
   */
  async sendMessage(messageData: SendMessageRequest): Promise<ChatMessage> {
    const { roomId, ...data } = messageData;
    const url = createApiUrl(API_ENDPOINTS.CHAT.MESSAGES, { roomId });

    // Handle file attachments
    if (data.attachments && data.attachments.length > 0) {
      const formData = new FormData();
      formData.append('content', data.content);
      formData.append('type', data.type);

      if (data.replyTo) {
        formData.append('replyTo', data.replyTo);
      }

      data.attachments.forEach(file => {
        formData.append('attachments', file);
      });

      return apiClient.post<ChatMessage>(url, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
    }

    return apiClient.post<ChatMessage>(url, data);
  }

  /**
   * Update message
   */
  async updateMessage(
    messageId: string,
    updateData: UpdateMessageRequest
  ): Promise<ChatMessage> {
    const url = createApiUrl('/chat/messages/:id', { id: messageId });
    return apiClient.put<ChatMessage>(url, updateData);
  }

  /**
   * Delete message
   */
  async deleteMessage(messageId: string): Promise<ApiResponse<null>> {
    const url = createApiUrl('/chat/messages/:id', { id: messageId });
    return apiClient.delete<ApiResponse<null>>(url);
  }

  /**
   * Mark message as read
   */
  async markAsRead(messageId: string): Promise<ApiResponse<null>> {
    const url = createApiUrl(API_ENDPOINTS.CHAT.MARK_READ, { id: messageId });
    return apiClient.put<ApiResponse<null>>(url);
  }

  /**
   * Mark all messages in room as read
   */
  async markRoomAsRead(roomId: string): Promise<ApiResponse<null>> {
    const url = createApiUrl(API_ENDPOINTS.CHAT.ROOM_READ, { id: roomId });
    return apiClient.put<ApiResponse<null>>(url);
  }

  /**
   * Get unread messages count
   */
  async getUnreadCount(): Promise<number> {
    const response = await apiClient.get<{ count: number }>(
      API_ENDPOINTS.CHAT.UNREAD_COUNT
    );
    return response.count;
  }

  /**
   * Get online users
   */
  async getOnlineUsers(): Promise<OnlineStatus[]> {
    return apiClient.get<OnlineStatus[]>('/chat/online-users');
  }

  /**
   * Start typing indicator
   */
  async startTyping(roomId: string): Promise<void> {
    const url = createApiUrl('/chat/rooms/:roomId/typing/start', { roomId });
    return apiClient.post(url);
  }

  /**
   * Stop typing indicator
   */
  async stopTyping(roomId: string): Promise<void> {
    const url = createApiUrl('/chat/rooms/:roomId/typing/stop', { roomId });
    return apiClient.post(url);
  }

  /**
   * Leave chat room
   */
  async leaveRoom(roomId: string): Promise<ApiResponse<null>> {
    const url = createApiUrl('/chat/rooms/:roomId/leave', { roomId });
    return apiClient.post<ApiResponse<null>>(url);
  }

  /**
   * Add users to chat room
   */
  async addUsersToRoom(
    roomId: string,
    userIds: string[]
  ): Promise<ApiResponse<null>> {
    const url = createApiUrl('/chat/rooms/:roomId/users', { roomId });
    return apiClient.post<ApiResponse<null>>(url, { userIds });
  }

  /**
   * Remove user from chat room
   */
  async removeUserFromRoom(
    roomId: string,
    userId: string
  ): Promise<ApiResponse<null>> {
    const url = createApiUrl('/chat/rooms/:roomId/users/:userId', { roomId, userId });
    return apiClient.delete<ApiResponse<null>>(url);
  }
}

// Export singleton instance
export const chatService = new ChatService();
