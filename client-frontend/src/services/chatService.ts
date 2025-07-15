import api from '@/lib/api';
import { ChatMessage, ChatRoom, ApiResponse, PaginatedResponse } from '@/types';

export const chatService = {
  async getChatRooms(page = 0, size = 10): Promise<PaginatedResponse<ChatRoom>> {
    const response = await api.get(`/chat/rooms?page=${page}&size=${size}`);
    return response.data;
  },

  async getChatRoom(roomId: string): Promise<ChatRoom> {
    const response = await api.get(`/chat/rooms/${roomId}`);
    return response.data;
  },

  async createChatRoom(participantId: string): Promise<ChatRoom> {
    const response = await api.post('/chat/rooms', { participantId });
    return response.data;
  },

  async getMessages(roomId: string, page = 0, size = 20): Promise<PaginatedResponse<ChatMessage>> {
    const response = await api.get(`/chat/rooms/${roomId}/messages?page=${page}&size=${size}`);
    return response.data;
  },

  async sendMessage(roomId: string, content: string): Promise<ChatMessage> {
    const response = await api.post(`/chat/rooms/${roomId}/messages`, { content });
    return response.data;
  },

  async markAsRead(messageId: string): Promise<ApiResponse<null>> {
    const response = await api.put(`/chat/messages/${messageId}/read`);
    return response.data;
  },

  async markRoomAsRead(roomId: string): Promise<ApiResponse<null>> {
    const response = await api.put(`/chat/rooms/${roomId}/read`);
    return response.data;
  },

  async deleteMessage(messageId: string): Promise<ApiResponse<null>> {
    const response = await api.delete(`/chat/messages/${messageId}`);
    return response.data;
  },

  async getUnreadCount(): Promise<number> {
    const response = await api.get('/chat/unread-count');
    return response.data;
  }
};
