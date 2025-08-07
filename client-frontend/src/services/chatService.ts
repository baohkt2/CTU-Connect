import api from '@/lib/api';
import { PaginatedResponse } from '@/types';

// Types matching backend DTOs
export interface CreateConversationRequest {
  participantIds: string[];
  type: 'DIRECT' | 'GROUP';
  name?: string;
}

export interface SendMessageRequest {
  conversationId: string;
  content: string;
  type: 'TEXT' | 'IMAGE' | 'FILE';
  replyToMessageId?: string;
}

export interface UpdateConversationRequest {
  name?: string;
  avatarUrl?: string;
}

export interface AddReactionRequest {
  messageId: string;
  emoji: string;
}

// Response interfaces matching backend
export interface ConversationResponse {
  id: string;
  name: string;
  type: 'DIRECT' | 'GROUP';
  participants: ParticipantInfo[];
  lastMessage?: MessageResponse;
  lastMessageAt?: string;
  unreadCount: number;
  avatarUrl?: string;
  createdAt: string;
  updatedAt: string;
}

export interface MessageResponse {
  id: string;
  conversationId: string;
  senderId: string;
  senderName: string;
  senderAvatar: string;
  type: 'TEXT' | 'IMAGE' | 'FILE' | 'SYSTEM';
  content: string;
  attachment?: MessageAttachment;
  replyToMessageId?: string;
  replyToMessage?: MessageResponse;
  reactions: MessageReaction[];
  status: 'SENT' | 'DELIVERED' | 'READ';
  readByUserIds: string[];
  createdAt: string;
  updatedAt: string;
  editedAt?: string;
}

export interface ParticipantInfo {
  userId: string;
  userName: string;
  userAvatar: string;
  presenceStatus: 'ONLINE' | 'OFFLINE' | 'AWAY';
  lastSeenAt?: string;
  isAdmin: boolean;
}

export interface MessageAttachment {
  id: string;
  type: 'IMAGE' | 'FILE';
  name: string;
  url: string;
  size: number;
  mimeType: string;
}

export interface MessageReaction {
  id: string;
  messageId: string;
  userId: string;
  userName: string;
  emoji: string;
  createdAt: string;
}

export interface ChatPageResponse<T> {
  content: T[];
  totalElements: number;
  totalPages: number;
  number: number;
  size: number;
  first: boolean;
  last: boolean;
}

export interface UserPresenceResponse {
  userId: string;
  userName: string;
  status: 'ONLINE' | 'OFFLINE' | 'AWAY';
  lastSeenAt?: string;
  connectedAt?: string;
}

export const chatService = {
  // Conversation operations
  async createConversation(request: CreateConversationRequest): Promise<ConversationResponse> {
    const response = await api.post('/chat/api/conversations', request);
    return response.data;
  },

  async getUserConversations(page = 0, size = 20): Promise<PaginatedResponse<ConversationResponse>> {
    const response = await api.get(`/chat/api/conversations?page=${page}&size=${size}`);
    return response.data;
  },

  async getConversation(conversationId: string): Promise<ConversationResponse> {
    const response = await api.get(`/chat/api/conversations/${conversationId}`);
    return response.data;
  },

  async updateConversation(conversationId: string, request: UpdateConversationRequest): Promise<ConversationResponse> {
    const response = await api.put(`/chat/api/conversations/${conversationId}`, request);
    return response.data;
  },

  async addParticipant(conversationId: string, participantId: string): Promise<void> {
    await api.post(`/chat/api/conversations/${conversationId}/participants?participantId=${participantId}`);
  },

  async removeParticipant(conversationId: string, participantId: string): Promise<void> {
    await api.delete(`/chat/api/conversations/${conversationId}/participants/${participantId}`);
  },

  async searchConversations(query: string): Promise<ConversationResponse[]> {
    const response = await api.get(`/chat/api/conversations/search?query=${encodeURIComponent(query)}`);
    return response.data;
  },

  async deleteConversation(conversationId: string): Promise<void> {
    await api.delete(`/chat/api/conversations/${conversationId}`);
  },

  // Message operations
  async sendMessage(request: SendMessageRequest): Promise<MessageResponse> {
    const response = await api.post('/chat/api/messages', request);
    return response.data;
  },

  async getMessages(conversationId: string, page = 0, size = 50): Promise<ChatPageResponse<MessageResponse>> {
    const response = await api.get(`/chat/api/messages/conversation/${conversationId}?page=${page}&size=${size}`);
    return response.data;
  },

  async editMessage(messageId: string, content: string): Promise<MessageResponse> {
    const response = await api.put(`/chat/api/messages/${messageId}?content=${encodeURIComponent(content)}`);
    return response.data;
  },

  async deleteMessage(messageId: string): Promise<void> {
    await api.delete(`/chat/api/messages/${messageId}`);
  },

  async addReaction(request: AddReactionRequest): Promise<void> {
    await api.post('/chat/api/messages/reactions', request);
  },

  async removeReaction(messageId: string, emoji: string): Promise<void> {
    await api.delete(`/chat/api/messages/${messageId}/reactions?emoji=${encodeURIComponent(emoji)}`);
  },

  async markAsRead(conversationId: string): Promise<void> {
    await api.post(`/chat/api/messages/mark-read/${conversationId}`);
  },

  async getUnreadCount(): Promise<number> {
    const response = await api.get('/chat/api/messages/unread-count');
    return response.data;
  },

  // User presence operations
  async getUserPresence(userId: string): Promise<UserPresenceResponse> {
    const response = await api.get(`/chat/api/presence/user/${userId}`);
    return response.data;
  },

  async getOnlineUsers(): Promise<UserPresenceResponse[]> {
    const response = await api.get('/chat/api/presence/online');
    return response.data;
  },

  async updatePresenceStatus(status: 'ONLINE' | 'OFFLINE' | 'AWAY'): Promise<void> {
    await api.put(`/chat/api/presence/status?status=${status}`);
  },

  // File upload for chat attachments
  async uploadChatFile(file: File): Promise<{ url: string; fileName: string; fileSize: number; mimeType: string }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/chat/api/files/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  },

  // Direct conversation creation
  async createDirectConversation(participantId: string): Promise<ConversationResponse> {
    return this.createConversation({
      participantIds: [participantId],
      type: 'DIRECT'
    });
  },

  // Group conversation creation
  async createGroupConversation(participantIds: string[], name: string): Promise<ConversationResponse> {
    return this.createConversation({
      participantIds,
      type: 'GROUP',
      name
    });
  }
};
