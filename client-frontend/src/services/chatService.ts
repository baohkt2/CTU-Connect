import axios, { AxiosResponse } from 'axios';
import { io, Socket } from 'socket.io-client';

// Types
export interface User {
  id: string;
  name: string;
  avatar: string;
  fullName: string;
  role: string;
}

export interface Conversation {
  id: string;
  name: string;
  type: 'DIRECT' | 'GROUP';
  participants: ParticipantInfo[];
  lastMessage?: Message;
  lastMessageAt?: string;
  unreadCount: number;
  avatarUrl?: string;
  createdAt: string;
  updatedAt: string;
}

export interface ParticipantInfo {
  userId: string;
  userName: string;
  userAvatar: string;
  presenceStatus: 'ONLINE' | 'OFFLINE' | 'AWAY';
  lastSeenAt?: string;
  isAdmin: boolean;
}

export interface Message {
  id: string;
  conversationId: string;
  senderId: string;
  senderName: string;
  senderAvatar: string;
  type: 'TEXT' | 'IMAGE' | 'FILE' | 'SYSTEM';
  content: string;
  attachment?: MessageAttachment;
  replyToMessageId?: string;
  replyToMessage?: Message;
  reactions: MessageReaction[];
  status: 'SENT' | 'DELIVERED' | 'READ';
  readByUserIds: string[];
  createdAt: string;
  updatedAt: string;
  editedAt?: string;
  isEdited: boolean;
  isDeleted: boolean;
}

export interface MessageAttachment {
  fileName: string;
  fileUrl: string;
  fileType: string;
  fileSize: number;
  thumbnailUrl?: string;
}

export interface MessageReaction {
  userId: string;
  userName: string;
  emoji: string;
  createdAt: string;
}

export interface UserPresence {
  userId: string;
  userName: string;
  userAvatar: string;
  status: 'ONLINE' | 'OFFLINE' | 'AWAY';
  currentActivity?: string;
  lastSeenAt: string;
}

export interface CreateConversationRequest {
  name?: string;
  participantIds: string[];
  type: 'DIRECT' | 'GROUP';
  description?: string;
  avatarUrl?: string;
}

export interface SendMessageRequest {
  conversationId: string;
  content: string;
  replyToMessageId?: string;
}

export interface AddReactionRequest {
  messageId: string;
  emoji: string;
}

// API Base URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8090';
const CHAT_API_URL = `${API_BASE_URL}/chat-service/api`;
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:8086';

class ChatService {
  private socket: Socket | null = null;
  private token: string | null = null;
  private currentUserId: string | null = null;

  constructor() {
    // Get token from localStorage or cookie
    if (typeof window !== 'undefined') {
      this.token = localStorage.getItem('token') || sessionStorage.getItem('token');
    }
  }

  // Authentication
  setToken(token: string) {
    this.token = token;
    if (typeof window !== 'undefined') {
      localStorage.setItem('token', token);
    }
  }

  setCurrentUserId(userId: string) {
    this.currentUserId = userId;
  }

  private getAuthHeaders() {
    return {
      Authorization: `Bearer ${this.token}`,
      'Content-Type': 'application/json',
    };
  }

  // WebSocket Connection
  connectWebSocket(userId: string): Promise<Socket> {
    return new Promise((resolve, reject) => {
      try {
        this.socket = io(`${WS_URL}/ws/chat`, {
          transports: ['websocket', 'polling'],
          extraHeaders: {
            userId: userId,
            Authorization: `Bearer ${this.token}`,
          },
        });

        this.socket.on('connect', () => {
          console.log('Connected to chat WebSocket');
          resolve(this.socket!);
        });

        this.socket.on('connect_error', (error) => {
          console.error('WebSocket connection error:', error);
          reject(error);
        });

        this.socket.on('disconnect', () => {
          console.log('Disconnected from chat WebSocket');
        });

      } catch (error) {
        reject(error);
      }
    });
  }

  disconnectWebSocket() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  // WebSocket Event Listeners
  onNewMessage(callback: (message: Message) => void) {
    if (this.socket) {
      this.socket.on('message', callback);
    }
  }

  onMessageUpdate(callback: (message: Message) => void) {
    if (this.socket) {
      this.socket.on('message.update', callback);
    }
  }

  onMessageDelete(callback: (messageId: string) => void) {
    if (this.socket) {
      this.socket.on('message.delete', callback);
    }
  }

  onTyping(callback: (data: { userId: string; isTyping: boolean }) => void) {
    if (this.socket) {
      this.socket.on('typing', callback);
    }
  }

  onPresenceUpdate(callback: (presence: UserPresence) => void) {
    if (this.socket) {
      this.socket.on('presence.update', callback);
    }
  }

  onReactionUpdate(callback: (message: Message) => void) {
    if (this.socket) {
      this.socket.on('reaction.update', callback);
    }
  }

  // Send typing status
  sendTypingStatus(conversationId: string, isTyping: boolean) {
    if (this.socket) {
      this.socket.emit('typing', { conversationId, isTyping });
    }
  }

  // Conversation APIs
  async getConversations(page = 0, size = 20): Promise<{ content: Conversation[]; totalPages: number; totalElements: number }> {
    const response: AxiosResponse = await axios.get(
      `${CHAT_API_URL}/conversations?page=${page}&size=${size}`,
      { headers: this.getAuthHeaders() }
    );
    return response.data;
  }

  async getConversation(conversationId: string): Promise<Conversation> {
    const response: AxiosResponse = await axios.get(
      `${CHAT_API_URL}/conversations/${conversationId}`,
      { headers: this.getAuthHeaders() }
    );
    return response.data;
  }

  async createConversation(request: CreateConversationRequest): Promise<Conversation> {
    const response: AxiosResponse = await axios.post(
      `${CHAT_API_URL}/conversations`,
      request,
      { headers: this.getAuthHeaders() }
    );
    return response.data;
  }

  async searchConversations(query: string): Promise<Conversation[]> {
    const response: AxiosResponse = await axios.get(
      `${CHAT_API_URL}/conversations/search?query=${encodeURIComponent(query)}`,
      { headers: this.getAuthHeaders() }
    );
    return response.data;
  }

  async addParticipant(conversationId: string, participantId: string): Promise<void> {
    await axios.post(
      `${CHAT_API_URL}/conversations/${conversationId}/participants?participantId=${participantId}`,
      {},
      { headers: this.getAuthHeaders() }
    );
  }

  async removeParticipant(conversationId: string, participantId: string): Promise<void> {
    await axios.delete(
      `${CHAT_API_URL}/conversations/${conversationId}/participants/${participantId}`,
      { headers: this.getAuthHeaders() }
    );
  }

  // Message APIs
  async getMessages(conversationId: string, page = 0, size = 50): Promise<{ content: Message[]; hasNext: boolean; hasPrevious: boolean }> {
    const response: AxiosResponse = await axios.get(
      `${CHAT_API_URL}/messages/conversation/${conversationId}?page=${page}&size=${size}`,
      { headers: this.getAuthHeaders() }
    );
    return response.data;
  }

  async sendMessage(request: SendMessageRequest): Promise<Message> {
    const response: AxiosResponse = await axios.post(
      `${CHAT_API_URL}/messages`,
      request,
      { headers: this.getAuthHeaders() }
    );
    return response.data;
  }

  async editMessage(messageId: string, content: string): Promise<Message> {
    const response: AxiosResponse = await axios.put(
      `${CHAT_API_URL}/messages/${messageId}?content=${encodeURIComponent(content)}`,
      {},
      { headers: this.getAuthHeaders() }
    );
    return response.data;
  }

  async deleteMessage(messageId: string): Promise<void> {
    await axios.delete(
      `${CHAT_API_URL}/messages/${messageId}`,
      { headers: this.getAuthHeaders() }
    );
  }

  async addReaction(request: AddReactionRequest): Promise<Message> {
    const response: AxiosResponse = await axios.post(
      `${CHAT_API_URL}/messages/reactions`,
      request,
      { headers: this.getAuthHeaders() }
    );
    return response.data;
  }

  async removeReaction(messageId: string): Promise<void> {
    await axios.delete(
      `${CHAT_API_URL}/messages/${messageId}/reactions`,
      { headers: this.getAuthHeaders() }
    );
  }

  async searchMessages(conversationId: string, query: string): Promise<Message[]> {
    const response: AxiosResponse = await axios.get(
      `${CHAT_API_URL}/messages/conversation/${conversationId}/search?query=${encodeURIComponent(query)}`,
      { headers: this.getAuthHeaders() }
    );
    return response.data;
  }

  async markAsRead(conversationId: string): Promise<void> {
    await axios.post(
      `${CHAT_API_URL}/messages/conversation/${conversationId}/mark-read`,
      {},
      { headers: this.getAuthHeaders() }
    );
  }

  async getUnreadCount(conversationId: string): Promise<number> {
    const response: AxiosResponse = await axios.get(
      `${CHAT_API_URL}/messages/conversation/${conversationId}/unread-count`,
      { headers: this.getAuthHeaders() }
    );
    return response.data;
  }

  // Presence APIs
  async getUserPresence(userId: string): Promise<UserPresence> {
    const response: AxiosResponse = await axios.get(
      `${CHAT_API_URL}/presence/${userId}`,
      { headers: this.getAuthHeaders() }
    );
    return response.data;
  }

  async getMultipleUserPresence(userIds: string[]): Promise<UserPresence[]> {
    const response: AxiosResponse = await axios.get(
      `${CHAT_API_URL}/presence/users?userIds=${userIds.join(',')}`,
      { headers: this.getAuthHeaders() }
    );
    return response.data;
  }

  async getOnlineUsers(): Promise<UserPresence[]> {
    const response: AxiosResponse = await axios.get(
      `${CHAT_API_URL}/presence/online`,
      { headers: this.getAuthHeaders() }
    );
    return response.data;
  }

  async setAway(): Promise<void> {
    await axios.post(
      `${CHAT_API_URL}/presence/away`,
      {},
      { headers: this.getAuthHeaders() }
    );
  }
}

// Export singleton instance
export const chatService = new ChatService();
