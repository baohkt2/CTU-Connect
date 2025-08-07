import SockJS from 'sockjs-client';
import { Client, IMessage, StompSubscription } from '@stomp/stompjs';
import { EventEmitter } from 'events';
import { ChatMessage } from '@/shared/types/chat';

export interface TypingEvent {
  userId: string;
  conversationId: string;
  isTyping: boolean;
}

export interface PresenceEvent {
  userId: string;
  status: 'ONLINE' | 'OFFLINE' | 'AWAY';
}

export class WebSocketService extends EventEmitter {
  private client: Client | null = null;
  private subscriptions: Map<string, StompSubscription> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 5000;
  private isConnected = false;

  constructor(private gatewayUrl: string = 'http://localhost:8090') {
    super();
  }

  /**
   * Connect to WebSocket server through API Gateway with JWT authentication
   */
  async connect(token: string): Promise<void> {
    if (this.isConnected) {
      console.warn('WebSocket already connected');
      return;
    }

    try {
      // Connect through API Gateway WebSocket endpoint
      const socket = new SockJS(`${this.gatewayUrl}/ws/chat`);

      this.client = new Client({
        webSocketFactory: () => socket,
        connectHeaders: {
          'Authorization': `Bearer ${token}`,
          // These headers will be processed by API Gateway's JWT filter
          // and converted to X-User-Id, X-Username headers for downstream services
        },
        debug: (str) => console.debug('STOMP Debug:', str),
        reconnectDelay: this.reconnectInterval,
        heartbeatIncoming: 4000,
        heartbeatOutgoing: 4000,
      });

      // Setup connection handlers
      this.client.onConnect = (frame) => {
        console.log('WebSocket connected through API Gateway:', frame);
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.emit('connected');
        this.setupGlobalSubscriptions();
      };

      this.client.onDisconnect = (frame) => {
        console.log('WebSocket disconnected:', frame);
        this.isConnected = false;
        this.emit('disconnected');
        this.cleanup();
      };

      this.client.onStompError = (frame) => {
        console.error('WebSocket STOMP error:', frame);
        this.emit('error', new Error(frame.body));
      };

      this.client.onWebSocketError = (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
        this.handleReconnect();
      };

      // Activate the client
      this.client.activate();
    } catch (error) {
      console.error('Failed to connect to WebSocket through API Gateway:', error);
      throw error;
    }
  }

  /**
   * Disconnect from WebSocket server
   */
  async disconnect(): Promise<void> {
    if (this.client) {
      this.client.deactivate();
      this.cleanup();
    }
  }

  /**
   * Join a conversation to receive real-time messages
   */
  joinConversation(conversationId: string): void {
    if (!this.client || !this.isConnected) {
      throw new Error('WebSocket not connected');
    }

    const destination = `/topic/conversations/${conversationId}`;

    if (this.subscriptions.has(conversationId)) {
      console.warn(`Already subscribed to conversation ${conversationId}`);
      return;
    }

    const subscription = this.client.subscribe(destination, (message: IMessage) => {
      try {
        const chatMessage: ChatMessage = JSON.parse(message.body);
        this.emit('message', chatMessage);
      } catch (error) {
        console.error('Error parsing message:', error);
      }
    });

    this.subscriptions.set(conversationId, subscription);
    console.log(`Joined conversation: ${conversationId}`);
  }

  /**
   * Leave a conversation
   */
  leaveConversation(conversationId: string): void {
    const subscription = this.subscriptions.get(conversationId);
    if (subscription) {
      subscription.unsubscribe();
      this.subscriptions.delete(conversationId);
      console.log(`Left conversation: ${conversationId}`);
    }
  }

  /**
   * Send a chat message
   */
  sendMessage(conversationId: string, content: string, type: string = 'TEXT', replyTo?: string): void {
    if (!this.client || !this.isConnected) {
      throw new Error('WebSocket not connected');
    }

    const messagePayload = {
      conversationId,
      content,
      type,
      replyTo,
      timestamp: new Date().toISOString()
    };

    this.client.publish({
      destination: '/app/chat.sendMessage',
      body: JSON.stringify(messagePayload)
    });
  }

  /**
   * Send typing indicator
   */
  sendTyping(conversationId: string, isTyping: boolean): void {
    if (!this.client || !this.isConnected) {
      return;
    }

    const typingPayload = {
      conversationId,
      isTyping,
      timestamp: new Date().toISOString()
    };

    this.client.publish({
      destination: '/app/chat.typing',
      body: JSON.stringify(typingPayload)
    });
  }

  /**
   * Update user presence status
   */
  updatePresence(status: 'ONLINE' | 'AWAY' | 'OFFLINE'): void {
    if (!this.client || !this.isConnected) {
      return;
    }

    const presencePayload = {
      status,
      timestamp: new Date().toISOString()
    };

    this.client.publish({
      destination: '/app/user.presence',
      body: JSON.stringify(presencePayload)
    });
  }

  /**
   * Get connection status
   */
  isWebSocketConnected(): boolean {
    return this.isConnected && this.client?.connected === true;
  }

  /**
   * Setup global subscriptions for user-specific events
   */
  private setupGlobalSubscriptions(): void {
    if (!this.client || !this.isConnected) {
      return;
    }

    // Subscribe to personal notifications
    this.client.subscribe('/user/queue/notifications', (message: IMessage) => {
      try {
        const notification = JSON.parse(message.body);
        this.emit('notification', notification);
      } catch (error) {
        console.error('Error parsing notification:', error);
      }
    });

    // Subscribe to typing indicators
    this.client.subscribe('/user/queue/typing', (message: IMessage) => {
      try {
        const typingEvent: TypingEvent = JSON.parse(message.body);
        this.emit('typing', typingEvent);
      } catch (error) {
        console.error('Error parsing typing event:', error);
      }
    });

    // Subscribe to presence updates
    this.client.subscribe('/topic/presence', (message: IMessage) => {
      try {
        const presenceEvent: PresenceEvent = JSON.parse(message.body);
        this.emit('presence', presenceEvent);
      } catch (error) {
        console.error('Error parsing presence event:', error);
      }
    });
  }

  /**
   * Handle reconnection logic
   */
  private handleReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

      setTimeout(() => {
        if (!this.isConnected && this.client) {
          this.client.activate();
        }
      }, this.reconnectInterval);
    } else {
      console.error('Max reconnection attempts reached');
      this.emit('maxReconnectAttemptsReached');
    }
  }

  /**
   * Cleanup resources
   */
  private cleanup(): void {
    this.subscriptions.forEach(subscription => subscription.unsubscribe());
    this.subscriptions.clear();
    this.isConnected = false;
    this.client = null;
  }
}

// Export singleton instance
export const webSocketService = new WebSocketService();
