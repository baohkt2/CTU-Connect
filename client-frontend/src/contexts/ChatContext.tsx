'use client';

import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { webSocketService, TypingEvent, PresenceEvent } from '@/services/websocket.service';
import { chatService } from '@/features/chat/services/chat.service';
import { useAuth } from '@/contexts/AuthContext';
import { ChatMessage, ChatRoom } from '@/shared/types/chat';
import { PaginatedResponse } from '@/shared/types';

// Chat State Interface
export interface ChatState {
  conversations: ChatRoom[];
  activeConversationId: string | null;
  messages: { [conversationId: string]: ChatMessage[] };
  onlineUsers: Set<string>;
  typingUsers: { [conversationId: string]: Set<string> };
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;
}

// Chat Context Interface
export interface ChatContextType extends ChatState {
  // Connection management
  connectToChat: () => Promise<void>;
  disconnectFromChat: () => Promise<void>;

  // Conversation management
  setActiveConversation: (conversationId: string | null) => void;
  loadConversations: () => Promise<void>;

  // Message management
  sendMessage: (conversationId: string, content: string, type?: string) => void;
  loadMessages: (conversationId: string, page?: number) => Promise<void>;
  
  // Reaction management
  addReaction: (messageId: string, emoji: string) => Promise<void>;
  removeReaction: (messageId: string) => Promise<void>;

  // Typing indicators
  startTyping: (conversationId: string) => void;
  stopTyping: (conversationId: string) => void;

  // Utilities
  clearError: () => void;
  getUnreadCount: (conversationId: string) => number;
  markAsRead: (conversationId: string) => void;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export const useChat = () => {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};

interface ChatProviderProps {
  children: ReactNode;
}

export const ChatProvider: React.FC<ChatProviderProps> = ({ children }) => {
  const { user, token } = useAuth();
  const [state, setState] = useState<ChatState>({
    conversations: [],
    activeConversationId: null,
    messages: {},
    onlineUsers: new Set(),
    typingUsers: {},
    isConnected: false,
    isLoading: false,
    error: null,
  });

  // WebSocket event handlers
  const handleMessage = useCallback((message: ChatMessage) => {
    setState(prev => ({
      ...prev,
      messages: {
        ...prev.messages,
        [message.roomId]: [
          ...(prev.messages[message.roomId] || []),
          message
        ].sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime())
      }
    }));
  }, []);

  const handleTyping = useCallback((event: TypingEvent) => {
    setState(prev => {
      const conversationTyping = new Set(prev.typingUsers[event.conversationId] || []);

      if (event.isTyping) {
        conversationTyping.add(event.userId);
      } else {
        conversationTyping.delete(event.userId);
      }

      return {
        ...prev,
        typingUsers: {
          ...prev.typingUsers,
          [event.conversationId]: conversationTyping
        }
      };
    });
  }, []);

  const handlePresence = useCallback((event: PresenceEvent) => {
    setState(prev => {
      const newOnlineUsers = new Set(prev.onlineUsers);

      if (event.status === 'ONLINE') {
        newOnlineUsers.add(event.userId);
      } else {
        newOnlineUsers.delete(event.userId);
      }

      return {
        ...prev,
        onlineUsers: newOnlineUsers
      };
    });
  }, []);

  const handleConnected = useCallback(() => {
    setState(prev => ({ ...prev, isConnected: true, error: null }));
  }, []);

  const handleDisconnected = useCallback(() => {
    setState(prev => ({ ...prev, isConnected: false }));
  }, []);

  const handleError = useCallback((error: Error) => {
    setState(prev => ({ ...prev, error: error.message, isConnected: false }));
  }, []);

  // Connection management
  const connectToChat = useCallback(async () => {
    if (!token) {
      setState(prev => ({ ...prev, error: 'Authentication token not available' }));
      return;
    }

    try {
      setState(prev => ({ ...prev, isLoading: true, error: null }));

      // Setup event listeners
      webSocketService.on('message', handleMessage);
      webSocketService.on('typing', handleTyping);
      webSocketService.on('presence', handlePresence);
      webSocketService.on('connected', handleConnected);
      webSocketService.on('disconnected', handleDisconnected);
      webSocketService.on('error', handleError);

      await webSocketService.connect(token);

      // Load conversations after connection
      await loadConversations();

      setState(prev => ({ ...prev, isLoading: false }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Connection failed',
        isLoading: false
      }));
    }
  }, [token, handleMessage, handleTyping, handlePresence, handleConnected, handleDisconnected, handleError]);

  const disconnectFromChat = useCallback(async () => {
    await webSocketService.disconnect();
    webSocketService.removeAllListeners();

    setState(prev => ({
      ...prev,
      isConnected: false,
      activeConversationId: null,
      typingUsers: {}
    }));
  }, []);

  // Conversation management
  const setActiveConversation = useCallback((conversationId: string | null) => {
    setState(prev => {
      // Leave previous conversation
      if (prev.activeConversationId && prev.activeConversationId !== conversationId) {
        webSocketService.leaveConversation(prev.activeConversationId);
      }

      // Join new conversation
      if (conversationId) {
        webSocketService.joinConversation(conversationId);
        // Load messages for this conversation
        loadMessages(conversationId);
      }

      return { ...prev, activeConversationId: conversationId };
    });
  }, []);

  const loadConversations = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, isLoading: true }));

      const response: PaginatedResponse<ChatRoom> = await chatService.getChatRooms();

      setState(prev => ({
        ...prev,
        conversations: response.content,
        isLoading: false
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to load conversations',
        isLoading: false
      }));
    }
  }, []);

  // Message management
  const sendMessage = useCallback((conversationId: string, content: string, type = 'TEXT') => {
    if (state.isConnected) {
      webSocketService.sendMessage(conversationId, content, type);
    } else {
      setState(prev => ({ ...prev, error: 'Not connected to chat service' }));
    }
  }, [state.isConnected]);

  const loadMessages = useCallback(async (conversationId: string, page = 0) => {
    try {
      const response: PaginatedResponse<ChatMessage> = await chatService.getMessages(conversationId, page, 50);

      setState(prev => ({
        ...prev,
        messages: {
          ...prev.messages,
          [conversationId]: page === 0 ? response.content : [
            ...response.content,
            ...(prev.messages[conversationId] || [])
          ].sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime())
        }
      }));
    } catch (error) {
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to load messages'
      }));
    }
  }, []);

  // Typing indicators
  const startTyping = useCallback((conversationId: string) => {
    if (state.isConnected) {
      webSocketService.sendTyping(conversationId, true);
    }
  }, [state.isConnected]);

  const stopTyping = useCallback((conversationId: string) => {
    if (state.isConnected) {
      webSocketService.sendTyping(conversationId, false);
    }
  }, [state.isConnected]);

  // Reaction management
  const addReaction = useCallback(async (messageId: string, emoji: string) => {
    try {
      // Implement reaction logic here (call API)
      console.log('Adding reaction:', messageId, emoji);
    } catch (error) {
      console.error('Error adding reaction:', error);
    }
  }, []);

  const removeReaction = useCallback(async (messageId: string) => {
    try {
      // Implement remove reaction logic here (call API)
      console.log('Removing reaction:', messageId);
    } catch (error) {
      console.error('Error removing reaction:', error);
    }
  }, []);

  // Utilities
  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  const getUnreadCount = useCallback((conversationId: string) => {
    const messages = state.messages[conversationId] || [];
    return messages.filter(msg =>
      msg.senderId !== user?.id &&
      !msg.readBy?.some(readStatus => readStatus.userId === user?.id)
    ).length;
  }, [state.messages, user?.id]);

  const markAsRead = useCallback(async (conversationId: string) => {
    try {
      await chatService.markRoomAsRead(conversationId);

      setState(prev => ({
        ...prev,
        messages: {
          ...prev.messages,
          [conversationId]: prev.messages[conversationId]?.map(msg => {
            const alreadyRead = msg.readBy?.some(readStatus => readStatus.userId === user?.id);

            if (alreadyRead) return msg;

            const newReadStatus = {
              userId: user?.id || '',
              user: user!,
              readAt: new Date().toISOString()
            };

            return {
              ...msg,
              readBy: [...(msg.readBy || []), newReadStatus],
              isRead: true
            };
          }) || []
        }
      }));
    } catch (error) {
      console.error('Failed to mark messages as read:', error);
    }
  }, [user]);

  const contextValue: ChatContextType = {
    ...state,
    connectToChat,
    disconnectFromChat,
    setActiveConversation,
    loadConversations,
    sendMessage,
    loadMessages,
    addReaction,
    removeReaction,
    startTyping,
    stopTyping,
    clearError,
    getUnreadCount,
    markAsRead,
  };

  return (
    <ChatContext.Provider value={contextValue}>
      {children}
    </ChatContext.Provider>
  );
};
