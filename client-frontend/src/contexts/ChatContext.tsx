import React, { createContext, useContext, useReducer, useEffect, ReactNode } from 'react';
import { chatService, Conversation, Message, UserPresence } from '../services/chatService';

// Chat State
interface ChatState {
  conversations: Conversation[];
  activeConversationId: string | null;
  messages: Record<string, Message[]>;
  onlineUsers: UserPresence[];
  typingUsers: Record<string, string[]>; // conversationId -> userIds
  unreadCounts: Record<string, number>;
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;
}

// Actions
type ChatAction =
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_CONNECTED'; payload: boolean }
  | { type: 'SET_CONVERSATIONS'; payload: Conversation[] }
  | { type: 'ADD_CONVERSATION'; payload: Conversation }
  | { type: 'UPDATE_CONVERSATION'; payload: Conversation }
  | { type: 'SET_ACTIVE_CONVERSATION'; payload: string | null }
  | { type: 'SET_MESSAGES'; payload: { conversationId: string; messages: Message[] } }
  | { type: 'ADD_MESSAGE'; payload: Message }
  | { type: 'UPDATE_MESSAGE'; payload: Message }
  | { type: 'DELETE_MESSAGE'; payload: { conversationId: string; messageId: string } }
  | { type: 'SET_ONLINE_USERS'; payload: UserPresence[] }
  | { type: 'UPDATE_USER_PRESENCE'; payload: UserPresence }
  | { type: 'SET_TYPING_USERS'; payload: { conversationId: string; userIds: string[] } }
  | { type: 'SET_UNREAD_COUNT'; payload: { conversationId: string; count: number } };

// Initial State
const initialState: ChatState = {
  conversations: [],
  activeConversationId: null,
  messages: {},
  onlineUsers: [],
  typingUsers: {},
  unreadCounts: {},
  isConnected: false,
  isLoading: false,
  error: null,
};

// Reducer
function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    
    case 'SET_CONNECTED':
      return { ...state, isConnected: action.payload };
    
    case 'SET_CONVERSATIONS':
      return { ...state, conversations: action.payload };
    
    case 'ADD_CONVERSATION':
      return {
        ...state,
        conversations: [action.payload, ...state.conversations],
      };
    
    case 'UPDATE_CONVERSATION':
      return {
        ...state,
        conversations: state.conversations.map(conv =>
          conv.id === action.payload.id ? action.payload : conv
        ),
      };
    
    case 'SET_ACTIVE_CONVERSATION':
      return { ...state, activeConversationId: action.payload };
    
    case 'SET_MESSAGES':
      return {
        ...state,
        messages: {
          ...state.messages,
          [action.payload.conversationId]: action.payload.messages,
        },
      };
    
    case 'ADD_MESSAGE':
      const conversationId = action.payload.conversationId;
      const currentMessages = state.messages[conversationId] || [];
      
      // Check if message already exists (prevent duplicates)
      const messageExists = currentMessages.some(msg => msg.id === action.payload.id);
      if (messageExists) return state;
      
      return {
        ...state,
        messages: {
          ...state.messages,
          [conversationId]: [...currentMessages, action.payload],
        },
        // Update conversation's last message
        conversations: state.conversations.map(conv =>
          conv.id === conversationId
            ? { ...conv, lastMessage: action.payload, lastMessageAt: action.payload.createdAt }
            : conv
        ),
      };
    
    case 'UPDATE_MESSAGE':
      const msgConversationId = action.payload.conversationId;
      const messagesForConv = state.messages[msgConversationId] || [];
      
      return {
        ...state,
        messages: {
          ...state.messages,
          [msgConversationId]: messagesForConv.map(msg =>
            msg.id === action.payload.id ? action.payload : msg
          ),
        },
      };
    
    case 'DELETE_MESSAGE':
      const { conversationId: delConvId, messageId } = action.payload;
      const messagesForDelConv = state.messages[delConvId] || [];
      
      return {
        ...state,
        messages: {
          ...state.messages,
          [delConvId]: messagesForDelConv.filter(msg => msg.id !== messageId),
        },
      };
    
    case 'SET_ONLINE_USERS':
      return { ...state, onlineUsers: action.payload };
    
    case 'UPDATE_USER_PRESENCE':
      return {
        ...state,
        onlineUsers: state.onlineUsers.map(user =>
          user.userId === action.payload.userId ? action.payload : user
        ),
      };
    
    case 'SET_TYPING_USERS':
      return {
        ...state,
        typingUsers: {
          ...state.typingUsers,
          [action.payload.conversationId]: action.payload.userIds,
        },
      };
    
    case 'SET_UNREAD_COUNT':
      return {
        ...state,
        unreadCounts: {
          ...state.unreadCounts,
          [action.payload.conversationId]: action.payload.count,
        },
      };
    
    default:
      return state;
  }
}

// Context
interface ChatContextType {
  state: ChatState;
  dispatch: React.Dispatch<ChatAction>;
  // Actions
  connectToChat: (userId: string) => Promise<void>;
  disconnectFromChat: () => void;
  loadConversations: () => Promise<void>;
  loadMessages: (conversationId: string) => Promise<void>;
  sendMessage: (conversationId: string, content: string, replyToMessageId?: string) => Promise<void>;
  setActiveConversation: (conversationId: string | null) => void;
  sendTypingStatus: (conversationId: string, isTyping: boolean) => void;
  markAsRead: (conversationId: string) => Promise<void>;
  addReaction: (messageId: string, emoji: string) => Promise<void>;
  removeReaction: (messageId: string) => Promise<void>;
  createConversation: (participantIds: string[], name?: string, type?: 'DIRECT' | 'GROUP') => Promise<Conversation>;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

// Provider
interface ChatProviderProps {
  children: ReactNode;
}

export function ChatProvider({ children }: ChatProviderProps) {
  const [state, dispatch] = useReducer(chatReducer, initialState);

  // Connect to chat
  const connectToChat = async (userId: string) => {
    try {
      dispatch({ type: 'SET_LOADING', payload: true });
      dispatch({ type: 'SET_ERROR', payload: null });

      chatService.setCurrentUserId(userId);
      
      // Connect WebSocket
      await chatService.connectWebSocket(userId);
      dispatch({ type: 'SET_CONNECTED', payload: true });

      // Setup WebSocket event listeners
      setupWebSocketListeners();

      // Load initial data
      await loadConversations();
      await loadOnlineUsers();

      dispatch({ type: 'SET_LOADING', payload: false });
    } catch (error) {
      console.error('Failed to connect to chat:', error);
      dispatch({ type: 'SET_ERROR', payload: 'Không thể kết nối đến chat' });
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };

  // Disconnect from chat
  const disconnectFromChat = () => {
    chatService.disconnectWebSocket();
    dispatch({ type: 'SET_CONNECTED', payload: false });
  };

  // Setup WebSocket event listeners
  const setupWebSocketListeners = () => {
    // New message
    chatService.onNewMessage((message: Message) => {
      dispatch({ type: 'ADD_MESSAGE', payload: message });
      
      // Update unread count if not in active conversation
      if (message.conversationId !== state.activeConversationId) {
        const currentCount = state.unreadCounts[message.conversationId] || 0;
        dispatch({
          type: 'SET_UNREAD_COUNT',
          payload: { conversationId: message.conversationId, count: currentCount + 1 }
        });
      }
    });

    // Message update (edit)
    chatService.onMessageUpdate((message: Message) => {
      dispatch({ type: 'UPDATE_MESSAGE', payload: message });
    });

    // Message delete
    chatService.onMessageDelete((messageId: string) => {
      // Find which conversation this message belongs to
      for (const [conversationId, messages] of Object.entries(state.messages)) {
        if (messages.some(msg => msg.id === messageId)) {
          dispatch({ type: 'DELETE_MESSAGE', payload: { conversationId, messageId } });
          break;
        }
      }
    });

    // Typing indicator
    chatService.onTyping(({ userId, isTyping }: { userId: string; isTyping: boolean }) => {
      // This would need the conversationId from the server
      // For now, we'll handle this in the component level
    });

    // Presence update
    chatService.onPresenceUpdate((presence: UserPresence) => {
      dispatch({ type: 'UPDATE_USER_PRESENCE', payload: presence });
    });

    // Reaction update
    chatService.onReactionUpdate((message: Message) => {
      dispatch({ type: 'UPDATE_MESSAGE', payload: message });
    });
  };

  // Load conversations
  const loadConversations = async () => {
    try {
      const response = await chatService.getConversations();
      dispatch({ type: 'SET_CONVERSATIONS', payload: response.content });
      
      // Load unread counts for all conversations
      for (const conversation of response.content) {
        const unreadCount = await chatService.getUnreadCount(conversation.id);
        dispatch({
          type: 'SET_UNREAD_COUNT',
          payload: { conversationId: conversation.id, count: unreadCount }
        });
      }
    } catch (error) {
      console.error('Failed to load conversations:', error);
      dispatch({ type: 'SET_ERROR', payload: 'Không thể tải danh sách cuộc trò chuyện' });
    }
  };

  // Load messages for a conversation
  const loadMessages = async (conversationId: string) => {
    try {
      const response = await chatService.getMessages(conversationId);
      dispatch({
        type: 'SET_MESSAGES',
        payload: { conversationId, messages: response.content.reverse() } // Reverse to show oldest first
      });
    } catch (error) {
      console.error('Failed to load messages:', error);
      dispatch({ type: 'SET_ERROR', payload: 'Không thể tải tin nhắn' });
    }
  };

  // Load online users
  const loadOnlineUsers = async () => {
    try {
      const onlineUsers = await chatService.getOnlineUsers();
      dispatch({ type: 'SET_ONLINE_USERS', payload: onlineUsers });
    } catch (error) {
      console.error('Failed to load online users:', error);
    }
  };

  // Send message
  const sendMessage = async (conversationId: string, content: string, replyToMessageId?: string) => {
    try {
      const message = await chatService.sendMessage({
        conversationId,
        content,
        replyToMessageId
      });
      // Message will be added via WebSocket event
    } catch (error) {
      console.error('Failed to send message:', error);
      dispatch({ type: 'SET_ERROR', payload: 'Không thể gửi tin nhắn' });
    }
  };

  // Set active conversation
  const setActiveConversation = async (conversationId: string | null) => {
    dispatch({ type: 'SET_ACTIVE_CONVERSATION', payload: conversationId });
    
    if (conversationId) {
      // Load messages if not already loaded
      if (!state.messages[conversationId]) {
        await loadMessages(conversationId);
      }
      
      // Mark as read
      await markAsRead(conversationId);
    }
  };

  // Send typing status
  const sendTypingStatus = (conversationId: string, isTyping: boolean) => {
    chatService.sendTypingStatus(conversationId, isTyping);
  };

  // Mark conversation as read
  const markAsRead = async (conversationId: string) => {
    try {
      await chatService.markAsRead(conversationId);
      dispatch({
        type: 'SET_UNREAD_COUNT',
        payload: { conversationId, count: 0 }
      });
    } catch (error) {
      console.error('Failed to mark as read:', error);
    }
  };

  // Add reaction
  const addReaction = async (messageId: string, emoji: string) => {
    try {
      await chatService.addReaction({ messageId, emoji });
      // Reaction will be updated via WebSocket event
    } catch (error) {
      console.error('Failed to add reaction:', error);
    }
  };

  // Remove reaction
  const removeReaction = async (messageId: string) => {
    try {
      await chatService.removeReaction(messageId);
      // Reaction will be updated via WebSocket event
    } catch (error) {
      console.error('Failed to remove reaction:', error);
    }
  };

  // Create conversation
  const createConversation = async (
    participantIds: string[], 
    name?: string, 
    type: 'DIRECT' | 'GROUP' = 'DIRECT'
  ): Promise<Conversation> => {
    try {
      const conversation = await chatService.createConversation({
        participantIds,
        name,
        type
      });
      dispatch({ type: 'ADD_CONVERSATION', payload: conversation });
      return conversation;
    } catch (error) {
      console.error('Failed to create conversation:', error);
      dispatch({ type: 'SET_ERROR', payload: 'Không thể tạo cuộc trò chuyện' });
      throw error;
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnectFromChat();
    };
  }, []);

  const contextValue: ChatContextType = {
    state,
    dispatch,
    connectToChat,
    disconnectFromChat,
    loadConversations,
    loadMessages,
    sendMessage,
    setActiveConversation,
    sendTypingStatus,
    markAsRead,
    addReaction,
    removeReaction,
    createConversation,
  };

  return (
    <ChatContext.Provider value={contextValue}>
      {children}
    </ChatContext.Provider>
  );
}

// Hook to use Chat Context
export function useChat() {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
}
