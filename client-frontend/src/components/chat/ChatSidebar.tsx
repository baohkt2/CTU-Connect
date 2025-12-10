'use client';

import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import api from '@/lib/api';
import { toast } from 'react-hot-toast';

interface Conversation {
  id: string;
  type: 'DIRECT' | 'GROUP';
  name?: string;
  participants: Array<{
    id: string;
    fullName: string;
    avatarUrl?: string;
    isOnline?: boolean;
  }>;
  lastMessage?: {
    content: string;
    createdAt: string;
    senderId: string;
  };
  unreadCount?: number;
  lastMessageAt?: string;
}

interface ChatSidebarProps {
  selectedConversationId: string | null;
  onSelectConversation: (id: string) => void;
  friendUserId?: string | null;
}

export default function ChatSidebar({
  selectedConversationId,
  onSelectConversation,
  friendUserId,
}: ChatSidebarProps) {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [creatingConversation, setCreatingConversation] = useState(false);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const processedFriendRef = useRef<string | null>(null);
  const isLoadingRef = useRef(false);

  // Memoize comparison function to check if conversations changed
  const conversationsChanged = useCallback((prev: Conversation[], next: Conversation[]) => {
    if (prev.length !== next.length) return true;
    
    for (let i = 0; i < prev.length; i++) {
      if (prev[i].id !== next[i].id || 
          prev[i].lastMessageAt !== next[i].lastMessageAt ||
          prev[i].unreadCount !== next[i].unreadCount) {
        return true;
      }
    }
    return false;
  }, []);

  const loadConversations = useCallback(async () => {
    // Prevent concurrent requests
    if (isLoadingRef.current) return;
    
    try {
      isLoadingRef.current = true;
      const response = await api.get('/chats/conversations');
      const newConversations = response.data.content || [];
      
      // Only update if actually changed to prevent unnecessary re-renders
      setConversations(prevConversations => {
        if (conversationsChanged(prevConversations, newConversations)) {
          return newConversations;
        }
        return prevConversations;
      });
    } catch (error) {
      console.error('Error loading conversations:', error);
      // Don't show error toast if it's just empty conversations or polling error
      if (loading) {
        // toast.error('Không thể tải danh sách trò chuyện');
      }
    } finally {
      setLoading(false);
      isLoadingRef.current = false;
    }
  }, [conversationsChanged, loading]);

  // Polling to refresh conversations list
  useEffect(() => {
    loadConversations();
    
    // Poll every 5 seconds to update conversation list (increased from 3s to reduce flicker)
    pollingIntervalRef.current = setInterval(() => {
      loadConversations();
    }, 5000);

    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, [loadConversations]);

  // Auto-create/select conversation if friendUserId is provided
  useEffect(() => {
    if (friendUserId && 
        !creatingConversation && 
        processedFriendRef.current !== friendUserId) {
      processedFriendRef.current = friendUserId;
      createOrGetConversationWithFriend(friendUserId);
    }
  }, [friendUserId, creatingConversation]);

  const createOrGetConversationWithFriend = useCallback(async (friendId: string) => {
    if (creatingConversation) return; // Prevent duplicate calls
    
    try {
      setCreatingConversation(true);
      const response = await api.post(`/chats/conversations/direct/${friendId}`);
      if (response.data && response.data.id) {
        onSelectConversation(response.data.id);
        // Reload conversations to include the new one
        await loadConversations();
      }
    } catch (error) {
      console.error('Error creating conversation:', error);
      toast.error('Không thể tạo cuộc trò chuyện');
    } finally {
      setCreatingConversation(false);
    }
  }, [creatingConversation, onSelectConversation, loadConversations]);

  const getConversationName = useCallback((conversation: Conversation) => {
    if (conversation.type === 'GROUP') {
      return conversation.name || 'Nhóm chat';
    }
    // For DIRECT, show the other participant's name
    const otherParticipant = conversation.participants[0];
    return otherParticipant?.fullName || 'Người dùng';
  }, []);

  const getConversationAvatar = useCallback((conversation: Conversation) => {
    if (conversation.type === 'GROUP') {
      return null; // Group avatar
    }
    const otherParticipant = conversation.participants[0];
    return otherParticipant?.avatarUrl;
  }, []);

  const formatLastMessageTime = useCallback((dateString?: string) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Vừa xong';
    if (diffMins < 60) return `${diffMins} phút`;
    if (diffHours < 24) return `${diffHours} giờ`;
    if (diffDays < 7) return `${diffDays} ngày`;
    return date.toLocaleDateString('vi-VN');
  }, []);

  const filteredConversations = useMemo(() => 
    conversations.filter(conv =>
      getConversationName(conv).toLowerCase().includes(searchQuery.toLowerCase())
    ),
    [conversations, searchQuery, getConversationName]
  );

  if (loading) {
    return (
      <div className="w-80 border-r border-gray-200 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="w-96 border-r border-gray-200 flex flex-col bg-white shadow-sm">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-indigo-50">
        <h2 className="text-2xl font-bold mb-4 text-gray-800">Tin nhắn</h2>
        {/* Search */}
        <div className="relative">
          <input
            type="text"
            placeholder="Tìm kiếm cuộc trò chuyện..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white shadow-sm"
          />
          <svg
            className="absolute left-3 top-3 h-5 w-5 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        </div>
      </div>

      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto">
        {creatingConversation && (
          <div className="flex items-center justify-center py-4 bg-blue-50">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600 mr-2"></div>
            <span className="text-sm text-blue-600">Đang tạo cuộc trò chuyện...</span>
          </div>
        )}
        
        {filteredConversations.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500 p-6">
            <div className="bg-gray-100 rounded-full p-6 mb-4">
              <svg
                className="h-16 w-16 text-gray-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                />
              </svg>
            </div>
            <p className="text-center text-lg font-medium text-gray-600 mb-1">
              Chưa có cuộc trò chuyện nào
            </p>
            <p className="text-center text-sm text-gray-500">
              Bắt đầu trò chuyện với bạn bè của bạn
            </p>
          </div>
        ) : (
          <div className="divide-y divide-gray-100">
            {filteredConversations.map((conversation) => (
              <ConversationItem
                key={conversation.id}
                conversation={conversation}
                isSelected={selectedConversationId === conversation.id}
                onSelect={() => onSelectConversation(conversation.id)}
                getConversationName={getConversationName}
                getConversationAvatar={getConversationAvatar}
                formatLastMessageTime={formatLastMessageTime}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// Memoized conversation item component to prevent unnecessary re-renders
const ConversationItem = React.memo(({
  conversation,
  isSelected,
  onSelect,
  getConversationName,
  getConversationAvatar,
  formatLastMessageTime,
}: {
  conversation: Conversation;
  isSelected: boolean;
  onSelect: () => void;
  getConversationName: (conv: Conversation) => string;
  getConversationAvatar: (conv: Conversation) => string | null | undefined;
  formatLastMessageTime: (date?: string) => string;
}) => {
  const conversationName = getConversationName(conversation);
  const conversationAvatar = getConversationAvatar(conversation);
  
  return (
    <div
      onClick={onSelect}
      className={`flex items-center p-4 cursor-pointer transition-all duration-150 ${
        isSelected 
          ? 'bg-blue-50 border-l-4 border-blue-600' 
          : 'hover:bg-gray-50 border-l-4 border-transparent'
      }`}
    >
      {/* Avatar */}
      <div className="relative mr-3 flex-shrink-0">
        <div className="w-14 h-14 rounded-full bg-gradient-to-br from-blue-400 to-indigo-500 flex items-center justify-center overflow-hidden shadow-md">
          {conversationAvatar ? (
            <img
              src={conversationAvatar}
              alt={conversationName}
              className="w-full h-full object-cover"
            />
          ) : (
            <span className="text-white text-xl font-bold">
              {conversationName.charAt(0).toUpperCase()}
            </span>
          )}
        </div>
        {/* Online status for direct chats */}
        {conversation.type === 'DIRECT' &&
          conversation.participants[0]?.isOnline && (
            <div className="absolute bottom-0 right-0 w-4 h-4 bg-green-500 border-2 border-white rounded-full"></div>
          )}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1">
          <h3 className="font-semibold text-gray-900 truncate text-base">
            {conversationName}
          </h3>
          <span className="text-xs text-gray-500 ml-2 flex-shrink-0">
            {formatLastMessageTime(conversation.lastMessageAt)}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <p className={`text-sm truncate ${
            conversation.unreadCount && conversation.unreadCount > 0
              ? 'text-gray-900 font-medium'
              : 'text-gray-600'
          }`}>
            {conversation.lastMessage?.content || 'Bắt đầu cuộc trò chuyện'}
          </p>
          {conversation.unreadCount && conversation.unreadCount > 0 && (
            <span className="ml-2 px-2 py-0.5 bg-blue-600 text-white text-xs rounded-full font-semibold flex-shrink-0">
              {conversation.unreadCount > 99 ? '99+' : conversation.unreadCount}
            </span>
          )}
        </div>
      </div>
    </div>
  );
});
