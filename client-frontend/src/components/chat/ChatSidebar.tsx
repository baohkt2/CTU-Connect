'use client';

import React, { useState, useEffect } from 'react';
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

  useEffect(() => {
    loadConversations();
  }, []);

  // Auto-create/select conversation if friendUserId is provided
  // Use ref to track if we've already created conversation for this friendUserId
  const processedFriendRef = React.useRef<string | null>(null);
  
  useEffect(() => {
    if (friendUserId && 
        !creatingConversation && 
        processedFriendRef.current !== friendUserId) {
      processedFriendRef.current = friendUserId;
      createOrGetConversationWithFriend(friendUserId);
    }
  }, [friendUserId]);

  const loadConversations = async () => {
    try {
      setLoading(true);
      const response = await api.get('/chats/conversations');
      setConversations(response.data.content || []);
    } catch (error) {
      console.error('Error loading conversations:', error);
      // Don't show error toast if it's just empty conversations
      if (error.response?.status !== 404) {
        toast.error('Không thể tải danh sách trò chuyện');
      }
    } finally {
      setLoading(false);
    }
  };

  const createOrGetConversationWithFriend = async (friendId: string) => {
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
  };

  const getConversationName = (conversation: Conversation) => {
    if (conversation.type === 'GROUP') {
      return conversation.name || 'Nhóm chat';
    }
    // For DIRECT, show the other participant's name
    const otherParticipant = conversation.participants[0];
    return otherParticipant?.fullName || 'Người dùng';
  };

  const getConversationAvatar = (conversation: Conversation) => {
    if (conversation.type === 'GROUP') {
      return null; // Group avatar
    }
    const otherParticipant = conversation.participants[0];
    return otherParticipant?.avatarUrl;
  };

  const formatLastMessageTime = (dateString?: string) => {
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
  };

  const filteredConversations = conversations.filter(conv =>
    getConversationName(conv).toLowerCase().includes(searchQuery.toLowerCase())
  );

  if (loading) {
    return (
      <div className="w-80 border-r border-gray-200 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="w-80 border-r border-gray-200 flex flex-col bg-white">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-2xl font-bold mb-3">Tin nhắn</h2>
        {/* Search */}
        <div className="relative">
          <input
            type="text"
            placeholder="Tìm kiếm..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <svg
            className="absolute left-3 top-2.5 h-5 w-5 text-gray-400"
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
        {filteredConversations.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500 p-4">
            <svg
              className="h-16 w-16 mb-2 text-gray-300"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
              />
            </svg>
            <p className="text-center">Chưa có cuộc trò chuyện nào</p>
          </div>
        ) : (
          filteredConversations.map((conversation) => (
            <div
              key={conversation.id}
              onClick={() => onSelectConversation(conversation.id)}
              className={`flex items-center p-3 cursor-pointer hover:bg-gray-100 transition-colors ${
                selectedConversationId === conversation.id ? 'bg-blue-50' : ''
              }`}
            >
              {/* Avatar */}
              <div className="relative mr-3">
                <div className="w-14 h-14 rounded-full bg-gray-300 flex items-center justify-center overflow-hidden">
                  {getConversationAvatar(conversation) ? (
                    <img
                      src={getConversationAvatar(conversation)!}
                      alt={getConversationName(conversation)}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <span className="text-white text-lg font-semibold">
                      {getConversationName(conversation).charAt(0).toUpperCase()}
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
                  <h3 className="font-semibold text-gray-900 truncate">
                    {getConversationName(conversation)}
                  </h3>
                  <span className="text-xs text-gray-500 ml-2">
                    {formatLastMessageTime(conversation.lastMessageAt)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <p className="text-sm text-gray-600 truncate">
                    {conversation.lastMessage?.content || 'Chưa có tin nhắn'}
                  </p>
                  {conversation.unreadCount && conversation.unreadCount > 0 && (
                    <span className="ml-2 px-2 py-0.5 bg-blue-600 text-white text-xs rounded-full">
                      {conversation.unreadCount}
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
