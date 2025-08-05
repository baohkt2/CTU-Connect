import React, { useState, useEffect } from 'react';
import { useChat } from '../../contexts/ChatContext';
import { Conversation } from '../../services/chatService';
import { formatDistanceToNow } from 'date-fns';
import { vi } from 'date-fns/locale';
import { MagnifyingGlassIcon, UsersIcon, UserIcon } from '@heroicons/react/24/outline';

interface ConversationListProps {
  isMobile: boolean;
  onConversationSelect?: () => void;
}

const ConversationList: React.FC<ConversationListProps> = ({ isMobile, onConversationSelect }) => {
  const { state, setActiveConversation, loadConversations } = useChat();
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredConversations, setFilteredConversations] = useState<Conversation[]>([]);

  // Filter conversations based on search query
  useEffect(() => {
    if (searchQuery.trim() === '') {
      setFilteredConversations(state.conversations);
    } else {
      const filtered = state.conversations.filter(conversation =>
        conversation.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        conversation.participants.some(p =>
          p.userName.toLowerCase().includes(searchQuery.toLowerCase())
        )
      );
      setFilteredConversations(filtered);
    }
  }, [searchQuery, state.conversations]);

  const handleConversationClick = (conversationId: string) => {
    setActiveConversation(conversationId);
    onConversationSelect?.();
  };

  const getConversationName = (conversation: Conversation) => {
    if (conversation.type === 'GROUP') {
      return conversation.name || 'Nhóm chat';
    } else {
      // For direct messages, show the other participant's name
      const otherParticipant = conversation.participants.find(
        p => p.userId !== state.activeConversationId // This should be current user ID
      );
      return otherParticipant?.userName || 'Chat trực tiếp';
    }
  };

  const getConversationAvatar = (conversation: Conversation) => {
    if (conversation.type === 'GROUP') {
      return conversation.avatarUrl || null;
    } else {
      const otherParticipant = conversation.participants.find(
        p => p.userId !== state.activeConversationId // This should be current user ID
      );
      return otherParticipant?.userAvatar || null;
    }
  };

  const getLastMessagePreview = (conversation: Conversation) => {
    if (!conversation.lastMessage) return 'Chưa có tin nhắn';

    const message = conversation.lastMessage;
    let preview = '';

    if (message.type === 'TEXT') {
      preview = message.content;
    } else if (message.type === 'IMAGE') {
      preview = '📷 Hình ảnh';
    } else if (message.type === 'FILE') {
      preview = '📎 File đính kèm';
    } else if (message.type === 'SYSTEM') {
      preview = message.content;
    }

    // Truncate long messages
    if (preview.length > 50) {
      preview = preview.substring(0, 50) + '...';
    }

    return preview;
  };

  const getLastMessageTime = (conversation: Conversation) => {
    if (!conversation.lastMessageAt) return '';

    try {
      return formatDistanceToNow(new Date(conversation.lastMessageAt), {
        addSuffix: true,
        locale: vi
      });
    } catch {
      return '';
    }
  };

  const getOnlineStatus = (conversation: Conversation) => {
    if (conversation.type === 'GROUP') return null;

    const otherParticipant = conversation.participants.find(
      p => p.userId !== state.activeConversationId // This should be current user ID
    );

    if (!otherParticipant) return null;

    const onlineUser = state.onlineUsers.find(u => u.userId === otherParticipant.userId);
    return onlineUser?.status || 'OFFLINE';
  };

  return (
    <div className="flex flex-col h-full">
      {/* Search Bar */}
      <div className="p-4 border-b">
        <div className="relative">
          <MagnifyingGlassIcon className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
          <input
            type="text"
            placeholder="Tìm kiếm cuộc trò chuyện..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
      </div>

      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto">
        {filteredConversations.length === 0 ? (
          <div className="p-4 text-center text-gray-500">
            {searchQuery ? 'Không tìm thấy cuộc trò chuyện nào' : 'Chưa có cuộc trò chuyện nào'}
          </div>
        ) : (
          <div className="divide-y divide-gray-200">
            {filteredConversations.map((conversation) => {
              const unreadCount = state.unreadCounts[conversation.id] || 0;
              const isActive = state.activeConversationId === conversation.id;
              const onlineStatus = getOnlineStatus(conversation);

              return (
                <div
                  key={conversation.id}
                  onClick={() => handleConversationClick(conversation.id)}
                  className={`p-4 cursor-pointer hover:bg-gray-50 transition-colors ${
                    isActive ? 'bg-blue-50 border-r-2 border-blue-500' : ''
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    {/* Avatar */}
                    <div className="relative flex-shrink-0">
                      {getConversationAvatar(conversation) ? (
                        <img
                          src={getConversationAvatar(conversation)!}
                          alt={getConversationName(conversation)}
                          className="w-12 h-12 rounded-full object-cover"
                        />
                      ) : (
                        <div className="w-12 h-12 rounded-full bg-gray-300 flex items-center justify-center">
                          {conversation.type === 'GROUP' ? (
                            <UsersIcon className="w-6 h-6 text-gray-600" />
                          ) : (
                            <UserIcon className="w-6 h-6 text-gray-600" />
                          )}
                        </div>
                      )}

                      {/* Online indicator for direct messages */}
                      {conversation.type === 'DIRECT' && onlineStatus === 'ONLINE' && (
                        <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 border-2 border-white rounded-full"></div>
                      )}
                    </div>

                    {/* Conversation Info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between">
                        <h3 className={`text-sm font-medium truncate ${
                          unreadCount > 0 ? 'text-gray-900' : 'text-gray-700'
                        }`}>
                          {getConversationName(conversation)}
                        </h3>
                        <div className="flex items-center space-x-2">
                          {unreadCount > 0 && (
                            <span className="inline-flex items-center justify-center px-2 py-1 text-xs font-bold leading-none text-white bg-red-500 rounded-full min-w-[20px]">
                              {unreadCount > 99 ? '99+' : unreadCount}
                            </span>
                          )}
                          <span className="text-xs text-gray-500">
                            {getLastMessageTime(conversation)}
                          </span>
                        </div>
                      </div>

                      <p className={`text-sm truncate mt-1 ${
                        unreadCount > 0 ? 'text-gray-900 font-medium' : 'text-gray-500'
                      }`}>
                        {getLastMessagePreview(conversation)}
                      </p>

                      {/* Participants count for group chats */}
                      {conversation.type === 'GROUP' && (
                        <div className="flex items-center mt-1 text-xs text-gray-400">
                          <UsersIcon className="w-3 h-3 mr-1" />
                          {conversation.participants.length} thành viên
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Refresh Button */}
      <div className="p-4 border-t">
        <button
          onClick={loadConversations}
          disabled={state.isLoading}
          className="w-full py-2 px-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {state.isLoading ? 'Đang tải...' : 'Làm mới'}
        </button>
      </div>
    </div>
  );
};

export default ConversationList;
