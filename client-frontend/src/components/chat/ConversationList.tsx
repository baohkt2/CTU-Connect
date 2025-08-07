import React, { useState, useEffect } from 'react';
import { useChat } from '@/contexts/ChatContext';
import { ChatRoom } from '@/shared/types/chat';
import { formatDistanceToNow } from 'date-fns';
import { MagnifyingGlassIcon, UsersIcon, UserIcon } from '@heroicons/react/24/outline';

interface ConversationListProps {
  conversations: ChatRoom[];
  activeConversationId: string | null;
  onConversationSelect: (conversationId: string) => void;
  currentUserId: string;
}

const ConversationList: React.FC<ConversationListProps> = ({
  conversations,
  activeConversationId,
  onConversationSelect,
  currentUserId
}) => {
  const { getUnreadCount } = useChat();
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredConversations, setFilteredConversations] = useState<ChatRoom[]>([]);

  // Filter conversations based on search query
  useEffect(() => {
    if (searchQuery.trim() === '') {
      setFilteredConversations(conversations);
    } else {
      const filtered = conversations.filter(conversation =>
        conversation.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        conversation.participants.some(p =>
          p.name?.toLowerCase().includes(searchQuery.toLowerCase())
        )
      );
      setFilteredConversations(filtered);
    }
  }, [searchQuery, conversations]);

  const handleConversationClick = (conversationId: string) => {
    onConversationSelect(conversationId);
  };

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Search Bar */}
      <div className="p-4 border-b border-gray-200">
        <div className="relative">
          <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search conversations..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
          />
        </div>
      </div>

      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto">
        {filteredConversations.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <div className="text-4xl mb-4">💬</div>
            <p className="text-center">
              {searchQuery ? 'No conversations found' : 'No conversations yet'}
            </p>
          </div>
        ) : (
          filteredConversations.map((conversation) => {
            const unreadCount = getUnreadCount(conversation.id);
            const isActive = activeConversationId === conversation.id;

            return (
              <div
                key={conversation.id}
                onClick={() => handleConversationClick(conversation.id)}
                className={`p-4 border-b border-gray-100 cursor-pointer hover:bg-gray-50 transition-colors ${
                  isActive ? 'bg-indigo-50 border-indigo-200' : ''
                }`}
              >
                <div className="flex items-start space-x-3">
                  {/* Avatar */}
                  <div className="flex-shrink-0">
                    {conversation.type === 'GROUP' ? (
                      <div className="w-12 h-12 bg-indigo-100 rounded-full flex items-center justify-center">
                        <UsersIcon className="h-6 w-6 text-indigo-600" />
                      </div>
                    ) : (
                      <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center">
                        <UserIcon className="h-6 w-6 text-gray-600" />
                      </div>
                    )}
                  </div>

                  {/* Conversation Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <h3 className={`text-sm font-medium truncate ${
                        isActive ? 'text-indigo-900' : 'text-gray-900'
                      }`}>
                        {conversation.name || 'Unnamed Conversation'}
                      </h3>
                      {conversation.lastMessage && (
                        <span className="text-xs text-gray-500">
                          {formatDistanceToNow(new Date(conversation.lastMessage.createdAt), {
                            addSuffix: true,
                            locale: { localize: { ordinalNumber: (n) => n } }
                          })}
                        </span>
                      )}
                    </div>

                    {conversation.lastMessage && (
                      <p className="text-sm text-gray-600 truncate mt-1">
                        <span className="font-medium">
                          {conversation.lastMessage.sender.name}:
                        </span>
                        {conversation.lastMessage.content}
                      </p>
                    )}

                    {/* Participants count for group chats */}
                    {conversation.type === 'GROUP' && (
                      <p className="text-xs text-gray-500 mt-1">
                        {conversation.participants.length} members
                      </p>
                    )}
                  </div>

                  {/* Unread Badge */}
                  {unreadCount > 0 && (
                    <div className="flex-shrink-0">
                      <span className="inline-flex items-center justify-center px-2 py-1 text-xs font-bold leading-none text-white bg-red-500 rounded-full min-w-[20px]">
                        {unreadCount > 99 ? '99+' : unreadCount}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default ConversationList;
