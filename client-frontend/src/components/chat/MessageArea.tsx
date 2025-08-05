import React, { useState, useEffect, useRef } from 'react';
import { useChat } from '../../contexts/ChatContext';
import { Message } from '../../services/chatService';
import MessageBubble from './MessageBubble';
import MessageInput from './MessageInput';
import TypingIndicator from './TypingIndicator';
import { ArrowLeftIcon, InformationCircleIcon } from '@heroicons/react/24/outline';
import { formatDistanceToNow } from 'date-fns';
import { vi } from 'date-fns/locale';

interface MessageAreaProps {
  conversationId: string;
  isMobile: boolean;
  onBackClick?: () => void;
}

const MessageArea: React.FC<MessageAreaProps> = ({ conversationId, isMobile, onBackClick }) => {
  const { state, loadMessages, sendMessage, sendTypingStatus, markAsRead } = useChat();
  const [replyToMessage, setReplyToMessage] = useState<Message | null>(null);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [hasMoreMessages, setHasMoreMessages] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const conversation = state.conversations.find(c => c.id === conversationId);
  const messages = state.messages[conversationId] || [];
  const typingUsers = state.typingUsers[conversationId] || [];

  // Auto scroll to bottom when new messages arrive
  useEffect(() => {
    scrollToBottom();
  }, [messages.length]);

  // Load messages when conversation changes
  useEffect(() => {
    loadMessages(conversationId);
    markAsRead(conversationId);
  }, [conversationId, loadMessages, markAsRead]);

  // Clear typing timeout on unmount
  useEffect(() => {
    return () => {
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
    };
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return;

    try {
      await sendMessage(conversationId, content, replyToMessage?.id);
      setReplyToMessage(null);
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };

  const handleTyping = (isTyping: boolean) => {
    sendTypingStatus(conversationId, isTyping);
    
    if (isTyping) {
      // Clear existing timeout
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
      
      // Set timeout to stop typing after 3 seconds of inactivity
      typingTimeoutRef.current = setTimeout(() => {
        sendTypingStatus(conversationId, false);
      }, 3000);
    } else {
      // Clear timeout and send stop typing immediately
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
    }
  };

  const handleReplyToMessage = (message: Message) => {
    setReplyToMessage(message);
  };

  const handleCancelReply = () => {
    setReplyToMessage(null);
  };

  const getConversationTitle = () => {
    if (!conversation) return 'Chat';
    
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

  const getOnlineStatus = () => {
    if (!conversation || conversation.type === 'GROUP') return null;
    
    const otherParticipant = conversation.participants.find(
      p => p.userId !== state.activeConversationId // This should be current user ID
    );
    
    if (!otherParticipant) return null;
    
    const onlineUser = state.onlineUsers.find(u => u.userId === otherParticipant.userId);
    return onlineUser?.status || 'OFFLINE';
  };

  const getLastSeenTime = () => {
    if (!conversation || conversation.type === 'GROUP') return null;
    
    const otherParticipant = conversation.participants.find(
      p => p.userId !== state.activeConversationId // This should be current user ID
    );
    
    if (!otherParticipant) return null;
    
    const onlineUser = state.onlineUsers.find(u => u.userId === otherParticipant.userId);
    if (!onlineUser || onlineUser.status === 'ONLINE') return null;
    
    try {
      return formatDistanceToNow(new Date(onlineUser.lastSeenAt), {
        addSuffix: true,
        locale: vi
      });
    } catch {
      return null;
    }
  };

  const loadMoreMessages = async () => {
    if (isLoadingMore || !hasMoreMessages) return;
    
    setIsLoadingMore(true);
    try {
      // This would need pagination support in the backend
      // For now, just simulate loading more messages
      await new Promise(resolve => setTimeout(resolve, 1000));
      setHasMoreMessages(false);
    } catch (error) {
      console.error('Failed to load more messages:', error);
    } finally {
      setIsLoadingMore(false);
    }
  };

  const handleScroll = () => {
    if (messagesContainerRef.current) {
      const { scrollTop } = messagesContainerRef.current;
      if (scrollTop === 0 && hasMoreMessages) {
        loadMoreMessages();
      }
    }
  };

  if (!conversation) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center text-gray-500">
          <p>Không tìm thấy cuộc trò chuyện</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b bg-white">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {isMobile && onBackClick && (
              <button
                onClick={onBackClick}
                className="p-1 hover:bg-gray-100 rounded-full"
              >
                <ArrowLeftIcon className="h-5 w-5" />
              </button>
            )}
            
            <div className="flex items-center space-x-3">
              {/* Avatar */}
              <div className="relative">
                {conversation.avatarUrl ? (
                  <img
                    src={conversation.avatarUrl}
                    alt={getConversationTitle()}
                    className="w-10 h-10 rounded-full object-cover"
                  />
                ) : (
                  <div className="w-10 h-10 rounded-full bg-gray-300 flex items-center justify-center">
                    <span className="text-sm font-medium text-gray-600">
                      {getConversationTitle().charAt(0).toUpperCase()}
                    </span>
                  </div>
                )}
                
                {/* Online indicator */}
                {conversation.type === 'DIRECT' && getOnlineStatus() === 'ONLINE' && (
                  <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 border-2 border-white rounded-full"></div>
                )}
              </div>
              
              {/* Conversation Info */}
              <div>
                <h3 className="font-medium text-gray-900">{getConversationTitle()}</h3>
                <div className="text-sm text-gray-500">
                  {conversation.type === 'GROUP' ? (
                    <span>{conversation.participants.length} thành viên</span>
                  ) : (
                    <span>
                      {getOnlineStatus() === 'ONLINE' 
                        ? 'Đang hoạt động' 
                        : getLastSeenTime() 
                          ? `Hoạt động ${getLastSeenTime()}`
                          : 'Không hoạt động'
                      }
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>
          
          {/* Action Buttons */}
          <div className="flex items-center space-x-2">
            <button className="p-2 hover:bg-gray-100 rounded-full">
              <InformationCircleIcon className="h-5 w-5 text-gray-500" />
            </button>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div 
        ref={messagesContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-4 space-y-4"
      >
        {/* Load More Button */}
        {hasMoreMessages && (
          <div className="text-center">
            <button
              onClick={loadMoreMessages}
              disabled={isLoadingMore}
              className="px-4 py-2 text-sm text-blue-600 hover:bg-blue-50 rounded-lg disabled:opacity-50"
            >
              {isLoadingMore ? 'Đang tải...' : 'Tải thêm tin nhắn'}
            </button>
          </div>
        )}

        {/* Messages */}
        {messages.map((message, index) => {
          const prevMessage = index > 0 ? messages[index - 1] : null;
          const nextMessage = index < messages.length - 1 ? messages[index + 1] : null;
          
          const showSenderInfo = !prevMessage || prevMessage.senderId !== message.senderId;
          const showTimestamp = !nextMessage || 
            nextMessage.senderId !== message.senderId ||
            (new Date(nextMessage.createdAt).getTime() - new Date(message.createdAt).getTime()) > 300000; // 5 minutes

          return (
            <MessageBubble
              key={message.id}
              message={message}
              showSenderInfo={showSenderInfo}
              showTimestamp={showTimestamp}
              onReply={() => handleReplyToMessage(message)}
              isMobile={isMobile}
            />
          );
        })}

        {/* Typing Indicator */}
        {typingUsers.length > 0 && (
          <TypingIndicator userIds={typingUsers} />
        )}

        {/* Scroll to bottom anchor */}
        <div ref={messagesEndRef} />
      </div>

      {/* Reply Preview */}
      {replyToMessage && (
        <div className="px-4 py-2 bg-gray-50 border-t">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <p className="text-sm text-gray-600">
                Trả lời <span className="font-medium">{replyToMessage.senderName}</span>
              </p>
              <p className="text-sm text-gray-800 truncate">
                {replyToMessage.content}
              </p>
            </div>
            <button
              onClick={handleCancelReply}
              className="ml-2 p-1 hover:bg-gray-200 rounded"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      {/* Message Input */}
      <MessageInput
        onSendMessage={handleSendMessage}
        onTyping={handleTyping}
        disabled={!state.isConnected}
        placeholder={
          state.isConnected 
            ? "Nhập tin nhắn..." 
            : "Đang kết nối..."
        }
      />
    </div>
  );
};

export default MessageArea;
