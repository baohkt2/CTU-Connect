import React, { useState, useRef, useEffect } from 'react';
import { useChat } from '@/contexts/ChatContext';
import { ChatMessage, ChatRoom } from '@/shared/types/chat';
import { formatDistanceToNow } from 'date-fns';
import {
  PaperAirplaneIcon,
  ArrowLeftIcon,
  FaceSmileIcon,
  PaperClipIcon,
  PhoneIcon,
  VideoCameraIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';

interface MessageAreaProps {
  conversation: ChatRoom;
  messages: ChatMessage[];
  currentUserId: string;
  onBack?: () => void;
}

const MessageArea: React.FC<MessageAreaProps> = ({
  conversation,
  messages,
  currentUserId,
  onBack
}) => {
  const { sendMessage, startTyping, stopTyping, typingUsers, onlineUsers } = useChat();
  const [newMessage, setNewMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const typingTimeoutRef = useRef<NodeJS.Timeout>();

  // Auto scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle typing indicators
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    setNewMessage(value);

    if (value.trim() && !isTyping) {
      setIsTyping(true);
      startTyping(conversation.id);
    }

    // Clear existing timeout
    if (typingTimeoutRef.current) {
      clearTimeout(typingTimeoutRef.current);
    }

    // Set new timeout to stop typing indicator
    typingTimeoutRef.current = setTimeout(() => {
      if (isTyping) {
        setIsTyping(false);
        stopTyping(conversation.id);
      }
    }, 1000);
  };

  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();

    if (!newMessage.trim()) return;

    sendMessage(conversation.id, newMessage.trim());
    setNewMessage('');

    // Stop typing indicator
    if (isTyping) {
      setIsTyping(false);
      stopTyping(conversation.id);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e);
    }
  };

  // Get other participants (excluding current user)
  const otherParticipants = conversation.participants.filter(p => p.id !== currentUserId);
  const conversationName = conversation.name || otherParticipants.map(p => p.name).join(', ');

  // Get typing users for this conversation
  const currentTypingUsers = typingUsers[conversation.id] || new Set();
  const typingUserNames = Array.from(currentTypingUsers);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-white">
        <div className="flex items-center space-x-3">
          {onBack && (
            <button
              onClick={onBack}
              className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full"
            >
              <ArrowLeftIcon className="h-5 w-5" />
            </button>
          )}

          <div>
            <h2 className="text-lg font-semibold text-gray-900">{conversationName}</h2>
            <div className="flex items-center space-x-2">
              {otherParticipants.map((participant) => {
                const isOnline = onlineUsers.has(participant.id);
                return (
                  <span key={participant.id} className="text-sm text-gray-500 flex items-center">
                    <span className={`w-2 h-2 rounded-full mr-1 ${isOnline ? 'bg-green-400' : 'bg-gray-300'}`} />
                    {participant.name}
                  </span>
                );
              })}
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <button className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full">
            <PhoneIcon className="h-5 w-5" />
          </button>
          <button className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full">
            <VideoCameraIcon className="h-5 w-5" />
          </button>
          <button className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full">
            <InformationCircleIcon className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500">
            <div className="text-4xl mb-4">💬</div>
            <p>No messages yet. Start the conversation!</p>
          </div>
        ) : (
          messages.map((message, index) => {
            const isOwn = message.senderId === currentUserId;
            const showAvatar = index === 0 || messages[index - 1].senderId !== message.senderId;
            const showTimestamp = index === 0 ||
              new Date(message.createdAt).getTime() - new Date(messages[index - 1].createdAt).getTime() > 300000; // 5 minutes

            return (
              <div key={message.id} className={`flex ${isOwn ? 'justify-end' : 'justify-start'}`}>
                <div className={`flex max-w-xs lg:max-w-md ${isOwn ? 'flex-row-reverse' : 'flex-row'}`}>
                  {/* Avatar */}
                  {showAvatar && !isOwn && (
                    <div className="flex-shrink-0 w-8 h-8 bg-gray-300 rounded-full mr-3 flex items-center justify-center">
                      <span className="text-xs font-medium text-gray-600">
                        {message.sender.name?.charAt(0).toUpperCase()}
                      </span>
                    </div>
                  )}

                  <div className={`${!showAvatar && !isOwn ? 'ml-11' : ''}`}>
                    {/* Timestamp */}
                    {showTimestamp && (
                      <div className="text-xs text-gray-500 text-center mb-2">
                        {formatDistanceToNow(new Date(message.createdAt), { addSuffix: true })}
                      </div>
                    )}

                    {/* Message bubble */}
                    <div
                      className={`px-4 py-2 rounded-lg ${
                        isOwn
                          ? 'bg-indigo-600 text-white'
                          : 'bg-gray-100 text-gray-900'
                      }`}
                    >
                      {!isOwn && showAvatar && (
                        <div className="text-xs font-medium mb-1">{message.sender.name}</div>
                      )}
                      <p className="text-sm whitespace-pre-wrap">{message.content}</p>

                      {/* Message status */}
                      {isOwn && (
                        <div className="text-xs text-indigo-200 mt-1 text-right">
                          {message.isRead ? 'Read' : 'Sent'}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            );
          })
        )}

        {/* Typing indicator */}
        {typingUserNames.length > 0 && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-lg px-4 py-2 text-sm text-gray-600">
              {typingUserNames.length === 1
                ? `${typingUserNames[0]} is typing...`
                : `${typingUserNames.join(', ')} are typing...`
              }
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Message Input */}
      <div className="p-4 border-t border-gray-200">
        <form onSubmit={handleSendMessage} className="flex items-end space-x-2">
          <button
            type="button"
            className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full"
          >
            <PaperClipIcon className="h-5 w-5" />
          </button>

          <div className="flex-1">
            <textarea
              value={newMessage}
              onChange={handleInputChange}
              onKeyPress={handleKeyPress}
              placeholder="Type a message..."
              rows={1}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none"
              style={{ minHeight: '40px', maxHeight: '120px' }}
            />
          </div>

          <button
            type="button"
            className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full"
          >
            <FaceSmileIcon className="h-5 w-5" />
          </button>

          <button
            type="submit"
            disabled={!newMessage.trim()}
            className="p-2 bg-indigo-600 text-white rounded-full hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <PaperAirplaneIcon className="h-5 w-5" />
          </button>
        </form>
      </div>
    </div>
  );
};

export default MessageArea;
