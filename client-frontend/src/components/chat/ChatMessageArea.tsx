/* eslint-disable @typescript-eslint/no-unused-vars */
'use client';

import React, { useState, useEffect, useRef } from 'react';
import api from '@/lib/api';
import { toast } from 'react-hot-toast';
import { useAuth } from '@/contexts/AuthContext';

interface Message {
  id: string;
  content: string;
  senderId: string;
  senderName: string;
  senderAvatar?: string;
  type: 'TEXT' | 'IMAGE' | 'FILE';
  createdAt: string;
  isEdited: boolean;
  attachment?: {
    fileName: string;
    fileUrl: string;
    fileType: string;
    fileSize: number;
    thumbnailUrl?: string;
  };
  replyToMessage?: {
    senderName: string;
    content: string;
  };
}

interface ChatMessageAreaProps {
  conversationId: string | null;
  currentUserId: string;
  conversationName?: string;
  conversationAvatar?: string;
  isOnline?: boolean;
}

export default function ChatMessageArea({
  conversationId,
  currentUserId,
  conversationName,
  conversationAvatar,
  isOnline,
}: ChatMessageAreaProps) {
  const { user } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [messageInput, setMessageInput] = useState('');
  const [uploadingFile, setUploadingFile] = useState(false);
  const [sendingMessage, setSendingMessage] = useState(false);
  const [newMessageIds, setNewMessageIds] = useState<Set<string>>(new Set());
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const lastMessageCountRef = useRef<number>(0);
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Polling mechanism - load messages every 2 seconds for real-time feel
  useEffect(() => {
    if (!conversationId) {
      // Clear interval if no conversation
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
      setMessages([]);
      lastMessageCountRef.current = 0;
      return;
    }

    // Initial load
    setLoading(true);
    loadMessages();

    // Set up polling interval (2 seconds for more real-time feel)
    pollingIntervalRef.current = setInterval(() => {
      loadMessages();
    }, 2000);

    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, [conversationId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const loadMessages = async () => {
    if (!conversationId) return;

    try {
      const response = await api.get(
        `/chats/messages/conversation/${conversationId}`
      );
      const newMessages = response.data.content?.reverse() || [];
      
      // Detect new messages and mark them for animation
      setMessages((prevMessages) => {
        // Check if we have genuinely new messages
        if (newMessages.length > prevMessages.length) {
          const prevIds = new Set(prevMessages.map(m => m.id));
          const newIds = newMessages
            .filter(m => !prevIds.has(m.id))
            .map(m => m.id);
          
          // Mark new message IDs for animation
          if (newIds.length > 0) {
            setNewMessageIds(new Set(newIds));
            // Clear animation class after 1 second
            setTimeout(() => {
              setNewMessageIds(new Set());
            }, 1000);
          }
        }
        
        // Only update if messages actually changed
        const prevStr = JSON.stringify(prevMessages.map(m => ({ id: m.id, content: m.content })));
        const newStr = JSON.stringify(newMessages.map(m => ({ id: m.id, content: m.content })));
        
        if (prevStr !== newStr) {
          lastMessageCountRef.current = newMessages.length;
          return newMessages;
        }
        return prevMessages;
      });
    } catch (error: any) {
      console.error('Error loading messages:', error);
      // Don't show error for 404 (empty conversation)
      if (error.response?.status === 404) {
        setMessages([]); // Just set empty messages
        lastMessageCountRef.current = 0;
      }
      // Silently fail for polling errors to avoid spamming toasts
    } finally {
      setLoading(false);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!messageInput.trim() || !conversationId || sendingMessage) return;

    const tempMessage = messageInput.trim();
    setMessageInput(''); // Clear immediately for better UX
    
    // Show typing indicator briefly
    setIsTyping(true);
    
    try {
      setSendingMessage(true);
      const response = await api.post('/chats/messages', {
        conversationId,
        content: tempMessage,
        type: 'TEXT',
      });

      // Add message to list immediately for better UX
      if (response.data) {
        setMessages((prev) => [...prev, response.data]);
        setNewMessageIds(new Set([response.data.id]));
        setTimeout(() => setNewMessageIds(new Set()), 1000);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      toast.error('Không thể gửi tin nhắn');
      setMessageInput(tempMessage); // Restore message on error
    } finally {
      setSendingMessage(false);
      setIsTyping(false);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !conversationId) return;

    try {
      setUploadingFile(true);

      // Upload to media service
      const formData = new FormData();
      formData.append('file', file);
      formData.append('type', file.type.startsWith('image/') ? 'image' : 'document');

      const uploadResponse = await api.post('/media/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      const mediaData = uploadResponse.data;
      
      if (mediaData && mediaData.cloudinaryUrl) {
        // Determine message type based on file type
        const messageType = file.type.startsWith('image/') ? 'IMAGE' : 'FILE';
        
        // Send message with attachment
        const messagePayload = {
          conversationId,
          content: mediaData.originalFileName || file.name,
          type: messageType,
          attachment: {
            fileName: mediaData.originalFileName || file.name,
            fileUrl: mediaData.cloudinaryUrl,
            fileType: mediaData.contentType || file.type,
            fileSize: mediaData.fileSize || file.size,
            thumbnailUrl: messageType === 'IMAGE' ? mediaData.cloudinaryUrl : undefined,
          },
        };

        const messageResponse = await api.post('/chats/messages', messagePayload);

        if (messageResponse.data) {
          setMessages((prev) => [...prev, messageResponse.data]);
          toast.success(messageType === 'IMAGE' ? 'Gửi ảnh thành công' : 'Gửi file thành công');
        }
      } else {
        toast.error('Upload file không thành công');
      }

      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (error: any) {
      console.error('Error uploading file:', error);
      toast.error(error.response?.data?.message || 'Không thể gửi file');
    } finally {
      setUploadingFile(false);
    }
  };

  const formatMessageTime = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleTimeString('vi-VN', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (!conversationId) {
    return (
      <div className="flex-1 flex items-center justify-center bg-gradient-to-br from-gray-50 to-blue-50">
        <div className="text-center text-gray-500 p-8">
          <div className="bg-white rounded-full p-8 shadow-lg mb-6 inline-block">
            <svg
              className="mx-auto h-20 w-20 text-blue-400"
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
          <p className="text-xl font-semibold text-gray-700 mb-2">CTU Connect Messenger</p>
          <p className="text-sm text-gray-500">Chọn một cuộc trò chuyện để bắt đầu nhắn tin</p>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col bg-white">
      {/* Chat Header */}
      <div className="border-b border-gray-200 p-4 bg-white shadow-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {/* Avatar */}
            <div className="relative">
              <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-400 to-indigo-500 flex items-center justify-center overflow-hidden">
                {conversationAvatar ? (
                  <img
                    src={conversationAvatar}
                    alt={conversationName || 'Chat'}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <span className="text-white font-semibold">
                    {conversationName?.charAt(0).toUpperCase() || 'C'}
                  </span>
                )}
              </div>
              {isOnline && (
                <div className="absolute bottom-0 right-0 w-3 h-3 bg-green-500 border-2 border-white rounded-full"></div>
              )}
            </div>
            {/* Name and status */}
            <div>
              <h3 className="font-semibold text-gray-900">{conversationName || 'Cuộc trò chuyện'}</h3>
              <p className="text-xs text-gray-500">
                {isOnline ? 'Đang hoạt động' : 'Không hoạt động'}
              </p>
            </div>
          </div>
          {/* Actions */}
          <div className="flex items-center space-x-2">
            <button className="p-2 text-blue-600 hover:bg-blue-50 rounded-full transition-colors">
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
              </svg>
            </button>
            <button className="p-2 text-blue-600 hover:bg-blue-50 rounded-full transition-colors">
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            </button>
            <button className="p-2 text-gray-600 hover:bg-gray-100 rounded-full transition-colors">
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => {
          const isOwn = message.senderId === currentUserId;
          const isNew = newMessageIds.has(message.id);

          return (
            <div
              key={message.id}
              className={`flex ${isOwn ? 'justify-end' : 'justify-start'} ${
                isNew ? 'animate-slideInUp' : ''
              }`}
              style={{
                animation: isNew ? 'slideInUp 0.3s ease-out' : 'none'
              }}
            >
              <div className={`flex items-end max-w-[70%] ${isOwn ? 'flex-row-reverse' : 'flex-row'}`}>
                {/* Avatar */}
                {!isOwn && (
                  <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center overflow-hidden mr-2 flex-shrink-0">
                    {message.senderAvatar ? (
                      <img
                        src={message.senderAvatar}
                        alt={message.senderName}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <span className="text-white text-sm">
                        {message.senderName.charAt(0).toUpperCase()}
                      </span>
                    )}
                  </div>
                )}

                {/* Message Bubble */}
                <div>
                  {!isOwn && (
                    <p className="text-xs text-gray-600 mb-1 ml-2">
                      {message.senderName}
                    </p>
                  )}
                  <div
                    className={`px-4 py-2 rounded-2xl shadow-sm transition-all ${
                      isOwn
                        ? 'bg-blue-600 text-white rounded-br-none'
                        : 'bg-gray-200 text-gray-900 rounded-bl-none'
                    } ${isNew ? 'scale-95 opacity-0' : 'scale-100 opacity-100'}`}
                    style={{
                      animation: isNew ? 'popIn 0.3s ease-out forwards' : 'none'
                    }}
                  >
                    {/* Reply preview */}
                    {message.replyToMessage && (
                      <div className={`mb-2 pb-2 border-l-2 pl-2 text-xs ${
                        isOwn ? 'border-blue-300' : 'border-gray-400'
                      }`}>
                        <p className="font-semibold">{message.replyToMessage.senderName}</p>
                        <p className="opacity-75">{message.replyToMessage.content}</p>
                      </div>
                    )}

                    {/* Image attachment */}
                    {message.type === 'IMAGE' && message.attachment && (
                      <img
                        src={message.attachment.fileUrl}
                        alt={message.attachment.fileName}
                        className="max-w-full rounded-lg mb-2 cursor-pointer"
                        onClick={() => window.open(message.attachment!.fileUrl, '_blank')}
                      />
                    )}

                    {/* File attachment */}
                    {message.type === 'FILE' && message.attachment && (
                      <a
                        href={message.attachment.fileUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center space-x-2 mb-2 p-2 bg-white bg-opacity-20 rounded"
                      >
                        <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
                          <path d="M8 4a3 3 0 00-3 3v4a5 5 0 0010 0V7a1 1 0 112 0v4a7 7 0 11-14 0V7a5 5 0 0110 0v4a3 3 0 11-6 0V7a1 1 0 012 0v4a1 1 0 102 0V7a3 3 0 00-3-3z" />
                        </svg>
                        <span className="text-sm">{message.attachment.fileName}</span>
                      </a>
                    )}

                    {/* Message content */}
                    <p className="break-words">{message.content}</p>

                    {/* Time and edited indicator */}
                    <p
                      className={`text-xs mt-1 ${
                        isOwn ? 'text-blue-200' : 'text-gray-500'
                      }`}
                    >
                      {formatMessageTime(message.createdAt)}
                      {message.isEdited && ' · Đã chỉnh sửa'}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
        
        {/* Typing Indicator */}
        {isTyping && (
          <div className="flex justify-start animate-slideInUp">
            <div className="flex items-end max-w-[70%]">
              <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center mr-2 flex-shrink-0">
                <svg className="h-4 w-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
              </div>
              <div className="px-4 py-3 rounded-2xl bg-gray-200 rounded-bl-none flex items-center space-x-1">
                <div className="w-2 h-2 bg-gray-500 rounded-full typing-dot"></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full typing-dot"></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full typing-dot"></div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-200 p-4">
        <form onSubmit={handleSendMessage} className="flex items-end space-x-2">
          {/* File upload button */}
          <input
            ref={fileInputRef}
            type="file"
            onChange={handleFileSelect}
            className="hidden"
            accept="image/*,.pdf,.doc,.docx,.txt"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploadingFile}
            className="p-2 text-blue-600 hover:bg-blue-50 rounded-full transition-colors disabled:opacity-50"
          >
            {uploadingFile ? (
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
            ) : (
              <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"
                />
              </svg>
            )}
          </button>

          {/* Message input */}
          <div className="flex-1 relative">
            <input
              type="text"
              value={messageInput}
              onChange={(e) => setMessageInput(e.target.value)}
              placeholder="Aa"
              className="w-full px-4 py-2 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Send button */}
          <button
            type="submit"
            disabled={!messageInput.trim() || sendingMessage}
            className="p-2 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {sendingMessage ? (
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
            ) : (
              <svg className="h-6 w-6" fill="currentColor" viewBox="0 0 20 20">
                <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
              </svg>
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
