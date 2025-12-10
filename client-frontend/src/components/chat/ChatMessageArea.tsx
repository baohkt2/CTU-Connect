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
}

export default function ChatMessageArea({
  conversationId,
  currentUserId,
}: ChatMessageAreaProps) {
  const { user } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [messageInput, setMessageInput] = useState('');
  const [uploadingFile, setUploadingFile] = useState(false);
  const [sendingMessage, setSendingMessage] = useState(false); // Prevent double send
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (conversationId) {
      loadMessages();
    }
  }, [conversationId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const loadMessages = async () => {
    if (!conversationId) return;

    try {
      setLoading(true);
      const response = await api.get(
        `/chats/messages/conversation/${conversationId}`
      );
      setMessages(response.data.content?.reverse() || []);
    } catch (error) {
      console.error('Error loading messages:', error);
      // Don't show error for 404 (empty conversation)
      if (error.response?.status === 404) {
        setMessages([]); // Just set empty messages
      } else {
        toast.error('Không thể tải tin nhắn');
      }
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

    try {
      setSendingMessage(true); // Prevent double send
      const response = await api.post('/chats/messages', {
        conversationId,
        content: messageInput.trim(),
      });

      if (response.data) {
        setMessages((prev) => [...prev, response.data]);
        setMessageInput('');
      }
    } catch (error) {
      console.error('Error sending message:', error);
      toast.error('Không thể gửi tin nhắn');
    } finally {
      setSendingMessage(false);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !conversationId) return;

    try {
      setUploadingFile(true);
      console.log('Uploading file:', file.name, 'Type:', file.type);

      // Upload to media service
      const formData = new FormData();
      formData.append('file', file);
      formData.append('type', file.type.startsWith('image/') ? 'image' : 'document');

      const uploadResponse = await api.post('/media/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      console.log('Upload response:', uploadResponse.data);

      if (uploadResponse.data && uploadResponse.data.cloudinaryUrl) {
        // Send message with attachment
        const messagePayload = {
          conversationId,
          content: file.name,
          attachment: {
            fileName: uploadResponse.data.originalFileName || file.name,
            fileUrl: uploadResponse.data.cloudinaryUrl,
            fileType: uploadResponse.data.contentType || file.type,
            fileSize: uploadResponse.data.fileSize || file.size,
            thumbnailUrl: uploadResponse.data.cloudinaryUrl,
          },
        };

        console.log('Sending message with attachment:', messagePayload);

        const messageResponse = await api.post('/chats/messages', messagePayload);

        console.log('Message response:', messageResponse.data);

        if (messageResponse.data) {
          setMessages((prev) => [...prev, messageResponse.data]);
          toast.success('Gửi file thành công');
        }
      } else {
        console.error('Upload response invalid:', uploadResponse.data);
        toast.error('Upload file không trả về URL');
      }

      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (error: any) {
      console.error('Error uploading file:', error);
      console.error('Error response:', error.response?.data);
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
      <div className="flex-1 flex items-center justify-center bg-gray-50">
        <div className="text-center text-gray-500">
          <svg
            className="mx-auto h-16 w-16 mb-4 text-gray-300"
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
          <p className="text-lg">Chọn một cuộc trò chuyện để bắt đầu nhắn tin</p>
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
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => {
          const isOwn = message.senderId === currentUserId;

          return (
            <div
              key={message.id}
              className={`flex ${isOwn ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`flex items-end max-w-[70%] ${isOwn ? 'flex-row-reverse' : 'flex-row'}`}>
                {/* Avatar */}
                {!isOwn && (
                  <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center overflow-hidden mr-2">
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
                    className={`px-4 py-2 rounded-2xl ${
                      isOwn
                        ? 'bg-blue-600 text-white rounded-br-none'
                        : 'bg-gray-200 text-gray-900 rounded-bl-none'
                    }`}
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
