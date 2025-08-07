import React, { useState, useRef, useEffect } from 'react';
import {
  PaperAirplaneIcon,
  PhotoIcon,
  PaperClipIcon,
  FaceSmileIcon,
  MicrophoneIcon
} from '@heroicons/react/24/outline';

interface MessageInputProps {
  onSendMessage: (content: string) => Promise<void>;
  onTyping: (isTyping: boolean) => void;
  disabled?: boolean;
  placeholder?: string;
}

const MessageInput: React.FC<MessageInputProps> = ({
  onSendMessage,
  onTyping,
  disabled = false,
  placeholder = "Nhập tin nhắn..."
}) => {
  const [message, setMessage] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [showEmojiPicker, setShowEmojiPicker] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  }, [message]);

  // Handle typing indicator
  useEffect(() => {
    if (message.trim()) {
      onTyping(true);

      // Clear existing timeout
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }

      // Set new timeout to stop typing indicator
      typingTimeoutRef.current = setTimeout(() => {
        onTyping(false);
      }, 1000);
    } else {
      onTyping(false);
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
    }

    return () => {
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
    };
  }, [message, onTyping]);

  const handleSendMessage = async () => {
    const trimmedMessage = message.trim();
    if (!trimmedMessage || disabled) return;

    try {
      await onSendMessage(trimmedMessage);
      setMessage('');
      onTyping(false);

      // Focus back to textarea
      textareaRef.current?.focus();
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleEmojiSelect = (emoji: string) => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const newMessage = message.slice(0, start) + emoji + message.slice(end);

    setMessage(newMessage);
    setShowEmojiPicker(false);

    // Set cursor position after emoji
    setTimeout(() => {
      textarea.focus();
      textarea.setSelectionRange(start + emoji.length, start + emoji.length);
    }, 0);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    // TODO: Implement file upload logic
    console.log('Files selected:', files);

    // Reset input
    event.target.value = '';
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    // TODO: Implement image upload logic
    console.log('Images selected:', files);

    // Reset input
    event.target.value = '';
  };

  const popularEmojis = ['😀', '😂', '❤️', '👍', '👎', '😮', '😢', '😡', '🎉', '🔥'];

  return (
    <div className="p-4 border-t bg-white">
      {/* Emoji Picker */}
      {showEmojiPicker && (
        <div className="mb-4 p-3 bg-gray-50 rounded-lg">
          <div className="grid grid-cols-10 gap-2">
            {popularEmojis.map((emoji) => (
              <button
                key={emoji}
                onClick={() => handleEmojiSelect(emoji)}
                className="p-2 hover:bg-gray-200 rounded text-lg transition-colors"
              >
                {emoji}
              </button>
            ))}
          </div>
          <div className="mt-2 text-xs text-gray-500 text-center">
            Nhấp để thêm emoji vào tin nhắn
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="flex items-end space-x-2">
        {/* Action Buttons - Left Side */}
        <div className="flex items-center space-x-1">
          {/* Image Upload */}
          <button
            onClick={() => imageInputRef.current?.click()}
            disabled={disabled}
            className="p-2 text-gray-500 hover:text-blue-600 hover:bg-blue-50 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Gửi hình ảnh"
          >
            <PhotoIcon className="h-5 w-5" />
          </button>

          {/* File Upload */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled}
            className="p-2 text-gray-500 hover:text-blue-600 hover:bg-blue-50 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Gửi file"
          >
            <PaperClipIcon className="h-5 w-5" />
          </button>

          {/* Emoji Picker Toggle */}
          <button
            onClick={() => setShowEmojiPicker(!showEmojiPicker)}
            disabled={disabled}
            className={`p-2 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
              showEmojiPicker 
                ? 'text-blue-600 bg-blue-50' 
                : 'text-gray-500 hover:text-blue-600 hover:bg-blue-50'
            }`}
            title="Emoji"
          >
            <FaceSmileIcon className="h-5 w-5" />
          </button>
        </div>

        {/* Message Input */}
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={placeholder}
            disabled={disabled}
            rows={1}
            className="w-full px-4 py-2 border border-gray-300 rounded-2xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none max-h-32 disabled:opacity-50 disabled:cursor-not-allowed"
            style={{ minHeight: '40px' }}
          />

          {/* Character Limit Indicator */}
          {message.length > 1800 && (
            <div className="absolute -top-6 right-2 text-xs text-gray-500">
              {message.length}/2000
            </div>
          )}
        </div>

        {/* Send Button or Voice Recording */}
        <div className="flex items-center">
          {message.trim() ? (
            <button
              onClick={handleSendMessage}
              disabled={disabled || !message.trim()}
              className="p-2 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Gửi tin nhắn"
            >
              <PaperAirplaneIcon className="h-5 w-5" />
            </button>
          ) : (
            <button
              onMouseDown={() => setIsRecording(true)}
              onMouseUp={() => setIsRecording(false)}
              onMouseLeave={() => setIsRecording(false)}
              disabled={disabled}
              className={`p-2 rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                isRecording 
                  ? 'bg-red-600 text-white' 
                  : 'text-gray-500 hover:text-red-600 hover:bg-red-50'
              }`}
              title="Ghi âm tin nhắn"
            >
              <MicrophoneIcon className="h-5 w-5" />
            </button>
          )}
        </div>
      </div>

      {/* Voice Recording Indicator */}
      {isRecording && (
        <div className="mt-2 flex items-center justify-center space-x-2 text-red-600">
          <div className="w-2 h-2 bg-red-600 rounded-full animate-pulse"></div>
          <span className="text-sm">Đang ghi âm... Thả để gửi</span>
        </div>
      )}

      {/* Hidden File Inputs */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        onChange={handleFileUpload}
        className="hidden"
        accept=".pdf,.doc,.docx,.txt,.zip,.rar"
      />

      <input
        ref={imageInputRef}
        type="file"
        multiple
        onChange={handleImageUpload}
        className="hidden"
        accept="image/*"
      />
    </div>
  );
};

export default MessageInput;
