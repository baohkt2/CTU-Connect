import React, { useState, useEffect } from 'react';
import { useChat } from '../../contexts/ChatContext';
import ConversationList from './ConversationList';
import MessageArea from './MessageArea';
import UserPresenceBar from './UserPresenceBar';
import NewConversationModal from './NewConversationModal';
import { PlusIcon, XMarkIcon } from '@heroicons/react/24/outline';

interface ChatWindowProps {
  isOpen: boolean;
  onClose: () => void;
  currentUserId: string;
}

const ChatWindow: React.FC<ChatWindowProps> = ({ isOpen, onClose, currentUserId }) => {
  const { state, connectToChat, disconnectFromChat } = useChat();
  const [showNewConversationModal, setShowNewConversationModal] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [showConversationList, setShowConversationList] = useState(true);

  // Check if mobile
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };
    
    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  // Connect to chat when component mounts
  useEffect(() => {
    if (isOpen && currentUserId && !state.isConnected) {
      connectToChat(currentUserId);
    }
    
    return () => {
      if (!isOpen) {
        disconnectFromChat();
      }
    };
  }, [isOpen, currentUserId, state.isConnected, connectToChat, disconnectFromChat]);

  // On mobile, hide conversation list when a conversation is selected
  useEffect(() => {
    if (isMobile && state.activeConversationId) {
      setShowConversationList(false);
    }
  }, [isMobile, state.activeConversationId]);

  const handleBackToConversations = () => {
    setShowConversationList(true);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
      <div className="bg-white rounded-lg shadow-xl w-full h-full md:w-4/5 md:h-4/5 max-w-6xl flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b bg-blue-600 text-white rounded-t-lg">
          <h2 className="text-lg font-semibold">Chat</h2>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowNewConversationModal(true)}
              className="p-2 hover:bg-blue-700 rounded-full transition-colors"
              title="Tạo cuộc trò chuyện mới"
            >
              <PlusIcon className="h-5 w-5" />
            </button>
            <button
              onClick={onClose}
              className="p-2 hover:bg-blue-700 rounded-full transition-colors"
            >
              <XMarkIcon className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex flex-1 overflow-hidden">
          {/* Conversation List */}
          <div className={`${
            isMobile 
              ? (showConversationList ? 'w-full' : 'hidden') 
              : 'w-1/3 border-r'
          } flex flex-col`}>
            <ConversationList 
              isMobile={isMobile}
              onConversationSelect={() => isMobile && setShowConversationList(false)}
            />
          </div>

          {/* Message Area */}
          <div className={`${
            isMobile 
              ? (showConversationList ? 'hidden' : 'w-full') 
              : 'flex-1'
          } flex flex-col`}>
            {state.activeConversationId ? (
              <MessageArea 
                conversationId={state.activeConversationId}
                isMobile={isMobile}
                onBackClick={isMobile ? handleBackToConversations : undefined}
              />
            ) : (
              <div className="flex-1 flex items-center justify-center text-gray-500">
                <div className="text-center">
                  <div className="mb-4">
                    <svg className="w-16 h-16 mx-auto text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Chọn một cuộc trò chuyện</h3>
                  <p className="text-sm text-gray-500">Chọn một cuộc trò chuyện từ danh sách bên trái để bắt đầu nhắn tin</p>
                </div>
              </div>
            )}
          </div>

          {/* User Presence Sidebar (Desktop only) */}
          {!isMobile && (
            <div className="w-64 border-l bg-gray-50">
              <UserPresenceBar />
            </div>
          )}
        </div>

        {/* Loading Overlay */}
        {state.isLoading && (
          <div className="absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center">
            <div className="flex items-center space-x-2">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
              <span className="text-gray-600">Đang kết nối...</span>
            </div>
          </div>
        )}

        {/* Error Message */}
        {state.error && (
          <div className="absolute top-16 left-4 right-4 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            <span className="block sm:inline">{state.error}</span>
          </div>
        )}

        {/* Connection Status */}
        {!state.isConnected && !state.isLoading && (
          <div className="absolute bottom-4 left-4 bg-yellow-100 border border-yellow-400 text-yellow-700 px-3 py-2 rounded-lg text-sm">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
              <span>Đang kết nối lại...</span>
            </div>
          </div>
        )}

        {state.isConnected && (
          <div className="absolute bottom-4 left-4 bg-green-100 border border-green-400 text-green-700 px-3 py-2 rounded-lg text-sm">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span>Đã kết nối</span>
            </div>
          </div>
        )}
      </div>

      {/* New Conversation Modal */}
      {showNewConversationModal && (
        <NewConversationModal
          isOpen={showNewConversationModal}
          onClose={() => setShowNewConversationModal(false)}
          currentUserId={currentUserId}
        />
      )}
    </div>
  );
};

export default ChatWindow;
