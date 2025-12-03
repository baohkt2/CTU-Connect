import React, { useState, useEffect } from 'react';
import { useChat } from '@/contexts/ChatContext';
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
  const {
    conversations,
    activeConversationId,
    messages,
    isConnected,
    isLoading,
    error,
    connectToChat,
    disconnectFromChat,
    setActiveConversation,
    clearError
  } = useChat();

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
    if (isOpen && currentUserId && !isConnected && !isLoading) {
      connectToChat();
    }

    return () => {
      if (!isOpen) {
        disconnectFromChat();
      }
    };
  }, [isOpen, currentUserId, isConnected, isLoading, connectToChat, disconnectFromChat]);

  // On mobile, hide conversation list when a conversation is selected
  useEffect(() => {
    if (isMobile && activeConversationId) {
      setShowConversationList(false);
    }
  }, [isMobile, activeConversationId]);

  // Clear error after 5 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => {
        clearError();
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [error, clearError]);

  if (!isOpen) {
    return null;
  }

  const handleBackToConversations = () => {
    setShowConversationList(true);
    setActiveConversation(null);
  };

  const currentConversation = conversations.find(conv => conv.id === activeConversationId);
  const currentMessages = activeConversationId ? messages[activeConversationId] || [] : [];

  return (
    <div className="fixed inset-0 bg-white z-50 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-white">
        <div className="flex items-center space-x-3">
          <h1 className="text-xl font-semibold text-gray-900">Messages</h1>
          {!isConnected && (
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
              Disconnected
            </span>
          )}
          {isConnected && (
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
              Connected
            </span>
          )}
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowNewConversationModal(true)}
            className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full"
            title="New conversation"
          >
            <PlusIcon className="h-5 w-5" />
          </button>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-full"
          >
            <XMarkIcon className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border-l-4 border-red-400 p-4">
          <div className="flex">
            <div className="ml-3">
              <p className="text-sm text-red-700">{error}</p>
            </div>
            <div className="ml-auto pl-3">
              <button
                onClick={clearError}
                className="text-red-400 hover:text-red-600"
              >
                <XMarkIcon className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Loading State */}
      {isLoading && (
        <div className="flex items-center justify-center p-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
          <span className="ml-2 text-gray-600">Loading...</span>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Desktop Layout or Mobile Conversation List */}
        {(!isMobile || showConversationList) && (
          <div className={`${isMobile ? 'w-full' : 'w-1/3 border-r border-gray-200'} flex flex-col`}>
            <ConversationList
              conversations={conversations}
              activeConversationId={activeConversationId}
              onConversationSelect={(conversationId) => {
                setActiveConversation(conversationId);
                if (isMobile) {
                  setShowConversationList(false);
                }
              }}
              currentUserId={currentUserId}
            />
          </div>
        )}

        {/* Desktop Message Area or Mobile Message View */}
        {(!isMobile || !showConversationList) && (
          <div className={`${isMobile ? 'w-full' : 'flex-1'} flex flex-col`}>
            {activeConversationId && currentConversation ? (
              <MessageArea
                conversation={currentConversation}
                messages={currentMessages}
                currentUserId={currentUserId}
                onBack={isMobile ? handleBackToConversations : undefined}
              />
            ) : (
              <div className="flex-1 flex items-center justify-center bg-gray-50">
                <div className="text-center">
                  <div className="text-gray-400 text-lg mb-2">💬</div>
                  <p className="text-gray-500">Select a conversation to start messaging</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* User Presence Bar */}
      <UserPresenceBar />

      {/* New Conversation Modal */}
      <NewConversationModal
        isOpen={showNewConversationModal}
        onClose={() => setShowNewConversationModal(false)}
        currentUserId={currentUserId}
      />
    </div>
  );
};

export default ChatWindow;
