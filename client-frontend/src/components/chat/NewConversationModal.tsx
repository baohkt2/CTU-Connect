import React, { useState, useEffect } from 'react';
import { useChat } from '../../contexts/ChatContext';
import { XMarkIcon, MagnifyingGlassIcon, UserIcon, UsersIcon } from '@heroicons/react/24/outline';

interface User {
  id: string;
  name: string;
  avatar: string;
  fullName: string;
}

interface NewConversationModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentUserId: string;
}

const NewConversationModal: React.FC<NewConversationModalProps> = ({
  isOpen,
  onClose,
  currentUserId
}) => {
  const { createConversation, setActiveConversation } = useChat();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedUsers, setSelectedUsers] = useState<User[]>([]);
  const [conversationType, setConversationType] = useState<'DIRECT' | 'GROUP'>('DIRECT');
  const [groupName, setGroupName] = useState('');
  const [users, setUsers] = useState<User[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isSearching, setIsSearching] = useState(false);

  // Mock users data - in real app, this would come from UserService
  useEffect(() => {
    if (isOpen) {
      // Simulate loading users
      setIsLoading(true);
      setTimeout(() => {
        setUsers([
          { id: 'user1', name: 'Nguyễn Văn A', avatar: '', fullName: 'Nguyễn Văn A' },
          { id: 'user2', name: 'Trần Thị B', avatar: '', fullName: 'Trần Thị B' },
          { id: 'user3', name: 'Lê Văn C', avatar: '', fullName: 'Lê Văn C' },
          { id: 'user4', name: 'Phạm Thị D', avatar: '', fullName: 'Phạm Thị D' },
          { id: 'user5', name: 'Hoàng Văn E', avatar: '', fullName: 'Hoàng Văn E' },
        ]);
        setIsLoading(false);
      }, 1000);
    }
  }, [isOpen]);

  // Filter users based on search query
  const filteredUsers = users.filter(user =>
    user.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    user.fullName.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleUserSelect = (user: User) => {
    if (conversationType === 'DIRECT') {
      setSelectedUsers([user]);
    } else {
      const isSelected = selectedUsers.some(u => u.id === user.id);
      if (isSelected) {
        setSelectedUsers(selectedUsers.filter(u => u.id !== user.id));
      } else {
        setSelectedUsers([...selectedUsers, user]);
      }
    }
  };

  const handleCreateConversation = async () => {
    if (selectedUsers.length === 0) return;

    try {
      setIsLoading(true);
      
      const participantIds = selectedUsers.map(user => user.id);
      const conversation = await createConversation(
        participantIds,
        conversationType === 'GROUP' ? groupName : undefined,
        conversationType
      );

      // Set as active conversation and close modal
      setActiveConversation(conversation.id);
      handleClose();
    } catch (error) {
      console.error('Failed to create conversation:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClose = () => {
    setSearchQuery('');
    setSelectedUsers([]);
    setConversationType('DIRECT');
    setGroupName('');
    onClose();
  };

  const canCreateConversation = () => {
    if (selectedUsers.length === 0) return false;
    if (conversationType === 'GROUP' && !groupName.trim()) return false;
    return true;
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md mx-4 max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <h2 className="text-lg font-semibold text-gray-900">Tạo cuộc trò chuyện mới</h2>
          <button
            onClick={handleClose}
            className="p-1 hover:bg-gray-100 rounded-full"
          >
            <XMarkIcon className="h-5 w-5" />
          </button>
        </div>

        {/* Conversation Type Selector */}
        <div className="p-4 border-b">
          <div className="flex space-x-1 bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => {
                setConversationType('DIRECT');
                setSelectedUsers([]);
              }}
              className={`flex-1 flex items-center justify-center space-x-2 py-2 px-3 rounded-md text-sm font-medium transition-colors ${
                conversationType === 'DIRECT'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <UserIcon className="h-4 w-4" />
              <span>Chat trực tiếp</span>
            </button>
            <button
              onClick={() => {
                setConversationType('GROUP');
                setSelectedUsers([]);
              }}
              className={`flex-1 flex items-center justify-center space-x-2 py-2 px-3 rounded-md text-sm font-medium transition-colors ${
                conversationType === 'GROUP'
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <UsersIcon className="h-4 w-4" />
              <span>Nhóm chat</span>
            </button>
          </div>
        </div>

        {/* Group Name Input (for group chats) */}
        {conversationType === 'GROUP' && (
          <div className="p-4 border-b">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Tên nhóm
            </label>
            <input
              type="text"
              value={groupName}
              onChange={(e) => setGroupName(e.target.value)}
              placeholder="Nhập tên nhóm..."
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              maxLength={100}
            />
          </div>
        )}

        {/* Selected Users */}
        {selectedUsers.length > 0 && (
          <div className="p-4 border-b">
            <h3 className="text-sm font-medium text-gray-700 mb-2">
              Đã chọn ({selectedUsers.length})
            </h3>
            <div className="flex flex-wrap gap-2">
              {selectedUsers.map(user => (
                <div
                  key={user.id}
                  className="flex items-center space-x-2 bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm"
                >
                  <span>{user.name}</span>
                  <button
                    onClick={() => handleUserSelect(user)}
                    className="hover:bg-blue-200 rounded-full p-0.5"
                  >
                    <XMarkIcon className="h-3 w-3" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Search */}
        <div className="p-4 border-b">
          <div className="relative">
            <MagnifyingGlassIcon className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
            <input
              type="text"
              placeholder="Tìm kiếm người dùng..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
        </div>

        {/* Users List */}
        <div className="flex-1 overflow-y-auto">
          {isLoading ? (
            <div className="p-8 text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
              <p className="mt-2 text-sm text-gray-500">Đang tải danh sách người dùng...</p>
            </div>
          ) : filteredUsers.length === 0 ? (
            <div className="p-8 text-center text-gray-500">
              <p className="text-sm">
                {searchQuery ? 'Không tìm thấy người dùng nào' : 'Không có người dùng nào'}
              </p>
            </div>
          ) : (
            <div className="divide-y divide-gray-200">
              {filteredUsers.map(user => {
                const isSelected = selectedUsers.some(u => u.id === user.id);
                const isDisabled = conversationType === 'DIRECT' && selectedUsers.length > 0 && !isSelected;
                
                return (
                  <div
                    key={user.id}
                    onClick={() => !isDisabled && handleUserSelect(user)}
                    className={`p-4 cursor-pointer transition-colors ${
                      isDisabled
                        ? 'opacity-50 cursor-not-allowed'
                        : isSelected
                        ? 'bg-blue-50'
                        : 'hover:bg-gray-50'
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      <div className="relative">
                        {user.avatar ? (
                          <img
                            src={user.avatar}
                            alt={user.name}
                            className="w-10 h-10 rounded-full object-cover"
                          />
                        ) : (
                          <div className="w-10 h-10 rounded-full bg-gray-300 flex items-center justify-center">
                            <span className="text-sm font-medium text-gray-600">
                              {user.name.charAt(0).toUpperCase()}
                            </span>
                          </div>
                        )}
                        {isSelected && (
                          <div className="absolute -top-1 -right-1 w-5 h-5 bg-blue-600 text-white rounded-full flex items-center justify-center">
                            <span className="text-xs">✓</span>
                          </div>
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-900 truncate">
                          {user.name}
                        </p>
                        <p className="text-sm text-gray-500 truncate">
                          {user.fullName}
                        </p>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t bg-gray-50">
          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-500">
              {conversationType === 'DIRECT' 
                ? 'Chọn 1 người để chat trực tiếp'
                : `Đã chọn ${selectedUsers.length} người`
              }
            </p>
            <div className="flex space-x-3">
              <button
                onClick={handleClose}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Hủy
              </button>
              <button
                onClick={handleCreateConversation}
                disabled={!canCreateConversation() || isLoading}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'Đang tạo...' : 'Tạo cuộc trò chuyện'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NewConversationModal;
