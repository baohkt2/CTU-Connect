'use client';

import React, { useState } from 'react';
import { User } from '@/types';
import { Users, UserCheck, UserPlus, Search, X } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Modal } from '@/components/ui/Modal';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import Avatar from '@/components/ui/Avatar';

interface ConnectionsModalProps {
  isOpen: boolean;
  onClose: () => void;
  user: User;
  type: 'followers' | 'following' | 'friends';
  connections: User[];
  isLoading: boolean;
  onLoadMore?: () => void;
  hasMore?: boolean;
}

export const ConnectionsModal: React.FC<ConnectionsModalProps> = ({
  isOpen,
  onClose,
  user,
  type,
  connections,
  isLoading,
  onLoadMore,
  hasMore = false
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [followingStates, setFollowingStates] = useState<{[key: string]: boolean}>({});

  const getTitle = () => {
    switch (type) {
      case 'followers': return `Người theo dõi ${user.fullName || user.name}`;
      case 'following': return `${user.fullName || user.name} đang theo dõi`;
      case 'friends': return `Bạn bè của ${user.fullName || user.name}`;
      default: return 'Kết nối';
    }
  };

  const filteredConnections = connections.filter(connection =>
    (connection.fullName || connection.name || '').toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleFollow = async (targetUserId: string) => {
    try {
      // TODO: Implement follow/unfollow logic
      setFollowingStates(prev => ({
        ...prev,
        [targetUserId]: !prev[targetUserId]
      }));
    } catch (error) {
      console.error('Error updating follow status:', error);
    }
  };

  const handleMessage = (targetUserId: string) => {
    // TODO: Implement messaging functionality
    console.log('Open chat with user:', targetUserId);
    onClose();
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={getTitle()}>
      <div className="space-y-4">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <Input
            type="text"
            placeholder="Tìm kiếm..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>

        {/* Connections List */}
        <div className="max-h-96 overflow-y-auto space-y-2">
          {isLoading && connections.length === 0 ? (
            <div className="flex justify-center py-8">
              <LoadingSpinner size="md" />
            </div>
          ) : filteredConnections.length === 0 ? (
            <div className="text-center py-8">
              <Users className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500 vietnamese-text">
                {searchQuery ? 'Không tìm thấy kết quả' : 'Chưa có kết nối nào'}
              </p>
            </div>
          ) : (
            filteredConnections.map((connection) => (
              <div
                key={connection.id}
                className="flex items-center justify-between p-3 hover:bg-gray-50 rounded-lg transition-colors"
              >
                <div className="flex items-center space-x-3">
                  <Avatar
                    id={connection.id}
                    src={connection.avatarUrl || ''}
                    alt={connection.fullName || connection.name || 'User'}
                    size="md"
                    className="ring-2 ring-white shadow-sm"
                  />
                  <div>
                    <h3 className="font-medium text-gray-900 vietnamese-text">
                      {connection.fullName || connection.name || 'Người dùng'}
                    </h3>
                    <div className="flex items-center space-x-2">
                      <span className={`text-xs px-2 py-0.5 rounded-full ${
                        connection.role === 'LECTURER' 
                          ? 'bg-blue-100 text-blue-700' 
                          : 'bg-green-100 text-green-700'
                      }`}>
                        {connection.role === 'LECTURER' ? 'Giảng viên' : 'Sinh viên'}
                      </span>
                      {connection.isOnline && (
                        <div className="flex items-center text-xs text-gray-500">
                          <div className="w-2 h-2 bg-green-500 rounded-full mr-1"></div>
                          Đang hoạt động
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleMessage(connection.id)}
                    className="text-xs"
                  >
                    Nhắn tin
                  </Button>

                  {type !== 'friends' && (
                    <Button
                      size="sm"
                      variant={followingStates[connection.id] ? "outline" : "primary"}
                      onClick={() => handleFollow(connection.id)}
                      className="text-xs flex items-center space-x-1"
                    >
                      {followingStates[connection.id] ? (
                        <>
                          <UserCheck className="h-3 w-3" />
                          <span>Đang theo dõi</span>
                        </>
                      ) : (
                        <>
                          <UserPlus className="h-3 w-3" />
                          <span>Theo dõi</span>
                        </>
                      )}
                    </Button>
                  )}
                </div>
              </div>
            ))
          )}
        </div>

        {/* Load More */}
        {hasMore && (
          <div className="text-center pt-4 border-t">
            <Button
              variant="outline"
              onClick={onLoadMore}
              disabled={isLoading}
              className="w-full"
            >
              {isLoading ? (
                <>
                  <LoadingSpinner size="sm" className="mr-2" />
                  Đang tải...
                </>
              ) : (
                'Xem thêm'
              )}
            </Button>
          </div>
        )}

        {/* Close Button */}
        <div className="flex justify-end pt-4 border-t">
          <Button variant="outline" onClick={onClose}>
            Đóng
          </Button>
        </div>
      </div>
    </Modal>
  );
};
