'use client';

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { User, PaginatedResponse } from '@/types';
import { userService } from '@/services/userService';
import { toast } from 'react-hot-toast';
import LoadingSpinner from '@/components/ui/LoadingSpinner';

interface FriendsListProps {
  userId?: string;
  showActions?: boolean;
}

export const FriendsList: React.FC<FriendsListProps> = ({
  userId,
  showActions = true
}) => {
  const router = useRouter();
  const [friends, setFriends] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pagination, setPagination] = useState({
    page: 0,
    totalPages: 0,
    totalElements: 0
  });

  useEffect(() => {
    loadFriends();
  }, [userId]);

  const loadFriends = async () => {
    try {
      setLoading(true);
      setError(null);

      // Sử dụng service có sẵn - trả về PaginatedResponse<User>
      const response: PaginatedResponse<User> = userId
        ? await userService.getFriends(userId, 0, 20)
        : await userService.getMyFriends();

      setFriends(response.content);
      setPagination({
        page: response.number,
        totalPages: response.totalPages,
        totalElements: response.totalElements
      });
    } catch (err) {
      setError('Không thể tải danh sách bạn bè');
      toast.error('Không thể tải danh sách bạn bè');
      console.error('Error loading friends:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRemoveFriend = async (friendId: string) => {
    try {
      await userService.removeFriend(friendId);
      setFriends(prev => prev.filter(friend => friend.id !== friendId));
      toast.success('Đã hủy kết bạn');
    } catch (err) {
      toast.error('Không thể hủy kết bạn');
      console.error('Error removing friend:', err);
    }
  };

  const handleViewProfile = (friendId: string) => {
    router.push(`/profile/${friendId}`);
  };
  
  const handleChatWithFriend = async (friendId: string) => {
    try {
      // Navigate to messages page with friend ID
      router.push(`/messages?userId=${friendId}`);
    } catch (err) {
      toast.error('Không thể mở chat');
      console.error('Error opening chat:', err);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center py-8">
        <LoadingSpinner />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <p className="text-red-500">{error}</p>
        <button
          onClick={loadFriends}
          className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Thử lại
        </button>
      </div>
    );
  }

  if (friends.length === 0) {
    return (
      <div className="text-center py-8">
        <p className="text-gray-500">Chưa có bạn bè</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">
        Bạn bè ({pagination.totalElements})
      </h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {friends.map((friend) => (
          <div 
            key={friend.id} 
            className="bg-white rounded-lg border p-4 shadow-sm hover:shadow-md transition-shadow"
          >
            <div className="flex flex-col items-center space-y-3">
              {/* Avatar - Clickable */}
              <div 
                onClick={() => handleViewProfile(friend.id)}
                className="w-16 h-16 rounded-full bg-gray-200 flex items-center justify-center overflow-hidden cursor-pointer hover:ring-2 hover:ring-blue-500 transition-all"
              >
                {friend.avatarUrl ? (
                  <img
                    src={friend.avatarUrl}
                    alt={friend.fullName}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <span className="text-gray-500 text-xl">
                    {friend.fullName?.charAt(0).toUpperCase() || 'U'}
                  </span>
                )}
              </div>

              {/* Name - Clickable */}
              <div className="text-center">
                <h4 
                  onClick={() => handleViewProfile(friend.id)}
                  className="font-semibold text-gray-900 cursor-pointer hover:text-blue-600 transition-colors"
                >
                  {friend.fullName}
                </h4>
                <p className="text-sm text-gray-500">@{friend.username}</p>
              </div>

              {showActions && (
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => handleViewProfile(friend.id)}
                    className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600 transition-colors"
                  >
                    Xem hồ sơ
                  </button>
                  <button
                    onClick={() => handleChatWithFriend(friend.id)}
                    className="px-3 py-1 bg-green-500 text-white rounded text-sm hover:bg-green-600 transition-colors flex items-center gap-1"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                    </svg>
                    Nhắn tin
                  </button>
                  <button
                    onClick={() => handleRemoveFriend(friend.id)}
                    className="px-3 py-1 bg-red-500 text-white rounded text-sm hover:bg-red-600 transition-colors"
                  >
                    Xóa kết bạn
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
