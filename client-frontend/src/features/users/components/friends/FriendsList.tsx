'use client';

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { User, PaginatedResponse } from '@/types';
import { userService } from '@/services/userService';
import { toast } from 'react-hot-toast';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import { MoreVertical, MessageCircle, UserMinus } from 'lucide-react';

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
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
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
      <h3 className="text-base sm:text-lg font-semibold px-2 sm:px-0">
        Bạn bè ({pagination.totalElements})
      </h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3 sm:gap-4">
        {friends.map((friend) => (
          <div 
            key={friend.id} 
            className="bg-white rounded-lg border shadow-sm hover:shadow-md transition-shadow relative"
          >
            {/* Menu ba chấm */}
            {showActions && (
              <div className="absolute top-2 right-2 z-10">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setOpenMenuId(openMenuId === friend.id ? null : friend.id);
                  }}
                  className="p-1.5 hover:bg-gray-100 rounded-full transition-colors"
                >
                  <MoreVertical className="w-4 h-4 text-gray-600" />
                </button>
                
                {/* Dropdown menu */}
                {openMenuId === friend.id && (
                  <>
                    <div 
                      className="fixed inset-0 z-10" 
                      onClick={() => setOpenMenuId(null)}
                    />
                    <div className="absolute right-0 mt-1 w-48 bg-white rounded-lg shadow-lg border py-1 z-20">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleRemoveFriend(friend.id);
                          setOpenMenuId(null);
                        }}
                        className="w-full px-4 py-2 text-left text-sm text-red-600 hover:bg-red-50 flex items-center gap-2"
                      >
                        <UserMinus className="w-4 h-4" />
                        Xóa kết bạn
                      </button>
                    </div>
                  </>
                )}
              </div>
            )}

            {/* Card content - clickable */}
            <div 
              onClick={() => handleViewProfile(friend.id)}
              className="p-3 sm:p-4 cursor-pointer"
            >
              <div className="flex flex-col items-center space-y-2 sm:space-y-3">
                {/* Avatar */}
                <div className="w-14 h-14 sm:w-16 sm:h-16 rounded-full bg-gray-200 flex items-center justify-center overflow-hidden hover:ring-2 hover:ring-blue-500 transition-all">
                  {friend.avatarUrl ? (
                    <img
                      src={friend.avatarUrl}
                      alt={friend.fullName}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <span className="text-gray-500 text-lg sm:text-xl">
                      {friend.fullName?.charAt(0).toUpperCase() || 'U'}
                    </span>
                  )}
                </div>

                {/* Name */}
                <div className="text-center w-full">
                  <h4 className="font-semibold text-sm sm:text-base text-gray-900 hover:text-blue-600 transition-colors truncate px-6">
                    {friend.fullName}
                  </h4>
                  <p className="text-xs sm:text-sm text-gray-500 truncate">@{friend.username}</p>
                </div>
              </div>
            </div>

            {/* Action button */}
            {showActions && (
              <div className="px-3 pb-3 sm:px-4 sm:pb-4 pt-0">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleChatWithFriend(friend.id);
                  }}
                  className="w-full px-3 py-2 bg-blue-500 text-white rounded-lg text-xs sm:text-sm hover:bg-blue-600 transition-colors flex items-center justify-center gap-2"
                >
                  <MessageCircle className="h-3.5 w-3.5 sm:h-4 sm:w-4" />
                  <span>Nhắn tin</span>
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
