'use client';

import React, { useEffect, useState } from 'react';
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
      setError('Failed to load friends');
      toast.error('Failed to load friends');
      console.error('Error loading friends:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRemoveFriend = async (friendId: string) => {
    try {
      await userService.removeFriend(friendId);
      setFriends(prev => prev.filter(friend => friend.id !== friendId));
      toast.success('Friend removed successfully');
    } catch (err) {
      toast.error('Failed to remove friend');
      console.error('Error removing friend:', err);
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
          Try Again
        </button>
      </div>
    );
  }

  if (friends.length === 0) {
    return (
      <div className="text-center py-8">
        <p className="text-gray-500">No friends yet</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">
        Friends ({pagination.totalElements})
      </h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {friends.map((friend) => (
          <div key={friend.id} className="bg-white rounded-lg border p-4 shadow-sm">
            <div className="flex flex-col items-center space-y-3">
              <div className="w-16 h-16 rounded-full bg-gray-200 flex items-center justify-center overflow-hidden">
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

              <div className="text-center">
                <h4 className="font-semibold text-gray-900">{friend.fullName}</h4>
                <p className="text-sm text-gray-500">@{friend.username}</p>
              </div>

              {showActions && (
                <button
                  onClick={() => handleRemoveFriend(friend.id)}
                  className="px-3 py-1 bg-red-500 text-white rounded text-sm hover:bg-red-600"
                >
                  Unfriend
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
