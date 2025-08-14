'use client';

import React, { useEffect, useState } from 'react';
import { User, PaginatedResponse } from '@/types';
import { userService } from '@/services/userService';
import LoadingSpinner from '@/components/ui/LoadingSpinner';

interface MutualFriendsProps {
  otherUserId: string;
  otherUserName: string;
}

export const MutualFriends: React.FC<MutualFriendsProps> = ({
  otherUserId,
  otherUserName
}) => {
  const [mutualFriends, setMutualFriends] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadMutualFriends();
  }, [otherUserId]);

  const loadMutualFriends = async () => {
    try {
      setLoading(true);
      setError(null);
      // Sử dụng PaginatedResponse<User> thay vì FriendsDTO
      const mutualData: PaginatedResponse<User> = await userService.getMutualFriends(otherUserId);
      setMutualFriends(mutualData.content || []);
    } catch (err) {
      setError('Failed to load mutual friends');
      console.error('Error loading mutual friends:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center py-4">
        <LoadingSpinner />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-4">
        <p className="text-red-500 text-sm">{error}</p>
      </div>
    );
  }

  if (mutualFriends.length === 0) {
    return (
      <div className="text-center py-4">
        <p className="text-gray-500 text-sm">
          No mutual friends with {otherUserName}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <h4 className="text-sm font-semibold text-gray-700">
        Mutual Friends ({mutualFriends.length})
      </h4>

      {/* Compact list view for mutual friends */}
      <div className="space-y-2">
        {mutualFriends.slice(0, 5).map((friend) => (
          <div key={friend.id} className="flex items-center space-x-2 p-2 bg-gray-50 rounded">
            <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center overflow-hidden flex-shrink-0">
              {friend.avatarUrl ? (
                <img
                  src={friend.avatarUrl}
                  alt={friend.fullName}
                  className="w-full h-full object-cover"
                />
              ) : (
                <span className="text-gray-500 text-xs">
                  {friend.fullName?.charAt(0).toUpperCase() || 'U'}
                </span>
              )}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900 truncate">
                {friend.fullName}
              </p>
              <p className="text-xs text-gray-500 truncate">
                @{friend.username}
              </p>
            </div>
          </div>
        ))}

        {mutualFriends.length > 5 && (
          <div className="text-center">
            <button className="text-blue-500 text-sm hover:text-blue-600">
              View all {mutualFriends.length} mutual friends
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
