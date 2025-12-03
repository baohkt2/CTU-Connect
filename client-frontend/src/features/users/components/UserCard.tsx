'use client';

import React from 'react';
import { User, FriendshipStatus } from '@/shared/types';
import Link from 'next/link';

interface UserCardProps {
  user: User;
  showActions?: boolean;
  compact?: boolean;
  friendshipStatus?: FriendshipStatus;
  onSendFriendRequest?: () => void;
  onAcceptFriendRequest?: () => void;
  onRejectFriendRequest?: () => void;
  onRemoveFriend?: () => void;
  onCancelRequest?: () => void;
}

export const UserCard: React.FC<UserCardProps> = ({
  user,
  showActions = true,
  compact = false,
  friendshipStatus,
  onSendFriendRequest,
  onAcceptFriendRequest,
  onRejectFriendRequest,
  onRemoveFriend,
  onCancelRequest
}) => {
  const renderActionButton = () => {
    if (!showActions) return null;

    switch (friendshipStatus) {
      case 'FRIENDS':
        return (
          <button
            onClick={onRemoveFriend}
            className="px-3 py-1 bg-red-500 text-white rounded text-sm hover:bg-red-600"
          >
            Unfriend
          </button>
        );
      case 'REQUEST_SENT':
        return (
          <button
            onClick={onCancelRequest}
            className="px-3 py-1 bg-gray-500 text-white rounded text-sm hover:bg-gray-600"
          >
            Cancel Request
          </button>
        );
      case 'REQUEST_RECEIVED':
        return (
          <div className="flex space-x-1">
            <button
              onClick={onAcceptFriendRequest}
              className="px-3 py-1 bg-green-500 text-white rounded text-sm hover:bg-green-600"
            >
              Accept
            </button>
            <button
              onClick={onRejectFriendRequest}
              className="px-3 py-1 bg-gray-500 text-white rounded text-sm hover:bg-gray-600"
            >
              Reject
            </button>
          </div>
        );
      case 'NOT_FRIENDS':
      default:
        return (
          <button
            onClick={onSendFriendRequest}
            className="px-3 py-1 bg-blue-500 text-white rounded text-sm hover:bg-blue-600"
          >
            Add Friend
          </button>
        );
    }
  };

  if (compact) {
    return (
      <div className="flex items-center space-x-3">
        <Link href={`/profile/${user.id}`}>
          <div className="w-10 h-10 rounded-full bg-gray-200 flex items-center justify-center overflow-hidden cursor-pointer">
            {user.avatar ? (
              <img
                src={user.avatar}
                alt={user.fullName}
                className="w-full h-full object-cover"
              />
            ) : (
              <span className="text-gray-500">
                {user.fullName.charAt(0).toUpperCase()}
              </span>
            )}
          </div>
        </Link>
        <div className="flex-1 min-w-0">
          <Link href={`/profile/${user.id}`}>
            <h4 className="font-semibold text-gray-900 hover:text-blue-600 cursor-pointer truncate">
              {user.fullName}
            </h4>
          </Link>
          <p className="text-sm text-gray-500 truncate">@{user.username}</p>
          {user.faculty && (
            <p className="text-xs text-gray-400 truncate">{user.faculty}</p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border p-4 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex flex-col items-center space-y-3">
        {/* Avatar */}
        <Link href={`/profile/${user.id}`}>
          <div className="w-16 h-16 rounded-full bg-gray-200 flex items-center justify-center overflow-hidden cursor-pointer hover:opacity-80">
            {user.avatar ? (
              <img
                src={user.avatar}
                alt={user.fullName}
                className="w-full h-full object-cover"
              />
            ) : (
              <span className="text-gray-500 text-xl">
                {user.fullName.charAt(0).toUpperCase()}
              </span>
            )}
          </div>
        </Link>

        {/* User Info */}
        <div className="text-center">
          <Link href={`/profile/${user.id}`}>
            <h4 className="font-semibold text-gray-900 hover:text-blue-600 cursor-pointer">
              {user.fullName}
            </h4>
          </Link>
          <p className="text-sm text-gray-500">@{user.username}</p>
          {user.faculty && (
            <p className="text-xs text-gray-400">{user.faculty}</p>
          )}
        </div>

        {/* Bio */}
        {user.bio && (
          <p className="text-xs text-gray-600 text-center line-clamp-2">
            {user.bio}
          </p>
        )}

        {/* Stats */}
        <div className="flex space-x-4 text-xs text-gray-500">
          {user.postsCount !== undefined && (
            <span>{user.postsCount} posts</span>
          )}
          {user.followersCount !== undefined && (
            <span>{user.followersCount} followers</span>
          )}
        </div>

        {/* Online Status */}
        {user.isOnline && (
          <div className="flex items-center space-x-1 text-xs text-green-600">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span>Online</span>
          </div>
        )}

        {/* Action Button */}
        {renderActionButton()}
      </div>
    </div>
  );
};
