'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { User, GraduationCap, Briefcase, Users } from 'lucide-react';
import { User as UserType } from '@/types';
import { FriendButton, FriendshipStatus } from '@/components/ui/FriendButton';
import { userService } from '@/services/userService';
import { useAuth } from '@/contexts/AuthContext';

interface UserCardProps {
  user: UserType;
  showFriendButton?: boolean;
  showMutualFriends?: boolean;
  className?: string;
}

export const UserCard: React.FC<UserCardProps> = ({
  user,
  showFriendButton = true,
  showMutualFriends = true,
  className = ''
}) => {
  const { user: currentUser } = useAuth();
  const [friendshipStatus, setFriendshipStatus] = useState<FriendshipStatus>('none');
  const [mutualFriendsCount, setMutualFriendsCount] = useState<number>(0);
  const [loading, setLoading] = useState(false);

  const isOwnProfile = currentUser?.id === user.id;

  useEffect(() => {
    if (!isOwnProfile && showFriendButton && user.id) {
      loadFriendshipData();
    }
  }, [user.id, isOwnProfile, showFriendButton]);

  const loadFriendshipData = async () => {
    if (!user.id) return;
    
    try {
      setLoading(true);
      
      // Load friendship status
      const statusResponse = await userService.getFriendshipStatus(user.id);
      setFriendshipStatus(statusResponse.status);
      
      // Load mutual friends count if needed
      if (showMutualFriends) {
        const count = await userService.getMutualFriendsCount(user.id);
        setMutualFriendsCount(count);
      }
    } catch (error) {
      console.error('Error loading friendship data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFriendStatusChange = (newStatus: FriendshipStatus) => {
    setFriendshipStatus(newStatus);
  };

  const getRoleDisplay = (role: string) => {
    switch (role) {
      case 'STUDENT': return 'Sinh viên';
      case 'LECTURER': return 'Giảng viên';
      case 'ADMIN': return 'Quản trị viên';
      default: return 'Người dùng';
    }
  };

  return (
    <div className={`bg-white rounded-lg border border-gray-200 hover:shadow-md transition-shadow p-4 ${className}`}>
      <div className="flex items-start space-x-4">
        {/* Avatar */}
        <Link href={`/profile/${user.id}`}>
          <div className="w-16 h-16 rounded-full overflow-hidden bg-gray-200 flex-shrink-0 cursor-pointer hover:opacity-80 transition-opacity">
            {user.avatarUrl ? (
              <img
                src={user.avatarUrl}
                alt={user.fullName || user.name || 'User avatar'}
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full bg-gradient-to-br from-blue-400 to-purple-600 flex items-center justify-center text-white text-xl font-bold">
                {(user.fullName || user.name || 'U').charAt(0).toUpperCase()}
              </div>
            )}
          </div>
        </Link>

        {/* User Info */}
        <div className="flex-1 min-w-0">
          <Link href={`/profile/${user.id}`}>
            <h3 className="text-lg font-semibold text-gray-900 hover:text-blue-600 truncate cursor-pointer">
              {user.fullName || user.name || 'Người dùng'}
            </h3>
          </Link>

          <div className="flex items-center space-x-2 mt-1">
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
              user.role === 'LECTURER' 
                ? 'bg-blue-100 text-blue-700' 
                : 'bg-green-100 text-green-700'
            }`}>
              {getRoleDisplay(user.role)}
            </span>
            {user.isOnline && (
              <span className="flex items-center text-xs text-gray-500">
                <div className="w-2 h-2 bg-green-500 rounded-full mr-1"></div>
                Đang hoạt động
              </span>
            )}
          </div>

          {/* Additional Info */}
          <div className="mt-2 text-sm text-gray-600 space-y-1">
            {user.role === 'STUDENT' && user.major && (
              <div className="flex items-center space-x-1">
                <GraduationCap className="h-4 w-4" />
                <span className="truncate">{user.major.name}</span>
              </div>
            )}
            
            {user.role === 'LECTURER' && user.faculty && (
              <div className="flex items-center space-x-1">
                <Briefcase className="h-4 w-4" />
                <span className="truncate">{user.faculty.name}</span>
              </div>
            )}

            {user.role === 'STUDENT' && user.batch && (
              <div className="text-xs text-gray-500">
                Khóa {user.batch.year}
              </div>
            )}

            {showMutualFriends && mutualFriendsCount > 0 && !isOwnProfile && (
              <div className="flex items-center space-x-1 text-blue-600">
                <Users className="h-4 w-4" />
                <span>{mutualFriendsCount} bạn chung</span>
              </div>
            )}
          </div>

          {/* Bio */}
          {user.bio && (
            <p className="mt-2 text-sm text-gray-600 line-clamp-2">
              {user.bio}
            </p>
          )}
        </div>

        {/* Friend Button */}
        {showFriendButton && !isOwnProfile && user.id && (
          <div className="flex-shrink-0">
            <FriendButton
              targetUserId={user.id}
              initialStatus={friendshipStatus}
              onStatusChange={handleFriendStatusChange}
              size="sm"
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default UserCard;
