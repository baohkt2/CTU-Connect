'use client';

import React, { useState, useEffect } from 'react';
import { UserPlus, UserMinus, UserCheck, Clock, X } from 'lucide-react';
import { userService } from '@/services/userService';
import { toast } from 'react-hot-toast';
import { useAuth } from '@/contexts/AuthContext';

export type FriendshipStatus = 
  | 'none'          // Not friends, no pending request
  | 'friends'       // Already friends
  | 'sent'          // Current user sent a friend request
  | 'received'      // Current user received a friend request
  | 'self';         // Viewing own profile

interface FriendButtonProps {
  targetUserId: string;
  initialStatus?: FriendshipStatus;
  onStatusChange?: (newStatus: FriendshipStatus) => void;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}

export const FriendButton: React.FC<FriendButtonProps> = ({
  targetUserId,
  initialStatus,
  onStatusChange,
  className = '',
  size = 'md'
}) => {
  const { user } = useAuth();
  const [status, setStatus] = useState<FriendshipStatus>(initialStatus || 'none');
  const [loading, setLoading] = useState(false);

  // Don't show button for own profile
  if (user?.id === targetUserId) {
    return null;
  }

  useEffect(() => {
    if (initialStatus) {
      setStatus(initialStatus);
    }
  }, [initialStatus]);

  const handleStatusChange = (newStatus: FriendshipStatus) => {
    setStatus(newStatus);
    if (onStatusChange) {
      onStatusChange(newStatus);
    }
  };

  const handleSendRequest = async () => {
    try {
      setLoading(true);
      await userService.sendFriendRequest(targetUserId);
      handleStatusChange('sent');
      toast.success('Đã gửi lời mời kết bạn');
    } catch (error) {
      console.error('Error sending friend request:', error);
      toast.error('Không thể gửi lời mời kết bạn');
    } finally {
      setLoading(false);
    }
  };

  const handleCancelRequest = async () => {
    try {
      setLoading(true);
      await userService.rejectFriendRequest(targetUserId);
      handleStatusChange('none');
      toast.success('Đã hủy lời mời kết bạn');
    } catch (error) {
      console.error('Error canceling friend request:', error);
      toast.error('Không thể hủy lời mời');
    } finally {
      setLoading(false);
    }
  };

  const handleAcceptRequest = async () => {
    try {
      setLoading(true);
      await userService.acceptFriendRequest(targetUserId);
      handleStatusChange('friends');
      toast.success('Đã chấp nhận lời mời kết bạn');
    } catch (error) {
      console.error('Error accepting friend request:', error);
      toast.error('Không thể chấp nhận lời mời');
    } finally {
      setLoading(false);
    }
  };

  const handleRejectRequest = async () => {
    try {
      setLoading(true);
      await userService.rejectFriendRequest(targetUserId);
      handleStatusChange('none');
      toast.success('Đã từ chối lời mời kết bạn');
    } catch (error) {
      console.error('Error rejecting friend request:', error);
      toast.error('Không thể từ chối lời mời');
    } finally {
      setLoading(false);
    }
  };

  const handleUnfriend = async () => {
    if (!confirm('Bạn có chắc chắn muốn hủy kết bạn?')) {
      return;
    }
    
    try {
      setLoading(true);
      await userService.removeFriend(targetUserId);
      handleStatusChange('none');
      toast.success('Đã hủy kết bạn');
    } catch (error) {
      console.error('Error unfriending:', error);
      toast.error('Không thể hủy kết bạn');
    } finally {
      setLoading(false);
    }
  };

  const sizeClasses = {
    sm: 'px-3 py-1.5 text-xs',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-2.5 text-base'
  };

  const baseClasses = `
    ${sizeClasses[size]}
    font-medium rounded-lg transition-all duration-200
    flex items-center justify-center space-x-2
    disabled:opacity-50 disabled:cursor-not-allowed
    ${className}
  `;

  if (status === 'friends') {
    return (
      <button
        onClick={handleUnfriend}
        disabled={loading}
        className={`${baseClasses} bg-gray-100 text-gray-700 hover:bg-red-50 hover:text-red-600 border border-gray-300`}
      >
        <UserCheck className="w-4 h-4" />
        <span>Bạn bè</span>
      </button>
    );
  }

  if (status === 'sent') {
    return (
      <button
        onClick={handleCancelRequest}
        disabled={loading}
        className={`${baseClasses} bg-gray-100 text-gray-700 hover:bg-red-50 hover:text-red-600 border border-gray-300`}
      >
        <Clock className="w-4 h-4" />
        <span>Hủy lời mời</span>
      </button>
    );
  }

  if (status === 'received') {
    return (
      <div className="flex space-x-2">
        <button
          onClick={handleAcceptRequest}
          disabled={loading}
          className={`${baseClasses} bg-blue-500 text-white hover:bg-blue-600`}
        >
          <UserCheck className="w-4 h-4" />
          <span>Chấp nhận</span>
        </button>
        <button
          onClick={handleRejectRequest}
          disabled={loading}
          className={`${baseClasses} bg-gray-100 text-gray-700 hover:bg-red-50 hover:text-red-600 border border-gray-300`}
        >
          <X className="w-4 h-4" />
          <span>Từ chối</span>
        </button>
      </div>
    );
  }

  // status === 'none'
  return (
    <button
      onClick={handleSendRequest}
      disabled={loading}
      className={`${baseClasses} bg-blue-500 text-white hover:bg-blue-600`}
    >
      <UserPlus className="w-4 h-4" />
      <span>Kết bạn</span>
    </button>
  );
};

export default FriendButton;
