'use client';

import React, { useEffect, useState } from 'react';
import { User } from '@/types';
import { userService } from '@/services/userService';
import { toast } from 'react-hot-toast';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import { UserPlus, X, Check } from 'lucide-react';

export const FriendRequestsList: React.FC = () => {
  const [requests, setRequests] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [processing, setProcessing] = useState<string | null>(null);

  useEffect(() => {
    loadFriendRequests();
  }, []);

  const loadFriendRequests = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await userService.getFriendRequests();
      setRequests(response);
    } catch (err) {
      setError('Failed to load friend requests');
      toast.error('Failed to load friend requests');
      console.error('Error loading friend requests:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleAcceptRequest = async (friendId: string) => {
    try {
      setProcessing(friendId);
      await userService.acceptFriendRequest(friendId);
      setRequests(prev => prev.filter(request => request.id !== friendId));
      toast.success('Friend request accepted');
    } catch (err) {
      toast.error('Failed to accept friend request');
      console.error('Error accepting friend request:', err);
    } finally {
      setProcessing(null);
    }
  };

  const handleRejectRequest = async (friendId: string) => {
    try {
      setProcessing(friendId);
      await userService.rejectFriendRequest(friendId);
      setRequests(prev => prev.filter(request => request.id !== friendId));
      toast.success('Friend request rejected');
    } catch (err) {
      toast.error('Failed to reject friend request');
      console.error('Error rejecting friend request:', err);
    } finally {
      setProcessing(null);
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
          onClick={loadFriendRequests}
          className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Try Again
        </button>
      </div>
    );
  }

  if (requests.length === 0) {
    return (
      <div className="text-center py-8">
        <UserPlus className="w-12 h-12 text-gray-300 mx-auto mb-3" />
        <p className="text-gray-500">No pending friend requests</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">
        Friend Requests ({requests.length})
      </h3>
      <div className="space-y-3">
        {requests.map((request) => (
          <div key={request.id} className="bg-white rounded-lg border p-4 shadow-sm">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-12 h-12 rounded-full bg-gray-200 flex items-center justify-center overflow-hidden">
                  {request.avatarUrl ? (
                    <img
                      src={request.avatarUrl}
                      alt={request.fullName}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <span className="text-gray-500 text-lg">
                      {request.fullName?.charAt(0).toUpperCase() || 'U'}
                    </span>
                  )}
                </div>

                <div>
                  <h4 className="font-semibold text-gray-900">{request.fullName}</h4>
                  <p className="text-sm text-gray-500">@{request.username}</p>
                  {request.major && (
                    <p className="text-xs text-gray-400 mt-0.5">{request.major.name}</p>
                  )}
                </div>
              </div>

              <div className="flex items-center space-x-2">
                <button
                  onClick={() => handleAcceptRequest(request.id)}
                  disabled={processing === request.id}
                  className="px-3 py-1.5 bg-blue-500 text-white rounded text-sm hover:bg-blue-600 disabled:opacity-50 flex items-center space-x-1"
                >
                  <Check className="w-4 h-4" />
                  <span>Accept</span>
                </button>
                <button
                  onClick={() => handleRejectRequest(request.id)}
                  disabled={processing === request.id}
                  className="px-3 py-1.5 bg-gray-200 text-gray-700 rounded text-sm hover:bg-gray-300 disabled:opacity-50 flex items-center space-x-1"
                >
                  <X className="w-4 h-4" />
                  <span>Reject</span>
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
