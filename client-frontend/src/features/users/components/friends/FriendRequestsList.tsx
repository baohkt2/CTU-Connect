'use client';

import React, { useEffect, useState } from 'react';
import { User } from '@/types';
import { userService } from '@/services/userService';
import { toast } from 'react-hot-toast';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import { UserPlus, X, Check, Send } from 'lucide-react';

interface FriendRequest extends User {
  requestType?: 'SENT' | 'RECEIVED';
}

export const FriendRequestsList: React.FC = () => {
  const [receivedRequests, setReceivedRequests] = useState<FriendRequest[]>([]);
  const [sentRequests, setSentRequests] = useState<FriendRequest[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [processing, setProcessing] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'received' | 'sent'>('received');

  useEffect(() => {
    loadAllFriendRequests();
  }, []);

  const loadAllFriendRequests = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Load both received and sent requests in parallel
      const [received, sent] = await Promise.all([
        userService.getFriendRequests(),      // RECEIVED
        userService.getSentFriendRequests()   // SENT
      ]);
      
      setReceivedRequests(received);
      setSentRequests(sent);
    } catch (err) {
      setError('Không thể tải lời mời kết bạn');
      toast.error('Không thể tải lời mời kết bạn');
      console.error('Error loading friend requests:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleAcceptRequest = async (friendId: string) => {
    try {
      setProcessing(friendId);
      await userService.acceptFriendRequest(friendId);
      setReceivedRequests(prev => prev.filter(request => request.id !== friendId));
      toast.success('Đã chấp nhận lời mời kết bạn');
    } catch (err) {
      toast.error('Không thể chấp nhận lời mời');
      console.error('Error accepting friend request:', err);
    } finally {
      setProcessing(null);
    }
  };

  const handleRejectRequest = async (friendId: string) => {
    try {
      setProcessing(friendId);
      await userService.rejectFriendRequest(friendId);
      setReceivedRequests(prev => prev.filter(request => request.id !== friendId));
      toast.success('Đã từ chối lời mời kết bạn');
    } catch (err) {
      toast.error('Không thể từ chối lời mời');
      console.error('Error rejecting friend request:', err);
    } finally {
      setProcessing(null);
    }
  };

  const handleCancelRequest = async (friendId: string) => {
    try {
      setProcessing(friendId);
      await userService.cancelFriendRequest(friendId);
      setSentRequests(prev => prev.filter(request => request.id !== friendId));
      toast.success('Đã hủy lời mời kết bạn');
    } catch (err) {
      toast.error('Không thể hủy lời mời');
      console.error('Error cancelling friend request:', err);
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
          onClick={loadAllFriendRequests}
          className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Thử lại
        </button>
      </div>
    );
  }

  const currentRequests = activeTab === 'received' ? receivedRequests : sentRequests;

  return (
    <div className="space-y-4">
      {/* Tabs */}
      <div className="flex space-x-2 border-b">
        <button
          onClick={() => setActiveTab('received')}
          className={`px-4 py-2 font-medium ${
            activeTab === 'received'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
        >
          Đã nhận ({receivedRequests.length})
        </button>
        <button
          onClick={() => setActiveTab('sent')}
          className={`px-4 py-2 font-medium ${
            activeTab === 'sent'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-500 hover:text-gray-700'
          }`}
        >
          Đã gửi ({sentRequests.length})
        </button>
      </div>

      {/* Content */}
      {currentRequests.length === 0 ? (
        <div className="text-center py-8">
          {activeTab === 'received' ? (
            <>
              <UserPlus className="w-12 h-12 text-gray-300 mx-auto mb-3" />
              <p className="text-gray-500">Không có lời mời kết bạn</p>
            </>
          ) : (
            <>
              <Send className="w-12 h-12 text-gray-300 mx-auto mb-3" />
              <p className="text-gray-500">Chưa gửi lời mời kết bạn nào</p>
            </>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          {currentRequests.map((request) => (
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
                  {activeTab === 'received' ? (
                    <>
                      <button
                        onClick={() => handleAcceptRequest(request.id)}
                        disabled={processing === request.id}
                        className="px-3 py-1.5 bg-blue-500 text-white rounded text-sm hover:bg-blue-600 disabled:opacity-50 flex items-center space-x-1"
                      >
                        <Check className="w-4 h-4" />
                        <span>Chấp nhận</span>
                      </button>
                      <button
                        onClick={() => handleRejectRequest(request.id)}
                        disabled={processing === request.id}
                        className="px-3 py-1.5 bg-gray-200 text-gray-700 rounded text-sm hover:bg-gray-300 disabled:opacity-50 flex items-center space-x-1"
                      >
                        <X className="w-4 h-4" />
                        <span>Từ chối</span>
                      </button>
                    </>
                  ) : (
                    <button
                      onClick={() => handleCancelRequest(request.id)}
                      disabled={processing === request.id}
                      className="px-3 py-1.5 bg-red-100 text-red-600 rounded text-sm hover:bg-red-200 disabled:opacity-50 flex items-center space-x-1"
                    >
                      <X className="w-4 h-4" />
                      <span>Hủy</span>
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
