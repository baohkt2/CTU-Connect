'use client';

import React, { useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';
import Layout from '@/components/layout/Layout';
import { FriendRequestsList } from '@/features/users/components/friends/FriendRequestsList';
import { FriendsList } from '@/features/users/components/friends/FriendsList';
import { FriendSuggestions } from '@/features/users/components/friends/FriendSuggestions';
import { Users, UserPlus, UserCheck } from 'lucide-react';

type TabType = 'friends' | 'requests' | 'suggestions';

export default function FriendsPage() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<TabType>('friends');

  useEffect(() => {
    if (!loading && !user) {
      router.push('/login');
    }
  }, [user, loading, router]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Đang tải...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <Layout>
      <div className="min-h-screen bg-gray-50">
        <div className="max-w-6xl mx-auto px-4 py-8">
          {/* Header */}
          <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
            <h1 className="text-2xl font-bold text-gray-900 mb-2">Bạn bè</h1>
            <p className="text-gray-600">Quản lý bạn bè và lời mời kết bạn của bạn</p>
          </div>

          {/* Tabs */}
          <div className="bg-white rounded-lg shadow-sm mb-6">
            <div className="border-b border-gray-200">
              <nav className="flex -mb-px">
                <button
                  onClick={() => setActiveTab('friends')}
                  className={`
                    flex items-center space-x-2 px-6 py-4 border-b-2 font-medium text-sm
                    ${activeTab === 'friends'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }
                  `}
                >
                  <Users className="w-5 h-5" />
                  <span>Bạn bè</span>
                </button>
                <button
                  onClick={() => setActiveTab('requests')}
                  className={`
                    flex items-center space-x-2 px-6 py-4 border-b-2 font-medium text-sm
                    ${activeTab === 'requests'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }
                  `}
                >
                  <UserCheck className="w-5 h-5" />
                  <span>Lời mời kết bạn</span>
                </button>
                <button
                  onClick={() => setActiveTab('suggestions')}
                  className={`
                    flex items-center space-x-2 px-6 py-4 border-b-2 font-medium text-sm
                    ${activeTab === 'suggestions'
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }
                  `}
                >
                  <UserPlus className="w-5 h-5" />
                  <span>Gợi ý kết bạn</span>
                </button>
              </nav>
            </div>
          </div>

          {/* Content */}
          <div className="bg-white rounded-lg shadow-sm p-6">
            {activeTab === 'friends' && <FriendsList />}
            {activeTab === 'requests' && <FriendRequestsList />}
            {activeTab === 'suggestions' && <FriendSuggestions />}
          </div>
        </div>
      </div>
    </Layout>
  );
}
