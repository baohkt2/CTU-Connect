'use client';

import React, { useEffect, useState } from 'react';
import Image from 'next/image';
import { useAuth } from '@/contexts/AuthContext';
import { userService } from '@/services/userService';
import { User } from '@/types';
import {
  UserPlusIcon,
  CalendarDaysIcon,
  ChevronRightIcon,
  CheckIcon
} from '@heroicons/react/24/outline';

interface FriendSuggestion {
  id: string;
  name: string;
  mutualFriends: number;
  avatar: string;
  faculty: string;
  userId: string;
}

const RightSidebar: React.FC = () => {
  const { user, isAuthenticated } = useAuth();
  const [friendSuggestions, setFriendSuggestions] = useState<FriendSuggestion[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sentRequests, setSentRequests] = useState<Set<string>>(new Set());

  // Fetch friend suggestions from API
  useEffect(() => {
    const fetchFriendSuggestions = async () => {
      if (!isAuthenticated || !user?.id) return;
      
      setIsLoading(true);
      try {
        const suggestions = await userService.getFriendSuggestions(5);
        const mappedSuggestions: FriendSuggestion[] = suggestions.map((u: User & { mutualFriendsCount?: number }) => ({
          id: u.id,
          userId: u.id,
          name: u.fullName || u.name || u.username || 'Người dùng',
          mutualFriends: u.mutualFriendsCount || 0,
          avatar: u.avatarUrl || getDefaultAvatar(u.gender),
          faculty: u.faculty || u.major || 'Chưa cập nhật'
        }));
        setFriendSuggestions(mappedSuggestions);
      } catch (error) {
        console.error('Failed to fetch friend suggestions:', error);
        // Keep empty state on error
        setFriendSuggestions([]);
      } finally {
        setIsLoading(false);
      }
    };

    fetchFriendSuggestions();
  }, [isAuthenticated, user?.id]);

  // Get default avatar based on gender
  const getDefaultAvatar = (gender?: string): string => {
    if (gender === 'FEMALE' || gender === 'female') return '👩‍🎓';
    if (gender === 'MALE' || gender === 'male') return '👨‍🎓';
    return '🧑‍🎓';
  };

  // Handle send friend request
  const handleAddFriend = async (userId: string) => {
    try {
      await userService.sendFriendRequest(userId);
      setSentRequests(prev => new Set(prev).add(userId));
    } catch (error) {
      console.error('Failed to send friend request:', error);
    }
  };

  const upcomingEvents = [
    {
      id: '1',
      title: 'Hội thảo Khoa học Công nghệ',
      date: '15 Th8',
      time: '14:00',
      location: 'Hội trường A'
    },
    {
      id: '2',
      title: 'Ngày hội Việc làm 2025',
      date: '20 Th8',
      time: '08:00',
      location: 'Sân vận động CTU'
    }
  ];

  const notifications = [
    {
      id: '1',
      title: 'Thông báo học phí',
      time: '2 giờ trước',
      type: 'academic'
    },
    {
      id: '2',
      title: 'Lịch thi cuối kỳ',
      time: '5 giờ trước',
      type: 'exam'
    },
    {
      id: '3',
      title: 'Hoạt động sinh viên',
      time: '1 ngày trước',
      type: 'activity'
    }
  ];

  const trending = [
    { tag: '#CTUConnect', posts: 245 },
    { tag: '#SinhVienCTU', posts: 189 },
    { tag: '#HocTap', posts: 156 },
    { tag: '#CongNghe', posts: 134 },
    { tag: '#TuyenDung', posts: 98 }
  ];

  return (
    <div className="hidden xl:block w-64 2xl:w-80 h-screen sticky top-16 overflow-y-auto custom-scrollbar bg-white border-l border-gray-100">
      <div className="p-3 2xl:p-4 space-y-4 2xl:space-y-6">

        {/* Friend Suggestions */}
        <div className="bg-white rounded-lg border border-gray-200 p-3 2xl:p-4">
          <div className="flex items-center justify-between mb-3 2xl:mb-4">
            <h3 className="font-semibold text-gray-900 vietnamese-text text-sm 2xl:text-base">Gợi ý kết bạn</h3>
            <a href="/friends" className="text-blue-600 hover:text-blue-700 text-xs 2xl:text-sm font-medium">
              Xem tất cả
            </a>
          </div>
          <div className="space-y-2 2xl:space-y-3">
            {isLoading ? (
              // Loading skeleton
              Array.from({ length: 3 }).map((_, index) => (
                <div key={index} className="flex items-center space-x-2 2xl:space-x-3 animate-pulse">
                  <div className="w-8 h-8 2xl:w-10 2xl:h-10 rounded-full bg-gray-200"></div>
                  <div className="flex-1 space-y-1">
                    <div className="h-3 bg-gray-200 rounded w-3/4"></div>
                    <div className="h-2 bg-gray-200 rounded w-1/2"></div>
                  </div>
                </div>
              ))
            ) : friendSuggestions.length === 0 ? (
              <p className="text-xs text-gray-500 text-center py-2">Không có gợi ý kết bạn</p>
            ) : (
              friendSuggestions.map((friend) => (
                <div key={friend.id} className="flex items-center space-x-2 2xl:space-x-3">
                  {friend.avatar.startsWith('http') ? (
                    <Image 
                      src={friend.avatar} 
                      alt={friend.name}
                      width={40}
                      height={40}
                      className="w-8 h-8 2xl:w-10 2xl:h-10 rounded-full object-cover"
                    />
                  ) : (
                    <div className="w-8 h-8 2xl:w-10 2xl:h-10 rounded-full bg-gray-100 flex items-center justify-center text-lg 2xl:text-xl">
                      {friend.avatar}
                    </div>
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-gray-900 vietnamese-text text-xs 2xl:text-sm truncate">
                      {friend.name}
                    </p>
                    <p className="text-xs 2xl:text-xs text-gray-500 truncate">
                      {friend.mutualFriends > 0 ? `${friend.mutualFriends} bạn chung • ` : ''}{friend.faculty}
                    </p>
                  </div>
                  {sentRequests.has(friend.userId) ? (
                    <span className="text-green-600 text-xs font-medium bg-green-50 px-2 py-1 rounded flex items-center gap-1">
                      <CheckIcon className="h-3 w-3" />
                      Đã gửi
                    </span>
                  ) : (
                    <button 
                      onClick={() => handleAddFriend(friend.userId)}
                      className="text-blue-600 hover:text-blue-700 text-xs font-medium bg-blue-50 hover:bg-blue-100 px-2 py-1 rounded transition-colors"
                      title="Kết bạn"
                    >
                      <UserPlusIcon className="h-3 w-3 2xl:h-4 2xl:w-4" />
                    </button>
                  )}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Upcoming Events */}
        <div className="bg-white rounded-lg border border-gray-200 p-3 2xl:p-4">
          <div className="flex items-center justify-between mb-3 2xl:mb-4">
            <h3 className="font-semibold text-gray-900 vietnamese-text text-sm 2xl:text-base">Sự kiện sắp tới</h3>
            <ChevronRightIcon className="h-4 w-4 2xl:h-5 2xl:w-5 text-gray-400" />
          </div>
          <div className="space-y-2 2xl:space-y-3">
            {upcomingEvents.map((event) => (
              <div key={event.id} className="flex items-start space-x-2 2xl:space-x-3 p-2 2xl:p-3 rounded-lg hover:bg-gray-50 transition-colors cursor-pointer">
                <div className="w-8 h-8 2xl:w-10 2xl:h-10 rounded-lg bg-indigo-100 flex items-center justify-center flex-shrink-0">
                  <CalendarDaysIcon className="h-4 w-4 2xl:h-5 2xl:w-5 text-indigo-600" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-gray-900 vietnamese-text text-xs 2xl:text-sm truncate">
                    {event.title}
                  </p>
                  <p className="text-xs text-gray-500 truncate">
                    {event.date} • {event.time}
                  </p>
                  <p className="text-xs text-gray-500 truncate">
                    📍 {event.location}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Quick Notifications */}
        <div className="bg-white rounded-lg border border-gray-200 p-3 2xl:p-4">
          <div className="flex items-center justify-between mb-3 2xl:mb-4">
            <h3 className="font-semibold text-gray-900 vietnamese-text text-sm 2xl:text-base">Thông báo nhanh</h3>
            <button className="text-blue-600 hover:text-blue-700 text-xs 2xl:text-sm font-medium">
              Xem tất cả
            </button>
          </div>
          <div className="space-y-2">
            {notifications.map((notification) => (
              <div key={notification.id} className="flex items-start space-x-2 2xl:space-x-3 p-2 rounded-lg hover:bg-gray-50 transition-colors cursor-pointer">
                <div className="w-2 h-2 rounded-full bg-blue-500 mt-2 flex-shrink-0"></div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-gray-900 vietnamese-text text-xs 2xl:text-sm truncate">
                    {notification.title}
                  </p>
                  <p className="text-xs text-gray-500">{notification.time}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Trending Tags */}
        <div className="bg-white rounded-lg border border-gray-200 p-3 2xl:p-4">
          <h3 className="font-semibold text-gray-900 vietnamese-text mb-3 2xl:mb-4 text-sm 2xl:text-base">Trending tại CTU</h3>
          <div className="space-y-2">
            {trending.map((trend, index) => (
              <div key={trend.tag} className="flex items-center justify-between p-2 rounded-lg hover:bg-gray-50 transition-colors cursor-pointer">
                <div className="flex items-center space-x-2">
                  <span className="text-xs 2xl:text-sm text-gray-500 font-medium">#{index + 1}</span>
                  <span className="font-medium text-blue-600 text-xs 2xl:text-sm">{trend.tag}</span>
                </div>
                <span className="text-xs text-gray-500">{trend.posts} bài viết</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RightSidebar;
