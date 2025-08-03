'use client';

import React from 'react';
import { useAuth } from '@/contexts/AuthContext';
import {
  UserPlusIcon,
  CalendarDaysIcon,
  GiftIcon,
  MegaphoneIcon,
  AcademicCapIcon,
  ChevronRightIcon
} from '@heroicons/react/24/outline';

const RightSidebar: React.FC = () => {
  const { user } = useAuth();

  const friendSuggestions = [
    {
      id: '1',
      name: 'Nguyễn Văn Nam',
      mutualFriends: 5,
      avatar: '👨‍🎓',
      faculty: 'Công nghệ thông tin'
    },
    {
      id: '2',
      name: 'Trần Thị Lan',
      mutualFriends: 3,
      avatar: '👩‍🎓',
      faculty: 'Kinh tế'
    },
    {
      id: '3',
      name: 'Lê Hoàng Minh',
      mutualFriends: 8,
      avatar: '👨‍💻',
      faculty: 'Công nghệ thông tin'
    }
  ];

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
    <div className="hidden xl:block w-80 h-screen sticky top-16 overflow-y-auto custom-scrollbar bg-white">
      <div className="p-4 space-y-6">

        {/* Friend Suggestions */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-gray-900 vietnamese-text">Gợi ý kết bạn</h3>
            <button className="text-blue-600 hover:text-blue-700 text-sm font-medium">
              Xem tất cả
            </button>
          </div>
          <div className="space-y-3">
            {friendSuggestions.map((friend) => (
              <div key={friend.id} className="flex items-center space-x-3">
                <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-400 to-purple-500 flex items-center justify-center text-xl">
                  {friend.avatar}
                </div>
                <div className="flex-1">
                  <p className="font-medium text-gray-900 text-sm vietnamese-text">{friend.name}</p>
                  <p className="text-xs text-gray-500">{friend.mutualFriends} bạn chung • {friend.faculty}</p>
                </div>
                <button className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded-lg text-sm font-medium transition-colors">
                  Kết bạn
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Upcoming Events */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-gray-900 vietnamese-text flex items-center">
              <CalendarDaysIcon className="h-5 w-5 mr-2 text-green-600" />
              Sự kiện sắp tới
            </h3>
            <ChevronRightIcon className="h-4 w-4 text-gray-400" />
          </div>
          <div className="space-y-3">
            {upcomingEvents.map((event) => (
              <div key={event.id} className="p-3 bg-green-50 rounded-lg hover:bg-green-100 transition-colors cursor-pointer">
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <p className="font-medium text-gray-900 text-sm vietnamese-text">{event.title}</p>
                    <p className="text-xs text-gray-600 mt-1">
                      {event.date} • {event.time}
                    </p>
                    <p className="text-xs text-gray-500">{event.location}</p>
                  </div>
                  <button className="text-green-600 hover:text-green-700 text-xs font-medium">
                    Quan tâm
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Recent Notifications */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-gray-900 vietnamese-text flex items-center">
              <MegaphoneIcon className="h-5 w-5 mr-2 text-orange-600" />
              Thông báo mới
            </h3>
            <button className="text-blue-600 hover:text-blue-700 text-sm font-medium">
              Xem tất cả
            </button>
          </div>
          <div className="space-y-3">
            {notifications.map((notification) => (
              <div key={notification.id} className="flex items-start space-x-3 p-2 hover:bg-gray-50 rounded-lg transition-colors cursor-pointer">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white text-sm ${
                  notification.type === 'academic' ? 'bg-blue-500' :
                  notification.type === 'exam' ? 'bg-red-500' : 'bg-green-500'
                }`}>
                  <AcademicCapIcon className="h-4 w-4" />
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-900 vietnamese-text">{notification.title}</p>
                  <p className="text-xs text-gray-500">{notification.time}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Trending Topics */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-gray-900 vietnamese-text">Xu hướng tại CTU</h3>
            <ChevronRightIcon className="h-4 w-4 text-gray-400" />
          </div>
          <div className="space-y-2">
            {trending.map((item, index) => (
              <div key={index} className="flex items-center justify-between p-2 hover:bg-gray-50 rounded-lg transition-colors cursor-pointer">
                <div>
                  <p className="text-sm font-medium text-blue-600">{item.tag}</p>
                  <p className="text-xs text-gray-500">{item.posts} bài viết</p>
                </div>
                <div className="text-xs text-gray-400">#{index + 1}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Birthday Reminders */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-gray-900 vietnamese-text flex items-center">
              <GiftIcon className="h-5 w-5 mr-2 text-pink-600" />
              Sinh nhật hôm nay
            </h3>
          </div>
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-pink-400 to-red-500 flex items-center justify-center text-white">
              🎂
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium text-gray-900 vietnamese-text">Hôm nay là sinh nhật của <strong>Phạm Thị Mai</strong></p>
              <button className="text-xs text-blue-600 hover:text-blue-700 font-medium mt-1">
                Gửi lời chúc mừng
              </button>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-xs text-gray-500 space-y-2 pb-4">
          <div className="flex flex-wrap gap-2">
            <a href="#" className="hover:text-gray-700">Quyền riêng tư</a>
            <span>•</span>
            <a href="#" className="hover:text-gray-700">Điều khoản</a>
            <span>•</span>
            <a href="#" className="hover:text-gray-700">Trợ giúp</a>
          </div>
          <p className="vietnamese-text">CTU Connect © 2025</p>
          <p className="vietnamese-text">Được phát triển bởi sinh viên CTU</p>
        </div>
      </div>
    </div>
  );
};

export default RightSidebar;
