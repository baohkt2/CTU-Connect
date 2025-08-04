'use client';

import React from 'react';
import Link from 'next/link';
import { useAuth } from '@/contexts/AuthContext';
import {
  HomeIcon,
  UserIcon,
  UserGroupIcon,
  BookmarkIcon,
  CalendarIcon,
  ClockIcon,
  TagIcon,
  AcademicCapIcon,
  BuildingLibraryIcon,
  ChevronRightIcon
} from '@heroicons/react/24/outline';
import Avatar from "@/components/ui/Avatar";

const LeftSidebar: React.FC = () => {
  const { user } = useAuth();

  const menuItems = [
    {
      name: 'Trang cá nhân',
      href: '/profile/me',
      icon: UserIcon,
      color: 'text-blue-600'
    },
    {
      name: 'Bạn bè',
      href: '/friends',
      icon: UserGroupIcon,
      color: 'text-green-600'
    },
    {
      name: 'Đã lưu',
      href: '/saved',
      icon: BookmarkIcon,
      color: 'text-purple-600'
    },
    {
      name: 'Sự kiện',
      href: '/events',
      icon: CalendarIcon,
      color: 'text-red-600'
    },
    {
      name: 'Kỷ niệm',
      href: '/memories',
      icon: ClockIcon,
      color: 'text-yellow-600'
    },
    {
      name: 'CTU Groups',
      href: '/groups',
      icon: AcademicCapIcon,
      color: 'text-indigo-600'
    },
    {
      name: 'Thư viện',
      href: '/library',
      icon: BuildingLibraryIcon,
      color: 'text-gray-600'
    }
  ];

  const shortcuts = [
    { name: 'Nhóm Công nghệ thông tin', href: '/groups/cntt', avatar: '💻' },
    { name: 'Sinh viên K47', href: '/groups/k47', avatar: '🎓' },
    { name: 'CLB Lập trình CTU', href: '/groups/programming', avatar: '⚡' },
    { name: 'Học bổng & Tuyển dụng', href: '/groups/scholarship', avatar: '💼' }
  ];

  return (
    <div className="hidden lg:block w-80 h-screen sticky top-16 overflow-y-auto custom-scrollbar bg-white">
      <div className="p-4 space-y-6">
        {/* User Profile Section */}
        <div className="flex items-center space-x-3 p-3 rounded-lg hover:bg-gray-50 transition-colors cursor-pointer">
          <Avatar
              src={user?.avatarUrl || '/default-avatar.png'}
              alt={user?.fullName || user?.username || 'Avatar'}
              size="md"
              className="ring-2 ring-white shadow-sm hover:ring-indigo-200 transition-all duration-200 cursor-pointer"
          />
          <div className="flex-1">
            <p className="font-medium text-gray-900 vietnamese-text">
              {user?.fullName || user?.name || 'Người dùng'}
            </p>
            <p className="text-sm text-gray-500">Xem trang cá nhân</p>
          </div>
        </div>

        <hr className="border-gray-200" />

        {/* Main Menu */}
        <div className="space-y-2">
          {menuItems.map((item) => {
            const IconComponent = item.icon;
            return (
              <Link
                key={item.name}
                href={item.href}
                className="flex items-center space-x-3 p-3 rounded-lg hover:bg-gray-50 transition-colors group"
              >
                <div className={`w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center group-hover:bg-gray-200 transition-colors`}>
                  <IconComponent className={`h-5 w-5 ${item.color}`} />
                </div>
                <span className="font-medium text-gray-700 vietnamese-text">{item.name}</span>
              </Link>
            );
          })}
        </div>

        <hr className="border-gray-200" />

        {/* Shortcuts */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-gray-600 vietnamese-text">Lối tắt của bạn</h3>
            <button className="text-blue-600 hover:text-blue-700 text-sm font-medium">
              Chỉnh sửa
            </button>
          </div>
          <div className="space-y-2">
            {shortcuts.map((shortcut) => (
              <Link
                key={shortcut.name}
                href={shortcut.href}
                className="flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <div className="w-8 h-8 rounded-lg bg-gray-100 flex items-center justify-center text-lg">
                  {shortcut.avatar}
                </div>
                <span className="text-sm font-medium text-gray-700 vietnamese-text">{shortcut.name}</span>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default LeftSidebar;
