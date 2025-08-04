'use client';

import React, { useState } from 'react';
import { User } from '@/types';
import { Camera, MapPin, Calendar, Briefcase, GraduationCap, Edit3, UserPlus, MessageCircle, MoreHorizontal } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { formatTimeAgo } from '@/utils/localization';

interface ProfileHeaderProps {
  user: User;
  isOwnProfile: boolean;
  isFollowing?: boolean;
  onFollow?: () => void;
  onMessage?: () => void;
  onEditProfile?: () => void;
  onEditCover?: () => void;
  onEditAvatar?: () => void;
}

export const ProfileHeader: React.FC<ProfileHeaderProps> = ({
  user,
  isOwnProfile,
  isFollowing = false,
  onFollow,
  onMessage,
  onEditProfile,
  onEditCover,
  onEditAvatar
}) => {
  const [showFullBio, setShowFullBio] = useState(false);

  const getRoleDisplay = (role: string) => {
    switch (role) {
      case 'STUDENT': return 'Sinh viên';
      case 'LECTURER': return 'Giảng viên';
      case 'ADMIN': return 'Quản trị viên';
      default: return 'Người dùng';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm overflow-hidden">
      {/* Cover Photo */}
      <div className="relative h-80 bg-gradient-to-r from-blue-500 to-purple-600">
        {user.backgroundUrl ? (
          <img
            src={user.backgroundUrl}
            alt="Ảnh bìa"
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500" />
        )}

        {/* Cover Photo Edit Button */}
        {isOwnProfile && (
          <button
            onClick={onEditCover}
            className="absolute bottom-4 right-4 bg-white bg-opacity-90 hover:bg-opacity-100 rounded-lg px-3 py-2 text-gray-700 font-medium transition-all duration-200 flex items-center space-x-2"
          >
            <Camera className="h-4 w-4" />
            <span className="text-sm">Chỉnh sửa ảnh bìa</span>
          </button>
        )}
      </div>

      {/* Profile Info Section */}
      <div className="px-6 pb-6">
        <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between -mt-20 relative">
          {/* Avatar and Basic Info */}
          <div className="flex flex-col sm:flex-row sm:items-end sm:space-x-5">
            {/* Avatar */}
            <div className="relative">
              <div className="w-40 h-40 rounded-full border-4 border-white shadow-xl overflow-hidden bg-gray-200">
                {user.avatarUrl ? (
                  <img
                    src={user.avatarUrl}
                    alt={user.fullName || user.name || 'Avatar'}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full bg-gradient-to-br from-blue-400 to-purple-600 flex items-center justify-center text-white text-4xl font-bold">
                    {(user.fullName || user.name || 'U').charAt(0).toUpperCase()}
                  </div>
                )}
              </div>

              {/* Avatar Edit Button */}
              {isOwnProfile && (
                <button
                  onClick={onEditAvatar}
                  className="absolute bottom-2 right-2 bg-gray-100 hover:bg-gray-200 rounded-full p-2 shadow-lg transition-colors"
                >
                  <Camera className="h-4 w-4 text-gray-600" />
                </button>
              )}
            </div>

            {/* Name and Title */}
            <div className="mt-4 sm:mt-0 flex-1">
              <div className="flex items-center space-x-2">
                <h1 className="text-3xl font-bold text-gray-900 vietnamese-text">
                  {user.fullName || user.name || 'Người dùng'}
                </h1>
                {user.isVerified && (
                  <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center">
                    <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  </div>
                )}
              </div>

              <div className="flex items-center space-x-2 mt-1">
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  user.role === 'LECTURER' 
                    ? 'bg-blue-100 text-blue-700' 
                    : 'bg-green-100 text-green-700'
                }`}>
                  {getRoleDisplay(user.role)}
                </span>
                {user.isOnline && (
                  <span className="flex items-center text-sm text-gray-500">
                    <div className="w-2 h-2 bg-green-500 rounded-full mr-1"></div>
                    Đang hoạt động
                  </span>
                )}
              </div>

              {/* Quick Info */}
              <div className="flex flex-wrap items-center mt-3 text-sm text-gray-600 space-x-4">
                {user.role === 'STUDENT' && user.major && (
                  <div className="flex items-center space-x-1">
                    <GraduationCap className="h-4 w-4" />
                    <span>{user.major.name}</span>
                  </div>
                )}

                {user.role === 'LECTURER' && user.faculty && (
                  <div className="flex items-center space-x-1">
                    <Briefcase className="h-4 w-4" />
                    <span>{user.faculty.name}</span>
                  </div>
                )}

                {user.createdAt && (
                  <div className="flex items-center space-x-1">
                    <Calendar className="h-4 w-4" />
                    <span>Tham gia {formatTimeAgo(user.createdAt)}</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center space-x-3 mt-4 lg:mt-0">
            {isOwnProfile ? (
              <>
                <Button
                  onClick={onEditProfile}
                  variant="outline"
                  className="flex items-center space-x-2"
                >
                  <Edit3 className="h-4 w-4" />
                  <span>Chỉnh sửa trang cá nhân</span>
                </Button>
              </>
            ) : (
              <>
                <Button
                  onClick={onFollow}
                  variant={isFollowing ? "outline" : "primary"}
                  className="flex items-center space-x-2"
                >
                  <UserPlus className="h-4 w-4" />
                  <span>{isFollowing ? 'Đang theo dõi' : 'Theo dõi'}</span>
                </Button>

                <Button
                  onClick={onMessage}
                  variant="outline"
                  className="flex items-center space-x-2"
                >
                  <MessageCircle className="h-4 w-4" />
                  <span>Nhắn tin</span>
                </Button>

                <button className="p-2 hover:bg-gray-100 rounded-lg transition-colors">
                  <MoreHorizontal className="h-5 w-5 text-gray-600" />
                </button>
              </>
            )}
          </div>
        </div>

        {/* Bio Section */}
        {user.bio && (
          <div className="mt-6 bg-gray-50 rounded-lg p-4">
            <h3 className="font-semibold text-gray-900 mb-2">Giới thiệu</h3>
            <p className="text-gray-700 vietnamese-text leading-relaxed">
              {showFullBio || user.bio.length <= 200
                ? user.bio
                : `${user.bio.substring(0, 200)}...`
              }
            </p>
            {user.bio.length > 200 && (
              <button
                onClick={() => setShowFullBio(!showFullBio)}
                className="text-blue-600 hover:text-blue-700 font-medium mt-2"
              >
                {showFullBio ? 'Thu gọn' : 'Xem thêm'}
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
