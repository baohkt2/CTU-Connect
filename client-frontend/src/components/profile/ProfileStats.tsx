'use client';

import React from 'react';
import { Users, FileText, Heart, Eye } from 'lucide-react';

interface ProfileStatsProps {
  stats: {
    posts?: number;
    followers?: number;
    following?: number;
    likes?: number;
    views?: number;
  };
  onStatsClick?: (type: 'posts' | 'followers' | 'following') => void;
}

export const ProfileStats: React.FC<ProfileStatsProps> = ({ stats, onStatsClick }) => {
  const formatNumber = (num: number): string => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  const statItems = [
    {
      key: 'posts' as const,
      label: 'Bài viết',
      value: stats.posts || 0,
      icon: <FileText className="h-5 w-5 text-blue-500" />,
      clickable: true
    },
    {
      key: 'followers' as const,
      label: 'Người theo dõi',
      value: stats.followers || 0,
      icon: <Users className="h-5 w-5 text-green-500" />,
      clickable: true
    },
    {
      key: 'following' as const,
      label: 'Đang theo dõi',
      value: stats.following || 0,
      icon: <Users className="h-5 w-5 text-purple-500" />,
      clickable: true
    },
    {
      key: 'likes' as const,
      label: 'Lượt thích',
      value: stats.likes || 0,
      icon: <Heart className="h-5 w-5 text-red-500" />,
      clickable: false
    },
    {
      key: 'views' as const,
      label: 'Lượt xem',
      value: stats.views || 0,
      icon: <Eye className="h-5 w-5 text-gray-500" />,
      clickable: false
    }
  ];

  return (
    <div className="bg-white rounded-lg shadow-sm p-6">
      <h2 className="text-xl font-bold text-gray-900 mb-6 vietnamese-text">Thống kê</h2>

      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        {statItems.map((item) => (
          <div
            key={item.key}
            className={`text-center p-4 rounded-lg border ${
              item.clickable 
                ? 'cursor-pointer hover:bg-gray-50 hover:border-gray-300 transition-all duration-200' 
                : 'bg-gray-50'
            }`}
            onClick={item.clickable ? () => onStatsClick?.(item.key) : undefined}
          >
            <div className="flex justify-center mb-2">
              {item.icon}
            </div>
            <div className="text-2xl font-bold text-gray-900 mb-1">
              {formatNumber(item.value)}
            </div>
            <div className="text-sm text-gray-600 vietnamese-text">
              {item.label}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
