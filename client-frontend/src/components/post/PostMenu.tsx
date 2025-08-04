'use client';

import React, { useState } from 'react';
import {
  MoreHorizontal,
  Edit3,
  Trash2,
  Flag,
  EyeOff,
  UserX,
  Copy,
  Bookmark,
  Share
} from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

interface PostMenuProps {
  post: any;
  onEdit?: () => void;
  onDelete?: () => void;
  onReport?: () => void;
  onHide?: () => void;
  onBlock?: () => void;
  onBookmark?: () => void;
  onShare?: () => void;
  onCopyLink?: () => void;
  className?: string;
}

export const PostMenu: React.FC<PostMenuProps> = ({
  post,
  onEdit,
  onDelete,
  onReport,
  onHide,
  onBlock,
  onBookmark,
  onShare,
  onCopyLink,
  className = ''
}) => {
  const { user } = useAuth();
  const [isOpen, setIsOpen] = useState(false);
  const isOwnPost = user?.id === post.authorId || user?.id === post.author?.id;

  const handleAction = (action: () => void) => {
    action();
    setIsOpen(false);
  };

  const menuItems = [
    // Own post actions
    ...(isOwnPost ? [
      {
        icon: <Edit3 className="h-4 w-4" />,
        label: 'Chỉnh sửa bài viết',
        action: onEdit,
        className: 'text-gray-700 hover:bg-gray-50'
      },
      {
        icon: <Trash2 className="h-4 w-4 text-red-500" />,
        label: 'Xóa bài viết',
        action: onDelete,
        className: 'text-red-600 hover:bg-red-50'
      }
    ] : []),

    // Common actions
    {
      icon: <Bookmark className="h-4 w-4" />,
      label: 'Lưu bài viết',
      action: onBookmark,
      className: 'text-gray-700 hover:bg-gray-50'
    },
    {
      icon: <Share className="h-4 w-4" />,
      label: 'Chia sẻ',
      action: onShare,
      className: 'text-gray-700 hover:bg-gray-50'
    },
    {
      icon: <Copy className="h-4 w-4" />,
      label: 'Sao chép liên kết',
      action: onCopyLink,
      className: 'text-gray-700 hover:bg-gray-50'
    },

    // Other user's post actions
    ...(!isOwnPost ? [
      {
        icon: <Flag className="h-4 w-4 text-red-500" />,
        label: 'Báo cáo bài viết',
        action: onReport,
        className: 'text-red-600 hover:bg-red-50'
      },
      {
        icon: <EyeOff className="h-4 w-4" />,
        label: 'Ẩn bài viết',
        action: onHide,
        className: 'text-gray-700 hover:bg-gray-50'
      },
      {
        icon: <UserX className="h-4 w-4 text-red-500" />,
        label: `Chặn bài viết từ ${post.author?.fullName || post.author?.name || post.authorName}`,
        action: onBlock,
        className: 'text-red-600 hover:bg-red-50'
      }
    ] : [])
  ].filter(item => item.action); // Only include items with actions

  return (
    <div className={`relative ${className}`}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="p-2 hover:bg-gray-100 rounded-full transition-colors"
        aria-label="Tùy chọn bài viết"
      >
        <MoreHorizontal className="h-4 w-4 text-gray-500" />
      </button>

      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          />

          {/* Menu */}
          <div className="absolute right-0 top-full mt-1 bg-white rounded-lg shadow-lg border border-gray-200 py-2 z-50 min-w-[220px]">
            {menuItems.map((item, index) => (
              <button
                key={index}
                onClick={() => handleAction(item.action!)}
                className={`
                  flex items-center space-x-3 w-full px-4 py-2 text-sm transition-colors vietnamese-text
                  ${item.className}
                `}
              >
                {item.icon}
                <span>{item.label}</span>
              </button>
            ))}

            {menuItems.length === 0 && (
              <div className="px-4 py-2 text-sm text-gray-500 vietnamese-text">
                Không có tùy chọn nào
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};
