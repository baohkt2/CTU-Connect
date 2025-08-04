'use client';

import React, { useState } from 'react';
import { Modal } from '@/components/ui/Modal';
import { Button } from '@/components/ui/Button';
import { Textarea } from '@/components/ui/Textarea';
import { Input } from '@/components/ui/Input';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { UpdatePostRequest } from '@/types';
import { X, Image, Video, Hash } from 'lucide-react';

interface PostEditModalProps {
  isOpen: boolean;
  onClose: () => void;
  post: any;
  onSave: (updatedPost: UpdatePostRequest) => Promise<void>;
}

export const PostEditModal: React.FC<PostEditModalProps> = ({
  isOpen,
  onClose,
  post,
  onSave
}) => {
  const [title, setTitle] = useState(post.title || '');
  const [content, setContent] = useState(post.content || '');
  const [category, setCategory] = useState(post.category || '');
  const [tags, setTags] = useState<string[]>(post.tags || []);
  const [tagInput, setTagInput] = useState('');
  const [visibility, setVisibility] = useState(post.visibility || post.privacy || 'PUBLIC');
  const [isSaving, setIsSaving] = useState(false);

  const handleAddTag = () => {
    if (tagInput.trim() && !tags.includes(tagInput.trim())) {
      setTags(prev => [...prev, tagInput.trim()]);
      setTagInput('');
    }
  };

  const handleRemoveTag = (tagToRemove: string) => {
    setTags(prev => prev.filter(tag => tag !== tagToRemove));
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && tagInput.trim()) {
      e.preventDefault();
      handleAddTag();
    }
  };

  const handleSave = async () => {
    if (!content.trim()) return;

    setIsSaving(true);
    try {
      const updatedPost: UpdatePostRequest = {
        title: title.trim() || undefined,
        content: content.trim(),
        category: category.trim() || undefined,
        tags: tags.length > 0 ? tags : undefined,
        visibility: visibility as 'PUBLIC' | 'FRIENDS' | 'PRIVATE'
      };

      await onSave(updatedPost);
      onClose();
    } catch (error) {
      console.error('Error saving post:', error);
    } finally {
      setIsSaving(false);
    }
  };

  const handleClose = () => {
    if (!isSaving) {
      onClose();
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={handleClose}
      title="Chỉnh sửa bài viết"
      size="lg"
    >
      <div className="space-y-6">
        {/* Title */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2 vietnamese-text">
            Tiêu đề (tùy chọn)
          </label>
          <Input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Nhập tiêu đề bài viết..."
            className="vietnamese-text"
            disabled={isSaving}
          />
        </div>

        {/* Content */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2 vietnamese-text">
            Nội dung <span className="text-red-500">*</span>
          </label>
          <Textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="Bạn đang nghĩ gì?"
            className="min-h-[120px] vietnamese-text"
            disabled={isSaving}
          />
        </div>

        {/* Category */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2 vietnamese-text">
            Danh mục
          </label>
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 vietnamese-text"
            disabled={isSaving}
          >
            <option value="">Chọn danh mục</option>
            <option value="Tin tức">Tin tức</option>
            <option value="Học tập">Học tập</option>
            <option value="Giải trí">Giải trí</option>
            <option value="Thể thao">Thể thao</option>
            <option value="Công nghệ">Công nghệ</option>
            <option value="Du lịch">Du lịch</option>
            <option value="Khác">Khác</option>
          </select>
        </div>

        {/* Tags */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2 vietnamese-text">
            Thẻ hashtag
          </label>
          <div className="flex flex-wrap gap-2 mb-2">
            {tags.map((tag, index) => (
              <span
                key={index}
                className="inline-flex items-center bg-blue-100 text-blue-800 text-sm px-2 py-1 rounded-full"
              >
                <Hash className="h-3 w-3 mr-1" />
                {tag}
                <button
                  type="button"
                  onClick={() => handleRemoveTag(tag)}
                  className="ml-1 hover:text-blue-600"
                  disabled={isSaving}
                >
                  <X className="h-3 w-3" />
                </button>
              </span>
            ))}
          </div>
          <div className="flex space-x-2">
            <Input
              type="text"
              value={tagInput}
              onChange={(e) => setTagInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Thêm thẻ hashtag..."
              className="flex-1 vietnamese-text"
              disabled={isSaving}
            />
            <Button
              type="button"
              onClick={handleAddTag}
              disabled={!tagInput.trim() || isSaving}
              variant="outline"
              size="sm"
            >
              Thêm
            </Button>
          </div>
        </div>

        {/* Visibility */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2 vietnamese-text">
            Quyền riêng tư
          </label>
          <select
            value={visibility}
            onChange={(e) => setVisibility(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 vietnamese-text"
            disabled={isSaving}
          >
            <option value="PUBLIC">Công khai</option>
            <option value="FRIENDS">Bạn bè</option>
            <option value="PRIVATE">Riêng tư</option>
          </select>
        </div>

        {/* Media Preview (if exists) */}
        {(post.images?.length > 0 || post.videos?.length > 0) && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2 vietnamese-text">
              Media hiện tại
            </label>
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-center space-x-4 text-sm text-gray-600">
                {post.images?.length > 0 && (
                  <div className="flex items-center space-x-1">
                    <Image className="h-4 w-4" />
                    <span>{post.images.length} ảnh</span>
                  </div>
                )}
                {post.videos?.length > 0 && (
                  <div className="flex items-center space-x-1">
                    <Video className="h-4 w-4" />
                    <span>{post.videos.length} video</span>
                  </div>
                )}
              </div>
              <p className="text-xs text-gray-500 mt-2 vietnamese-text">
                Lưu ý: Không thể chỉnh sửa media trong phiên bản hiện tại
              </p>
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex justify-end space-x-3 pt-4 border-t">
          <Button
            variant="outline"
            onClick={handleClose}
            disabled={isSaving}
          >
            Hủy
          </Button>
          <Button
            onClick={handleSave}
            disabled={!content.trim() || isSaving}
            className="flex items-center space-x-2"
          >
            {isSaving ? (
              <>
                <LoadingSpinner size="sm" />
                <span>Đang lưu...</span>
              </>
            ) : (
              <span>Lưu thay đổi</span>
            )}
          </Button>
        </div>
      </div>
    </Modal>
  );
};
