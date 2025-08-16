'use client';

import React, { useState, useRef, useCallback } from 'react';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorAlert } from '@/components/ui/ErrorAlert';
import RichTextEditor, { RichTextEditorRef } from '@/components/ui/RichTextEditor';
import { postService } from '@/services/postService';
import { CreatePostRequest } from '@/types';
import { t } from '@/utils/localization';
import { stripHtml, isHtmlEmpty, validateContent } from '@/utils/richTextUtils';
import { X, Image, Hash, Globe, Users, Lock, Video, Plus, Upload } from 'lucide-react';

interface CreatePostProps {
  onPostCreated?: (post: any) => void;
  onCancel?: () => void;
  className?: string;
}

export const CreatePost: React.FC<CreatePostProps> = ({
                                                        onPostCreated,
                                                        onCancel,
                                                        className = ''
                                                      }) => {
  const [formData, setFormData] = useState<CreatePostRequest>({
    title: '',
    content: '',
    tags: [],
    category: '',
    visibility: 'PUBLIC'
  });

  const [richContent, setRichContent] = useState<string>('');
  const [files, setFiles] = useState<File[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tagInput, setTagInput] = useState('');
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const richTextEditorRef = useRef<RichTextEditorRef>(null);

  const handleInputChange = (field: keyof CreatePostRequest, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
    setError(null); // Clear error when user types
  };

  const handleContentChange = (content: string) => {
    setRichContent(content);
    // Save the rich HTML content instead of stripping it
    setFormData(prev => ({
      ...prev,
      content: content // Save HTML content with formatting
    }));
    setError(null);
  };

  const handleAddTag = useCallback(() => {
    const newTag = tagInput.trim();
    const tags = formData.tags ?? [];

    if (newTag && !tags.includes(newTag) && tags.length < 5) { // Limit to 5 tags
      setFormData(prev => ({
        ...prev,
        tags: [...(prev.tags ?? []), newTag]
      }));
      setTagInput('');
    }
  }, [formData.tags, tagInput]);

  const handleRemoveTag = useCallback((tagToRemove: string) => {
    setFormData(prev => ({
      ...prev,
      tags: (prev.tags ?? []).filter(tag => tag !== tagToRemove)
    }));
  }, []);

  const handleTagInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      handleAddTag();
    } else if (e.key === 'Backspace' && tagInput === '' && formData.tags && formData.tags.length > 0) {
      e.preventDefault();
      handleRemoveTag(formData.tags && formData.tags[formData.tags.length - 1]);
    }
  };

  const handleFileSelect = (selectedFiles: File[]) => {
    const maxSize = 50 * 1024 * 1024; // 50MB
    const maxFiles = 5;

    const validFiles = selectedFiles.filter(f => {
      const type = f.type;
      const isValidType = type.startsWith('image/') || type.startsWith('video/');
      const isValidSize = f.size <= maxSize;

      if (!isValidType) {
        setError('Chỉ hỗ trợ file ảnh và video');
        return false;
      }

      if (!isValidSize) {
        setError('Kích thước file không được vượt quá 50MB');
        return false;
      }

      return true;
    });

    if (files.length + validFiles.length > maxFiles) {
      setError(`Chỉ được tải lên tối đa ${maxFiles} file`);
      return;
    }

    setFiles(prev => [...prev, ...validFiles]);
    setError(null);
  };

  const handleRemoveFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Validate using rich text content
    if (isHtmlEmpty(richContent) || !validateContent(richContent, 1)) {
      setError('Nội dung bài viết không được để trống');
      return;
    }

    setIsLoading(true);
    setError(null);
    setUploadProgress(0);

    try {
      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      const result = await postService.createPost(formData, files);

      clearInterval(progressInterval);
      setUploadProgress(100);

      setSuccessMessage('Đã tạo bài viết thành công!');

      // Reset form including rich text content
      setFormData({
        title: '',
        content: '',
        tags: [],
        category: '',
        visibility: 'PUBLIC'
      });
      setRichContent('');
      setFiles([]);
      setTagInput('');

      onPostCreated?.(result);

      // Hide success message after 3 seconds
      setTimeout(() => setSuccessMessage(null), 3000);

    } catch (error: any) {
      console.error('Không thể tạo bài viết:', error);
      setError(error.response?.data?.message || 'Có lỗi xảy ra khi tạo bài viết');
    } finally {
      setIsLoading(false);
      setUploadProgress(0);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className={`create-post bg-white rounded-lg shadow-sm border border-gray-200 p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-900 flex items-center">
          <Plus className="h-5 w-5 mr-2 text-indigo-600" />
          {t('posts.createPost')}
        </h2>
        {onCancel && (
          <Button variant="ghost" size="sm" onClick={onCancel}>
            <X className="h-4 w-4" />
          </Button>
        )}
      </div>

      {/* Success Message */}
      {successMessage && (
        <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg text-green-800">
          {successMessage}
        </div>
      )}

      {/* Error Message */}
      {error && (
        <ErrorAlert message={error} onClose={() => setError(null)} className="mb-4" />
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Title */}
        <div>
          <Input
            type="text"
            value={formData.title}
            onChange={(e) => handleInputChange('title', e.target.value)}
            placeholder="Tiêu đề bài viết (tùy chọn)"
            className="text-lg font-medium border-0 border-b-2 border-gray-200 focus:border-indigo-500 rounded-none px-0 bg-transparent"
            maxLength={100}
          />
          <div className="text-xs text-gray-500 mt-1 text-right">
            {formData.title?.length}/100
          </div>
        </div>

        {/* Content */}
        <div>
          <RichTextEditor
            value={richContent}
            onChange={handleContentChange}
            placeholder="Bạn đang nghĩ gì? Hãy chia sẻ với cộng đồng CTU..."
            className="min-h-[120px] border-gray-200 focus:border-indigo-500 focus:ring-indigo-500 resize-none text-gray-700 leading-relaxed"
            ref={richTextEditorRef}
            required
          />
          <div className="text-xs text-gray-500 mt-1 text-right">
            {formData.content.length}/2000
          </div>
        </div>

        {/* Category Selection */}
        <div>
          <select
            value={formData.category}
            onChange={(e) => handleInputChange('category', e.target.value)}
            className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:border-indigo-500 focus:ring-indigo-500 text-gray-700"
          >
            <option value="">Chọn danh mục</option>
            <option value="research">🔬 Nghiên cứu khoa học</option>
            <option value="teaching">🎓 Đào tạo & Giảng dạy</option>
            <option value="aquaculture">🐟 Thủy sản & Nuôi trồng</option>
            <option value="technology">💻 Công nghệ & Kỹ thuật</option>
            <option value="climate">🌡️ Khí hậu & Môi trường</option>
            <option value="student">👥 Sinh viên CTU</option>
            <option value="events">📅 Sự kiện CTU</option>
            <option value="discussion">💬 Trao đổi học thuật</option>
            <option value="other">📝 Khác</option>
          </select>
        </div>

        {/* Tags */}
        <div>
          <div className="flex flex-wrap gap-2 mb-2">
            {formData.tags?.map((tag, index) => (
              <span
                key={index}
                className="inline-flex items-center bg-blue-50 text-blue-700 px-3 py-1 rounded-full text-sm font-medium"
              >
                #{tag}
                <button
                  type="button"
                  onClick={() => handleRemoveTag(tag)}
                  className="ml-2 text-blue-500 hover:text-blue-700"
                >
                  <X className="h-3 w-3" />
                </button>
              </span>
            ))}
          </div>
          <div className="flex items-center space-x-2">
            <div className="flex-1 relative">
              <Hash className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                type="text"
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyDown={handleTagInputKeyDown}
                placeholder="Thêm thẻ (nhấn Enter hoặc dấu phẩy)"
                className="pl-10"
                maxLength={20}
                disabled={formData.tags && formData.tags.length >= 5}
              />
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleAddTag}
              disabled={!tagInput.trim() || (formData.tags && formData.tags.length >= 5)}
            >
              Thêm
            </Button>
          </div>
          <div className="text-xs text-gray-500 mt-1">
            Tối đa 5 thẻ, mỗi thẻ không quá 20 ký tự
          </div>
        </div>

        {/* File Upload */}
        <div>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*,video/*"
            onChange={(e) => handleFileSelect(Array.from(e.target.files || []))}
            className="hidden"
          />

          <div className="border-2 border-dashed border-gray-200 rounded-lg p-4 hover:border-indigo-300 transition-colors">
            <div className="text-center">
              <Upload className="mx-auto h-8 w-8 text-gray-400 mb-2" />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="text-indigo-600 hover:text-indigo-700 font-medium"
              >
                Thêm ảnh hoặc video
              </button>
              <p className="text-sm text-gray-500 mt-1">
                Hỗ trợ JPG, PNG, GIF, MP4, AVI. Tối đa 50MB mỗi file, 5 file.
              </p>
            </div>
          </div>

          {/* File Preview */}
          {files.length > 0 && (
            <div className="mt-3 grid grid-cols-2 md:grid-cols-3 gap-3">
              {files.map((file, index) => (
                <div key={index} className="relative group">
                  <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden">
                    {file.type.startsWith('image/') ? (
                      <img
                        src={URL.createObjectURL(file)}
                        alt="Preview"
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <Video className="h-8 w-8 text-gray-400" />
                      </div>
                    )}
                  </div>
                  <button
                    type="button"
                    onClick={() => handleRemoveFile(index)}
                    className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <X className="h-3 w-3" />
                  </button>
                  <div className="text-xs text-gray-500 mt-1 truncate">
                    {file.name} ({formatFileSize(file.size)})
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Visibility */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Quyền riêng tư
          </label>
          <div className="flex space-x-4">
            <label className="flex items-center">
              <input
                type="radio"
                name="visibility"
                value="PUBLIC"
                checked={formData.visibility === 'PUBLIC'}
                onChange={(e) => handleInputChange('visibility', e.target.value)}
                className="mr-2"
              />
              <Globe className="h-4 w-4 mr-1 text-green-600" />
              <span className="text-sm">Công khai</span>
            </label>
            <label className="flex items-center">
              <input
                type="radio"
                name="visibility"
                value="FRIENDS"
                checked={formData.visibility === 'FRIENDS'}
                onChange={(e) => handleInputChange('visibility', e.target.value)}
                className="mr-2"
              />
              <Users className="h-4 w-4 mr-1 text-blue-600" />
              <span className="text-sm">Bạn bè</span>
            </label>
            <label className="flex items-center">
              <input
                type="radio"
                name="visibility"
                value="PRIVATE"
                checked={formData.visibility === 'PRIVATE'}
                onChange={(e) => handleInputChange('visibility', e.target.value)}
                className="mr-2"
              />
              <Lock className="h-4 w-4 mr-1 text-gray-600" />
              <span className="text-sm">Riêng tư</span>
            </label>
          </div>
        </div>

        {/* Upload Progress */}
        {isLoading && uploadProgress > 0 && (
          <div className="bg-gray-50 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600">Đang tải lên...</span>
              <span className="text-sm font-medium text-indigo-600">{uploadProgress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-indigo-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
          </div>
        )}

        {/* Submit Buttons */}
        <div className="flex justify-end space-x-3 pt-4 border-t border-gray-100">
          {onCancel && (
            <Button
              type="button"
              variant="outline"
              onClick={onCancel}
              disabled={isLoading}
            >
              {t('actions.cancel')}
            </Button>
          )}
          <Button
            type="submit"
            disabled={isLoading || !formData.content.trim()}
            className="flex items-center space-x-2"
          >
            {isLoading ? (
              <LoadingSpinner size="sm" />
            ) : (
              <Plus className="h-4 w-4" />
            )}
            <span>{isLoading ? 'Đang đăng...' : 'Đăng bài viết'}</span>
          </Button>
        </div>
      </form>
    </div>
  );
};
