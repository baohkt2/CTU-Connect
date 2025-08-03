'use client';

import React, { useState, useRef, useCallback } from 'react';
import { Button } from '@/components/ui/Button';
import { Textarea } from '@/components/ui/Textarea';
import { Input } from '@/components/ui/Input';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorAlert } from '@/components/ui/ErrorAlert';
import { postService } from '@/services/postService';
import { CreatePostRequest } from '@/types';
import { X, Image, Hash, Globe, Users, Lock, Video } from 'lucide-react';

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

  const [files, setFiles] = useState<File[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tagInput, setTagInput] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleInputChange = (field: keyof CreatePostRequest, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleAddTag = useCallback(() => {
    const newTag = tagInput.trim();
    const tags = formData.tags ?? [];

    if (newTag && !tags.includes(newTag)) {
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
      // Xóa tag cuối nếu input rỗng và backspace
      e.preventDefault();
      handleRemoveTag(formData.tags && formData.tags[formData.tags.length - 1]);
    }
  };

  const handleFileSelect = (selectedFiles: File[]) => {
    // Lọc giữ file ảnh/video hợp lệ, tránh trùng (có thể mở rộng)
    const filtered = selectedFiles.filter(f => {
      const type = f.type;
      return type.startsWith('image/') || type.startsWith('video/');
    });
    setFiles(prev => [...prev, ...filtered]);
  };

  const handleRemoveFile = (index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!formData.content.trim()) {
      setError('Content is required');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const post = await postService.createPost(formData, files.length > 0 ? files : undefined);

      // Reset form
      setFormData({
        title: '',
        content: '',
        tags: [],
        category: '',
        visibility: 'PUBLIC'
      });
      setFiles([]);
      setTagInput('');

      if (onPostCreated) {
        onPostCreated(post);
      }
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to create post');
    } finally {
      setIsLoading(false);
    }
  };

  const visibilityOptions = [
    { value: 'PUBLIC', label: 'Public', icon: Globe },
    { value: 'FRIENDS', label: 'Friends', icon: Users },
    { value: 'PRIVATE', label: 'Private', icon: Lock }
  ];

  // Hiển thị thumbnail cho các file đã chọn (ảnh hoặc video)
  const renderFilePreview = (file: File, index: number) => {
    const isImage = file.type.startsWith('image/');
    const previewUrl = URL.createObjectURL(file);

    return (
        <div key={index} className="relative w-24 h-24 rounded-md overflow-hidden border border-gray-300 flex-shrink-0">
          {isImage ? (
              <img
                  src={previewUrl}
                  alt={file.name}
                  className="object-cover w-full h-full"
                  loading="lazy"
              />
          ) : (
              <video
                  src={previewUrl}
                  className="object-cover w-full h-full"
                  preload="metadata"
                  controls={false}
                  muted
              />
          )}
          <button
              type="button"
              aria-label={`Remove file ${file.name}`}
              onClick={() => handleRemoveFile(index)}
              className="absolute top-1 right-1 bg-black bg-opacity-50 rounded-full p-1 text-white hover:bg-opacity-80 transition"
          >
            <X className="w-4 h-4 pointer-events-none" />
          </button>
        </div>
    );
  };

  return (
      <section className={`bg-white rounded-lg shadow-md p-6 max-w-3xl mx-auto ${className}`} aria-label="Create a new post">
        <h3 className="text-lg font-semibold mb-4">Create Post</h3>

        {error && (
            <ErrorAlert
                message={error}
                onClose={() => setError(null)}
                className="mb-4"
            />
        )}

        <form onSubmit={handleSubmit} className="space-y-5" noValidate>
          {/* Title */}
          <Input
              id="post-title"
              placeholder="Post title (optional)"
              value={formData.title}
              onChange={(e) => handleInputChange('title', e.target.value)}
              aria-label="Post title"
              maxLength={150}
          />

          {/* Content */}
          <Textarea
              id="post-content"
              placeholder="What's on your mind?"
              value={formData.content}
              onChange={(e) => handleInputChange('content', e.target.value)}
              rows={5}
              required
              aria-required="true"
              aria-describedby="content-help"
              maxLength={2000}
          />
          <p id="content-help" className="text-xs text-gray-500 select-none">
            Max 2000 characters
          </p>

          {/* Tags */}
          <div>
            <label htmlFor="tag-input" className="flex items-center gap-2 mb-2 text-sm font-medium text-gray-700">
              <Hash aria-hidden="true" /> Tags
            </label>
            <div className="flex gap-2 mb-3">
              <Input
                  id="tag-input"
                  placeholder="Add tags (press Enter or comma)"
                  value={tagInput}
                  onChange={(e) => setTagInput(e.target.value)}
                  onKeyDown={handleTagInputKeyDown}
                  maxLength={20}
                  autoComplete="off"
                  aria-describedby="tag-help"
              />
              <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleAddTag}
                  disabled={!tagInput.trim()}
                  aria-label="Add tag"
              >
                Add
              </Button>
            </div>
            <p id="tag-help" className="text-xs text-gray-500 select-none mb-2">
              Press Enter or comma to add a tag. Maximum 20 characters per tag.
            </p>
            {formData.tags && formData.tags.length > 0 && (
                <div className="flex flex-wrap gap-2 max-h-28 overflow-y-auto pb-1">
                  {formData.tags && formData.tags.map((tag, i) => (
                      <span
                          key={i}
                          className="inline-flex items-center px-3 py-1 text-sm bg-blue-100 text-blue-800 rounded-full select-none"
                      >
                  #{tag}
                        <button
                            type="button"
                            aria-label={`Remove tag ${tag}`}
                            onClick={() => handleRemoveTag(tag)}
                            className="ml-1 inline-flex justify-center items-center text-blue-600 hover:text-blue-900 focus:outline-none focus:ring-2 focus:ring-blue-400 rounded"
                        >
                    <X className="h-3 w-3" />
                  </button>
                </span>
                  ))}
                </div>
            )}
          </div>

          {/* Category */}
          <Input
              id="post-category"
              placeholder="Category (optional)"
              value={formData.category}
              onChange={(e) => handleInputChange('category', e.target.value)}
              aria-label="Category"
              maxLength={50}
          />

          {/* File Upload */}
          <div>
            <label className="mb-2 block text-sm font-medium text-gray-700 flex items-center gap-2">
              <Image className="h-4 w-4" aria-hidden="true" />
              Photos/Videos
            </label>
            <input
                ref={fileInputRef}
                type="file"
                multiple
                accept="image/*,video/*"
                onChange={(e) => {
                  const selectedFiles = Array.from(e.target.files || []);
                  handleFileSelect(selectedFiles);
                  e.target.value = ''; // reset để có thể chọn lại file đã chọn trước đó
                }}
                className="sr-only"
                aria-label="Add photos or videos"
            />
            <Button
                type="button"
                variant="outline"
                onClick={() => fileInputRef.current?.click()}
                className="mb-3 w-full"
            >
              Add Photos/Videos
            </Button>

            {/* Preview files */}
            {files.length > 0 && (
                <div className="flex gap-3 flex-wrap max-h-36 overflow-y-auto">
                  {files.map(renderFilePreview)}
                </div>
            )}
          </div>

          {/* Visibility */}
          <fieldset>
            <legend className="mb-2 text-sm font-medium text-gray-700 flex items-center gap-2">
              <Globe className="h-4 w-4" aria-hidden="true" /> Visibility
            </legend>
            <div className="flex gap-2 flex-wrap">
              {visibilityOptions.map(({ value, label, icon: Icon }) => {
                const isSelected = formData.visibility === value;
                return (
                    <button
                        key={value}
                        type="button"
                        aria-pressed={isSelected}
                        onClick={() => handleInputChange('visibility', value)}
                        className={`flex items-center gap-2 px-4 py-2 border rounded-md text-sm transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 ${
                            isSelected
                                ? 'bg-blue-50 border-blue-400 text-blue-700'
                                : 'border-gray-300 text-gray-700 hover:bg-gray-50'
                        }`}
                    >
                      <Icon className="h-4 w-4" aria-hidden="true" />
                      {label}
                    </button>
                );
              })}
            </div>
          </fieldset>

          {/* Buttons */}
          <div className="flex justify-end gap-3 pt-4">
            {onCancel && (
                <Button
                    type="button"
                    variant="outline"
                    onClick={onCancel}
                    disabled={isLoading}
                >
                  Cancel
                </Button>
            )}
            <Button
                type="submit"
                disabled={isLoading || !formData.content.trim()}
                aria-disabled={isLoading || !formData.content.trim()}
                loading={isLoading}
            >
              {isLoading ? (
                  <>
                    <LoadingSpinner size="sm" />
                    <span className="ml-2">Creating...</span>
                  </>
              ) : (
                  'Create Post'
              )}
            </Button>
          </div>
        </form>
      </section>
  );
};
