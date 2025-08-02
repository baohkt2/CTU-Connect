'use client';

import React, { useState, useRef } from 'react';
import { Button } from '@/components/ui/Button';
import { Textarea } from '@/components/ui/Textarea';
import { Input } from '@/components/ui/Input';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorAlert } from '@/components/ui/ErrorAlert';
import { postService } from '@/services/postService';
import { CreatePostRequest } from '@/types';
import { X, Image, Hash, Globe, Users, Lock } from 'lucide-react';

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
  const [tagInput, setTagInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleInputChange = (field: keyof CreatePostRequest, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleAddTag = () => {
    if (tagInput.trim() && !formData.tags?.includes(tagInput.trim())) {
      setFormData(prev => ({
        ...prev,
        tags: [...(prev.tags || []), tagInput.trim()]
      }));
      setTagInput('');
    }
  };

  const handleRemoveTag = (tagToRemove: string) => {
    setFormData(prev => ({
      ...prev,
      tags: prev.tags?.filter(tag => tag !== tagToRemove) || []
    }));
  };

  const handleTagInputKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      handleAddTag();
    }
  };

  const handleFileSelect = (selectedFiles: File[]) => {
    setFiles(selectedFiles);
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

  return (
    <div className={`bg-white rounded-lg shadow-md p-6 ${className}`}>
      <h3 className="text-lg font-semibold mb-4">Create Post</h3>
      
      {error && (
        <ErrorAlert 
          message={error} 
          onClose={() => setError(null)} 
          className="mb-4"
        />
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Title Input */}
        <Input
          placeholder="Post title (optional)"
          value={formData.title || ''}
          onChange={(e) => handleInputChange('title', e.target.value)}
        />

        {/* Content Textarea */}
        <Textarea
          placeholder="What's on your mind?"
          value={formData.content}
          onChange={(e) => handleInputChange('content', e.target.value)}
          rows={4}
          required
        />

        {/* Tags Input */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Hash className="h-4 w-4 text-gray-500" />
            <Input
              placeholder="Add tags (press Enter or comma to add)"
              value={tagInput}
              onChange={(e) => setTagInput(e.target.value)}
              onKeyDown={handleTagInputKeyDown}
            />
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleAddTag}
              disabled={!tagInput.trim()}
            >
              Add
            </Button>
          </div>
          
          {/* Display Tags */}
          {formData.tags && formData.tags.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {formData.tags.map((tag, index) => (
                <span
                  key={index}
                  className="inline-flex items-center px-2 py-1 bg-blue-100 text-blue-800 text-sm rounded-full"
                >
                  #{tag}
                  <button
                    type="button"
                    onClick={() => handleRemoveTag(tag)}
                    className="ml-1 text-blue-600 hover:text-blue-800"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Category Input */}
        <Input
          placeholder="Category (optional)"
          value={formData.category || ''}
          onChange={(e) => handleInputChange('category', e.target.value)}
        />

        {/* File Upload */}
        <div>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*,video/*"
            onChange={(e) => {
              const selectedFiles = Array.from(e.target.files || []);
              handleFileSelect(selectedFiles);
            }}
            className="hidden"
          />
          
          <Button
            type="button"
            variant="outline"
            onClick={() => fileInputRef.current?.click()}
            className="w-full"
          >
            {/* eslint-disable-next-line jsx-a11y/alt-text */}
            <Image className="h-4 w-4 mr-2" />
            Add Photos/Videos
          </Button>

          {/* Display Selected Files */}
          {files.length > 0 && (
            <div className="mt-2 space-y-2">
              {files.map((file, index) => (
                <div
                  key={index}
                  className="flex items-center justify-between p-2 bg-gray-50 rounded"
                >
                  <span className="text-sm text-gray-700">{file.name}</span>
                  <button
                    type="button"
                    onClick={() => handleRemoveFile(index)}
                    className="text-red-500 hover:text-red-700"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Visibility Selector */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Visibility
          </label>
          <div className="flex gap-2">
            {visibilityOptions.map(({ value, label, icon: Icon }) => (
              <button
                key={value}
                type="button"
                onClick={() => handleInputChange('visibility', value)}
                className={`flex items-center px-3 py-2 rounded-md border transition-colors ${
                  formData.visibility === value
                    ? 'bg-blue-50 border-blue-300 text-blue-700'
                    : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
                }`}
              >
                <Icon className="h-4 w-4 mr-2" />
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex justify-end gap-2 pt-4">
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
            loading={isLoading}
          >
            {isLoading ? 'Creating...' : 'Create Post'}
          </Button>
        </div>
      </form>
    </div>
  );
};
