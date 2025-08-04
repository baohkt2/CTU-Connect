'use client';

import React, { useState } from 'react';
import { User } from '@/types';
import { Camera, X, Upload, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Modal } from '@/components/ui/Modal';

interface ProfilePhotoModalProps {
  isOpen: boolean;
  onClose: () => void;
  user: User;
  type: 'avatar' | 'cover';
  onPhotoUpdate: (photoUrl: string) => void;
}

export const ProfilePhotoModal: React.FC<ProfilePhotoModalProps> = ({
  isOpen,
  onClose,
  user,
  type,
  onPhotoUpdate
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    try {
      // TODO: Implement actual upload logic
      // const uploadResult = await mediaService.uploadImage(selectedFile);
      // onPhotoUpdate(uploadResult.url);

      // Mock upload for now
      setTimeout(() => {
        onPhotoUpdate(previewUrl || '');
        setIsUploading(false);
        onClose();
        resetModal();
      }, 2000);
    } catch (error) {
      console.error('Error uploading photo:', error);
      setIsUploading(false);
    }
  };

  const resetModal = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
  };

  const handleClose = () => {
    if (!isUploading) {
      resetModal();
      onClose();
    }
  };

  const title = type === 'avatar' ? 'Cập nhật ảnh đại diện' : 'Cập nhật ảnh bìa';
  const aspectRatio = type === 'avatar' ? 'aspect-square' : 'aspect-[3/1]';

  return (
    <Modal isOpen={isOpen} onClose={handleClose} title={title}>
      <div className="space-y-6">
        {/* Current Photo */}
        <div className="text-center">
          <h3 className="text-sm font-medium text-gray-900 mb-3 vietnamese-text">
            {type === 'avatar' ? 'Ảnh đại diện hiện tại' : 'Ảnh bìa hiện tại'}
          </h3>
          <div className={`mx-auto bg-gray-200 rounded-lg overflow-hidden ${
            type === 'avatar' ? 'w-32 h-32 rounded-full' : 'w-full h-40'
          }`}>
            {(type === 'avatar' ? user.avatarUrl : user.backgroundUrl) ? (
              <img
                src={type === 'avatar' ? user.avatarUrl! : user.backgroundUrl!}
                alt={title}
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full bg-gradient-to-br from-blue-400 to-purple-600 flex items-center justify-center text-white text-2xl font-bold">
                {type === 'avatar' ? (user.fullName || user.name || 'U').charAt(0).toUpperCase() : ''}
              </div>
            )}
          </div>
        </div>

        {/* Preview New Photo */}
        {previewUrl && (
          <div className="text-center">
            <h3 className="text-sm font-medium text-gray-900 mb-3 vietnamese-text">
              Xem trước
            </h3>
            <div className={`mx-auto bg-gray-200 rounded-lg overflow-hidden ${
              type === 'avatar' ? 'w-32 h-32 rounded-full' : 'w-full h-40'
            }`}>
              <img
                src={previewUrl}
                alt="Preview"
                className="w-full h-full object-cover"
              />
            </div>
          </div>
        )}

        {/* File Upload */}
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-gray-400 transition-colors">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
            id="photo-upload"
            disabled={isUploading}
          />
          <label htmlFor="photo-upload" className="cursor-pointer">
            <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-sm text-gray-600 vietnamese-text">
              Nhấp để chọn ảnh hoặc kéo thả ảnh vào đây
            </p>
            <p className="text-xs text-gray-500 mt-1 vietnamese-text">
              Định dạng: JPG, PNG (Tối đa 10MB)
            </p>
          </label>
        </div>

        {/* Action Buttons */}
        <div className="flex justify-end space-x-3">
          <Button
            variant="outline"
            onClick={handleClose}
            disabled={isUploading}
          >
            Hủy
          </Button>
          <Button
            onClick={handleUpload}
            disabled={!selectedFile || isUploading}
            className="flex items-center space-x-2"
          >
            {isUploading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Đang tải lên...</span>
              </>
            ) : (
              <>
                <Camera className="h-4 w-4" />
                <span>Cập nhật ảnh</span>
              </>
            )}
          </Button>
        </div>
      </div>
    </Modal>
  );
};
