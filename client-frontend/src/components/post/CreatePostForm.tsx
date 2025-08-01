import React, { useState } from 'react';
import { usePostHooks } from '@/hooks/usePostHooks';
import { useAuth } from '@/contexts/AuthContext';
import Button from '@/components/ui/Button';
import Textarea from '@/components/ui/Textarea';
import Avatar from '@/components/ui/Avatar';
import Card from '@/components/ui/Card';
import { PhotoIcon, XMarkIcon } from '@heroicons/react/24/outline';

const CreatePostForm: React.FC = () => {
  const { user } = useAuth();
  const { useCreatePost } = usePostHooks();
  const createPostMutation = useCreatePost();

  const [content, setContent] = useState('');
  const [images, setImages] = useState<File[]>([]);
  const [imagePreviews, setImagePreviews] = useState<string[]>([]);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length + images.length > 4) {
      alert('Bạn chỉ có thể tải lên tối đa 4 ảnh');
      return;
    }

    const newImages = [...images, ...files];
    setImages(newImages);

    // Create previews
    const newPreviews = [...imagePreviews];
    files.forEach(file => {
      const reader = new FileReader();
      reader.onload = (e) => {
        newPreviews.push(e.target?.result as string);
        setImagePreviews([...newPreviews]);
      };
      reader.readAsDataURL(file);
    });
  };

  const removeImage = (index: number) => {
    const newImages = images.filter((_, i) => i !== index);
    const newPreviews = imagePreviews.filter((_, i) => i !== index);
    setImages(newImages);
    setImagePreviews(newPreviews);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!content.trim()) return;

    try {
      await createPostMutation.mutateAsync({
        content: content.trim(),
        images: images.length > 0 ? images : undefined,
      });

      // Reset form
      setContent('');
      setImages([]);
      setImagePreviews([]);
    } catch (error) {
      console.error('Error creating post:', error);
    }
  };

  return (
    <Card className="mb-6">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="flex space-x-3">
          <Avatar
            src={user?.avatarUrl || '/default-avatar.png'}
            alt={user?.fullName || 'User'}
            size="md"
          />
          <div className="flex-1">
            <Textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              placeholder="Bạn đang nghĩ gì?"
              rows={3}
              className="resize-none border-none focus:ring-0 p-0 text-lg"
            />
          </div>
        </div>

        {/* Image previews */}
        {imagePreviews.length > 0 && (
          <div className="grid grid-cols-2 gap-2">
            {imagePreviews.map((preview, index) => (
              <div key={index} className="relative">
                <img
                  src={preview}
                  alt={`Preview ${index + 1}`}
                  className="w-full h-32 object-cover rounded-lg"
                />
                <button
                  type="button"
                  onClick={() => removeImage(index)}
                  className="absolute top-2 right-2 bg-black bg-opacity-50 text-white rounded-full p-1 hover:bg-opacity-70"
                >
                  <XMarkIcon className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="flex items-center justify-between pt-4 border-t">
          <div className="flex items-center space-x-4">
            <label className="flex items-center space-x-2 cursor-pointer text-gray-600 hover:text-blue-600">
              <PhotoIcon className="w-5 h-5" />
              <span>Ảnh</span>
              <input
                type="file"
                accept="image/*"
                multiple
                onChange={handleImageUpload}
                className="hidden"
              />
            </label>
          </div>

          <Button
            type="submit"
            disabled={!content.trim() || createPostMutation.isPending}
            loading={createPostMutation.isPending}
          >
            Đăng bài
          </Button>
        </div>
      </form>
    </Card>
  );
};

export default CreatePostForm;
