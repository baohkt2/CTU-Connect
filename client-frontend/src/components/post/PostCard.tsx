import React, { useState } from 'react';
import { usePostHooks, useCommentHooks } from '@/hooks/usePostHooks';
import { useAuth } from '@/contexts/AuthContext';
import { Post } from '@/types';
import { formatDate } from '@/utils/helpers';
import Avatar from '@/components/ui/Avatar';
import Button from '@/components/ui/Button';
import Card from '@/components/ui/Card';
import Modal from '@/components/ui/Modal';
import Textarea from '@/components/ui/Textarea';
import Link from 'next/link';
import {
  HeartIcon,
  ChatBubbleOvalLeftIcon,
  ShareIcon,
  EllipsisHorizontalIcon,
  TrashIcon
} from '@heroicons/react/24/outline';
import { HeartIcon as HeartIconSolid } from '@heroicons/react/24/solid';

interface PostCardProps {
  post: Post;
}

const PostCard: React.FC<PostCardProps> = ({ post }) => {
  const { user } = useAuth();
  const { useLikePost, useUnlikePost, useDeletePost } = usePostHooks();
  const { useComments, useCreateComment } = useCommentHooks();

  const [showComments, setShowComments] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [commentText, setCommentText] = useState('');

  const likePostMutation = useLikePost();
  const unlikePostMutation = useUnlikePost();
  const deletePostMutation = useDeletePost();
  const createCommentMutation = useCreateComment();

  const { data: commentsData } = useComments(post.id, 0, 10);

  const handleLike = async () => {
    if (post.isLiked) {
      await unlikePostMutation.mutateAsync(post.id);
    } else {
      await likePostMutation.mutateAsync(post.id);
    }
  };

  const handleDeletePost = async () => {
    try {
      await deletePostMutation.mutateAsync(post.id);
      setShowDeleteModal(false);
    } catch (error) {
      console.error('Error deleting post:', error);
    }
  };

  const handleAddComment = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!commentText.trim()) return;

    try {
      await createCommentMutation.mutateAsync({
        postId: post.id,
        content: commentText.trim()
      });
      setCommentText('');
    } catch (error) {
      console.error('Error adding comment:', error);
    }
  };

  const isOwner = user?.id === post.authorId;

  return (
    <>
      <Card className="mb-4" hover>
        {/* Post Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <Avatar
              src={post.author.avatarUrl || '/default-avatar.png'}
              alt={post.author.fullName}
              size="md"
              online={post.author.isOnline}
            />
            <div>
              <Link
                href={`/profile/${post.author.id}`}
                className="font-semibold text-gray-900 hover:text-blue-600"
              >
                {post.author.fullName}
              </Link>
              <p className="text-sm text-gray-500">
                @{post.author.username} · {formatDate(post.createdAt)}
              </p>
            </div>
          </div>

          {isOwner && (
            <div className="relative group">
              <button className="p-2 rounded-full hover:bg-gray-100">
                <EllipsisHorizontalIcon className="w-5 h-5 text-gray-500" />
              </button>
              <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 z-10 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all">
                <button
                  onClick={() => setShowDeleteModal(true)}
                  className="flex items-center w-full px-4 py-2 text-sm text-red-600 hover:bg-red-50"
                >
                  <TrashIcon className="w-4 h-4 mr-2" />
                  Xóa bài đăng
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Post Content */}
        <div className="mb-4">
          <p className="text-gray-900 whitespace-pre-wrap">{post.content}</p>
        </div>

        {/* Post Images */}
        {post.images && post.images.length > 0 && (
          <div className={`mb-4 grid gap-2 ${
            post.images.length === 1 ? 'grid-cols-1' : 
            post.images.length === 2 ? 'grid-cols-2' : 
            'grid-cols-2'
          }`}>
            {post.images.map((image, index) => (
              <img
                key={index}
                src={image}
                alt={`Post image ${index + 1}`}
                className="w-full h-64 object-cover rounded-lg cursor-pointer hover:opacity-90 transition-opacity"
                onClick={() => {
                  // TODO: Implement image modal/gallery
                }}
              />
            ))}
          </div>
        )}

        {/* Post Actions */}
        <div className="flex items-center justify-between pt-4 border-t">
          <div className="flex items-center space-x-4">
            <button
              onClick={handleLike}
              className={`flex items-center space-x-2 px-3 py-1 rounded-full transition-colors ${
                post.isLiked 
                  ? 'text-red-600 hover:bg-red-50' 
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              {post.isLiked ? (
                <HeartIconSolid className="w-5 h-5" />
              ) : (
                <HeartIcon className="w-5 h-5" />
              )}
              <span className="text-sm">{post.likes}</span>
            </button>

            <button
              onClick={() => setShowComments(!showComments)}
              className="flex items-center space-x-2 px-3 py-1 rounded-full text-gray-600 hover:bg-gray-100 transition-colors"
            >
              <ChatBubbleOvalLeftIcon className="w-5 h-5" />
              <span className="text-sm">{post.comments}</span>
            </button>

            <button className="flex items-center space-x-2 px-3 py-1 rounded-full text-gray-600 hover:bg-gray-100 transition-colors">
              <ShareIcon className="w-5 h-5" />
              <span className="text-sm">Chia sẻ</span>
            </button>
          </div>
        </div>

        {/* Comments Section */}
        {showComments && (
          <div className="mt-4 pt-4 border-t">
            {/* Add Comment Form */}
            <form onSubmit={handleAddComment} className="mb-4">
              <div className="flex space-x-3">
                <Avatar
                  src={user?.avatarUrl || '/default-avatar.png'}
                  alt={user?.fullName || 'User'}
                  size="sm"
                />
                <div className="flex-1">
                  <Textarea
                    value={commentText}
                    onChange={(e) => setCommentText(e.target.value)}
                    placeholder="Viết bình luận..."
                    rows={2}
                    className="resize-none"
                  />
                  <div className="flex justify-end mt-2">
                    <Button
                      type="submit"
                      size="sm"
                      disabled={!commentText.trim() || createCommentMutation.isPending}
                      loading={createCommentMutation.isPending}
                    >
                      Bình luận
                    </Button>
                  </div>
                </div>
              </div>
            </form>

            {/* Comments List */}
            {commentsData?.content && commentsData.content.length > 0 && (
              <div className="space-y-3">
                {commentsData.content.map((comment) => (
                  <div key={comment.id} className="flex space-x-3">
                    <Avatar
                      src={comment.author.avatarUrl || '/default-avatar.png'}
                      alt={comment.author.fullName}
                      size="sm"
                    />
                    <div className="flex-1">
                      <div className="bg-gray-50 rounded-lg px-3 py-2">
                        <Link
                          href={`/profile/${comment.author.id}`}
                          className="font-semibold text-sm text-gray-900 hover:text-blue-600"
                        >
                          {comment.author.fullName}
                        </Link>
                        <p className="text-gray-700 text-sm mt-1">{comment.content}</p>
                      </div>
                      <div className="flex items-center space-x-4 mt-1">
                        <span className="text-xs text-gray-500">
                          {formatDate(comment.createdAt)}
                        </span>
                        <button className="text-xs text-gray-500 hover:text-blue-600">
                          Thích
                        </button>
                        <button className="text-xs text-gray-500 hover:text-blue-600">
                          Trả lời
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </Card>

      {/* Delete Post Modal */}
      <Modal
        isOpen={showDeleteModal}
        onClose={() => setShowDeleteModal(false)}
        title="Xóa bài đăng"
      >
        <div className="space-y-4">
          <p className="text-gray-700">
            Bạn có chắc chắn muốn xóa bài đăng này không? Hành động này không thể hoàn tác.
          </p>
          <div className="flex justify-end space-x-3">
            <Button
              variant="secondary"
              onClick={() => setShowDeleteModal(false)}
            >
              Hủy
            </Button>
            <Button
              variant="danger"
              onClick={handleDeletePost}
              loading={deletePostMutation.isPending}
            >
              Xóa
            </Button>
          </div>
        </div>
      </Modal>
    </>
  );
};

export default PostCard;
