'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Post, CreateCommentRequest } from '@/types';
import { postService } from '@/services/postService';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Textarea } from '@/components/ui/Textarea';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { t, formatTimeAgo } from '@/utils/localization';
import {
  Heart,
  MessageCircle,
  Share,
  Bookmark,
  MoreHorizontal,
  Send,
  Eye,
  Globe,
  Users,
  Lock
} from 'lucide-react';

interface PostCardProps {
  post: any; // Sử dụng any để phù hợp với structure mới
  onPostUpdate?: (updatedPost: any) => void;
  onPostDelete?: (postId: string) => void;
  className?: string;
}

export const PostCard: React.FC<PostCardProps> = ({
                                                    post,
                                                    onPostUpdate,
                                                    onPostDelete,
                                                    className = ''
                                                  }) => {
  const [isLiked, setIsLiked] = useState(false);
  const [isBookmarked, setIsBookmarked] = useState(false);
  const [showComments, setShowComments] = useState(false);
  const [commentText, setCommentText] = useState('');
  const [comments, setComments] = useState<any[]>([]);
  const [isLoadingComments, setIsLoadingComments] = useState(false);
  const [isSubmittingComment, setIsSubmittingComment] = useState(false);
  const [isLoadingInteraction, setIsLoadingInteraction] = useState(false);
  const [actionFeedback, setActionFeedback] = useState<string | null>(null);

  // Load trạng thái like, bookmark khi mount
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const status = await postService.getInteractionStatus(post.id);
        if (mounted) {
          setIsLiked(status.hasLiked);
          setIsBookmarked(status.hasBookmarked);
        }
      } catch (error) {
        console.debug('Không thể tải trạng thái tương tác:', error);
      }
    })();
    return () => { mounted = false; };
  }, [post.id]);

  // Show feedback message temporarily
  const showFeedback = (message: string) => {
    setActionFeedback(message);
    setTimeout(() => setActionFeedback(null), 2000);
  };

  // Toggle phần comment
  const toggleComments = useCallback(async () => {
    if (!showComments && comments.length === 0) {
      setIsLoadingComments(true);
      try {
        const response = await postService.getComments(post.id);
        // Đọc từ response.content thay vì response trực tiếp
        setComments(response.content || []);
      } catch (error) {
        console.error('Không thể tải bình luận:', error);
        showFeedback('Không thể tải bình luận');
      } finally {
        setIsLoadingComments(false);
      }
    }
    setShowComments(v => !v);
  }, [showComments, comments.length, post.id]);

  // Chức năng tương tác: like, bookmark, share
  const handleInteraction = useCallback(async (type: 'like' | 'bookmark' | 'share') => {
    if (isLoadingInteraction) return;
    setIsLoadingInteraction(true);
    try {
      if (type === 'like') {
        await postService.toggleLike(post.id);
        setIsLiked(l => {
          const newLiked = !l;
          const newLikes = newLiked ? post.stats.likes + 1 : post.stats.likes - 1;
          onPostUpdate?.({
            ...post,
            stats: { ...post.stats, likes: newLikes }
          });
          showFeedback(newLiked ? 'Đã thích bài viết' : 'Đã bỏ thích');
          return newLiked;
        });
      } else if (type === 'bookmark') {
        await postService.toggleBookmark(post.id);
        setIsBookmarked(b => {
          const newBookmarked = !b;
          const newBookmarks = newBookmarked ? post.stats.bookmarks + 1 : post.stats.bookmarks - 1;
          onPostUpdate?.({
            ...post,
            stats: { ...post.stats, bookmarks: newBookmarks }
          });
          showFeedback(newBookmarked ? 'Đã lưu bài viết' : 'Đã bỏ lưu bài viết');
          return newBookmarked;
        });
      } else if (type === 'share') {
        await postService.sharePost(post.id);
        onPostUpdate?.({
          ...post,
          stats: { ...post.stats, shares: post.stats.shares + 1 }
        });
        await navigator.clipboard.writeText(`${window.location.origin}/posts/${post.id}`);
        showFeedback('Đã sao chép liên kết bài viết');
      }
    } catch (error) {
      console.error(`Không thể thực hiện ${type}:`, error);
      showFeedback('Có lỗi xảy ra, vui lòng thử lại');
    } finally {
      setIsLoadingInteraction(false);
    }
  }, [isLoadingInteraction, onPostUpdate, post]);

  // Gửi comment
  const handleSubmitComment = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    if (!commentText.trim() || isSubmittingComment) return;

    setIsSubmittingComment(true);
    try {
      const commentData: CreateCommentRequest = { content: commentText.trim() };
      const newComment = await postService.createComment(post.id, commentData);
      setComments(prev => [...prev, newComment]);
      setCommentText('');
      onPostUpdate?.({
        ...post,
        stats: { ...post.stats, comments: post.stats.comments + 1 }
      });
      showFeedback('Đã thêm bình luận');
    } catch (error) {
      console.error('Không thể tạo bình luận:', error);
      showFeedback('Không thể gửi bình luận');
    } finally {
      setIsSubmittingComment(false);
    }
  }, [commentText, isSubmittingComment, onPostUpdate, post]);

  // Format số lượng hiển thị
  const formatStats = (count: number): string => {
    if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`;
    if (count >= 1000) return `${(count / 1000).toFixed(1)}K`;
    return count.toString();
  };

  // Get privacy icon
  const getPrivacyIcon = () => {
    switch (post.privacy || post.visibility) {
      case 'PUBLIC':
        return <Globe className="h-3 w-3 text-green-600" />;
      case 'FRIENDS':
        return <Users className="h-3 w-3 text-blue-600" />;
      case 'PRIVATE':
        return <Lock className="h-3 w-3 text-gray-600" />;
      default:
        return <Globe className="h-3 w-3 text-green-600" />;
    }
  };

  const getPrivacyText = () => {
    switch (post.privacy || post.visibility) {
      case 'PUBLIC': return 'Công khai';
      case 'FRIENDS': return 'Bạn bè';
      case 'PRIVATE': return 'Riêng tư';
      default: return 'Công khai';
    }
  };

  return (
    <Card className={`post-card bg-white rounded-xl shadow-sm hover:shadow-md transition-all duration-300 border border-gray-100 mb-6  ${className}`}>
      {/* Action feedback */}
      {actionFeedback && (
        <div className="absolute top-4 right-4 z-10 bg-green-500 text-white px-4 py-2 rounded-lg text-sm animate-fade-in shadow-lg">
          {actionFeedback}
        </div>
      )}

      <header className="flex items-center justify-between p-6 pb-4">
        <div className="flex items-center space-x-4">
          <div className="relative">
            {post.author?.avatarUrl || post.authorAvatar ? (
              <img
                src={post.author?.avatarUrl || post.authorAvatar}
                alt="Avatar"
                className="w-12 h-12 rounded-full object-cover ring-2 ring-white shadow-sm"
              />
            ) : (
              <div className="w-12 h-12 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center text-white font-semibold text-lg">
                {(post.author?.fullName || post.author?.name || post.authorName)?.charAt(0)?.toUpperCase() || 'U'}
              </div>
            )}
            {/* Online status indicator */}
            <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-500 border-2 border-white rounded-full"></div>
          </div>
          <div className="flex-1">
            <div className="flex items-center space-x-2">
              <h3 className="font-semibold text-gray-900 hover:text-indigo-600 cursor-pointer transition-colors vietnamese-text">
                {post.author?.fullName || post.author?.name || post.authorName || 'Người dùng'}
              </h3>
              {post.author?.role && (
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                  post.author.role === 'LECTURER' 
                    ? 'bg-blue-100 text-blue-800' 
                    : 'bg-green-100 text-green-800'
                }`}>
                  {post.author.role === 'LECTURER' ? 'Giảng viên' : 'Sinh viên'}
                </span>
              )}
            </div>
            <div className="flex items-center space-x-2 mt-1">
              <p className="text-sm text-gray-500">
                {formatTimeAgo(post.createdAt)}
              </p>
              <span className="text-gray-300">•</span>
              <div className="flex items-center space-x-1">
                {getPrivacyIcon()}
                <span className="text-xs text-gray-500">{getPrivacyText()}</span>
              </div>
            </div>
          </div>
        </div>
        <Button variant="ghost" size="sm" aria-label="Tùy chọn khác" disabled>
          <MoreHorizontal className="h-5 w-5" />
        </Button>
      </header>

      <div className="px-6 pb-4">
        {/* Title */}
        {post.title && (
          <h2 className="text-xl font-semibold text-gray-900 mb-3 line-clamp-2 vietnamese-text">
            {post.title}
          </h2>
        )}

        {/* Content */}
        <div className="text-gray-700 whitespace-pre-wrap leading-relaxed text-base mb-4 vietnamese-text">
          {post.content}
        </div>

        {/* Category */}
        {post.category && (
          <div className="mb-3">
            <span className="inline-flex items-center bg-indigo-50 text-indigo-700 px-3 py-1 rounded-full text-sm font-medium">
              📚 {post.category}
            </span>
          </div>
        )}

        {/* Tags */}
        {post.tags && post.tags.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-4">
            {post.tags.map((tag: string, index: number) => (
              <span
                key={index}
                className="inline-block bg-blue-50 text-blue-700 px-3 py-1 rounded-full text-sm font-medium hover:bg-blue-100 transition-colors cursor-pointer"
              >
                #{tag}
              </span>
            ))}
          </div>
        )}

        {/* Media - Images */}
        {post.images && post.images.length > 0 && (
          <div className="mb-4">
            <div className={`grid gap-2 rounded-xl overflow-hidden ${
              post.images.length === 1 ? 'grid-cols-1' :
              post.images.length === 2 ? 'grid-cols-2' :
              post.images.length === 3 ? 'grid-cols-2' : 'grid-cols-2'
            }`}>
              {post.images.slice(0, 4).map((imageUrl: string, index: number) => (
                <div
                  key={index}
                  className={`relative bg-gray-100 ${
                    post.images.length === 3 && index === 0 ? 'row-span-2' :
                    post.images.length > 4 && index === 3 ? 'relative' : ''
                  }`}
                >
                  <img
                    src={imageUrl}
                    alt={`Ảnh bài viết ${index + 1}`}
                    className="w-full h-full object-cover cursor-pointer hover:opacity-95 transition-opacity min-h-[200px] max-h-[400px]"
                    onClick={() => window.open(imageUrl, '_blank')}
                  />
                  {post.images.length > 4 && index === 3 && (
                    <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center cursor-pointer">
                      <span className="text-white text-xl font-semibold">
                        +{post.images.length - 4}
                      </span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Media - Videos */}
        {post.videos && post.videos.length > 0 && (
          <div className="mb-4">
            <div className="grid gap-3">
              {post.videos.map((videoUrl: string, index: number) => (
                <div key={index} className="rounded-xl overflow-hidden bg-gray-50">
                  <video
                    src={videoUrl}
                    controls
                    className="w-full h-auto max-h-[500px] object-cover"
                    preload="metadata"
                  >
                    Trình duyệt của bạn không hỗ trợ video.
                  </video>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Stats Bar */}
      {(post.stats?.likes > 0 || post.stats?.comments > 0 || post.stats?.shares > 0) && (
        <div className="px-6 py-3 border-t border-gray-100">
          <div className="flex items-center justify-between text-sm text-gray-500">
            <div className="flex items-center space-x-4">
              {post.stats?.likes > 0 && (
                <span className="flex items-center space-x-1">
                  <div className="flex -space-x-1">
                    <div className="w-5 h-5 bg-red-500 rounded-full flex items-center justify-center">
                      <Heart className="w-3 h-3 text-white fill-current" />
                    </div>
                  </div>
                  <span>{formatStats(post.stats.likes)}</span>
                </span>
              )}
            </div>
            <div className="flex items-center space-x-4">
              {post.stats?.comments > 0 && (
                <span>{formatStats(post.stats.comments)} bình luận</span>
              )}
              {post.stats?.shares > 0 && (
                <span>{formatStats(post.stats.shares)} chia sẻ</span>
              )}
              {post.stats?.views > 0 && (
                <span>{formatStats(post.stats.views)} lượt xem</span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="border-t border-gray-100 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <button
              onClick={() => handleInteraction('like')}
              disabled={isLoadingInteraction}
              className={`flex items-center space-x-2 px-4 py-3 rounded-xl transition-all duration-200 flex-1 justify-center ${
                isLiked 
                  ? 'bg-red-50 text-red-600 hover:bg-red-100' 
                  : 'text-gray-600 hover:bg-gray-50 hover:text-red-600'
              }`}
              aria-label={isLiked ? t('posts.unlikePost') : t('posts.likePost')}
            >
              <Heart className={`h-5 w-5 ${isLiked ? 'fill-current' : ''} transition-transform hover:scale-110`} />
              <span className="font-medium">Thích</span>
            </button>

            <button
              onClick={toggleComments}
              className="flex items-center space-x-2 px-4 py-3 rounded-xl text-gray-600 hover:bg-gray-50 hover:text-blue-600 transition-all duration-200 flex-1 justify-center"
              aria-label={showComments ? t('posts.hideComments') : t('posts.viewComments')}
            >
              <MessageCircle className="h-5 w-5 transition-transform hover:scale-110" />
              <span className="font-medium">Bình luận</span>
            </button>

            <button
              onClick={() => handleInteraction('share')}
              disabled={isLoadingInteraction}
              className="flex items-center space-x-2 px-4 py-3 rounded-xl text-gray-600 hover:bg-gray-50 hover:text-green-600 transition-all duration-200 flex-1 justify-center"
              aria-label={t('posts.sharePost')}
            >
              <Share className="h-5 w-5 transition-transform hover:scale-110" />
              <span className="font-medium">Chia sẻ</span>
            </button>
          </div>
        </div>
      </div>

      {/* Comments Section */}
      {showComments && (
        <div className="border-t border-gray-100 bg-gray-50">
          <div className="p-6">
            {/* Comment Form */}
            <form onSubmit={handleSubmitComment} className="mb-6">
              <div className="flex space-x-4">
                <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-full flex items-center justify-center text-white font-semibold flex-shrink-0">
                  U
                </div>
                <div className="flex-1">
                  <Textarea
                    value={commentText}
                    onChange={(e) => setCommentText(e.target.value)}
                    placeholder={t('posts.writeComment')}
                    className="min-h-[60px] resize-none border-gray-200 focus:border-indigo-500 focus:ring-indigo-500 rounded-xl bg-white"
                    disabled={isSubmittingComment}
                  />
                  <div className="flex justify-end mt-3">
                    <Button
                      type="submit"
                      size="sm"
                      disabled={!commentText.trim() || isSubmittingComment}
                      className="flex items-center space-x-2 rounded-xl"
                    >
                      {isSubmittingComment ? (
                        <LoadingSpinner size="sm" />
                      ) : (
                        <Send className="h-4 w-4" />
                      )}
                      <span>{isSubmittingComment ? t('actions.loading') : 'Gửi'}</span>
                    </Button>
                  </div>
                </div>
              </div>
            </form>

            {/* Comments List */}
            {isLoadingComments ? (
              <div className="flex justify-center py-8">
                <LoadingSpinner />
                <span className="ml-3 text-gray-600">{t('actions.loading')}</span>
              </div>
            ) : (
              <div className="space-y-4">
                {comments.length === 0 ? (
                  <p className="text-gray-500 text-center py-8 italic vietnamese-text">
                    Chưa có bình luận nào. Hãy là người đầu tiên bình luận!
                  </p>
                ) : (
                  comments.map((comment) => (
                    <div key={comment.id} className="flex space-x-4 bg-white p-4 rounded-xl">
                      <div className="flex-shrink-0">
                        {comment.author?.avatarUrl ? (
                          <img
                            src={comment.author.avatarUrl}
                            alt="Avatar"
                            className="w-10 h-10 rounded-full object-cover"
                          />
                        ) : (
                          <div className="w-10 h-10 bg-gradient-to-br from-green-500 to-blue-600 rounded-full flex items-center justify-center text-white font-semibold">
                            {comment.author?.fullName?.charAt(0) || comment.author?.name?.charAt(0) || 'A'}
                          </div>
                        )}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-2">
                          <span className="font-semibold text-gray-900 vietnamese-text">
                            {comment.author?.fullName || comment.author?.name || 'Ẩn danh'}
                          </span>
                          {comment.author?.role && (
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                              comment.author.role === 'LECTURER' 
                                ? 'bg-blue-100 text-blue-800' 
                                : 'bg-green-100 text-green-800'
                            }`}>
                              {comment.author.role === 'LECTURER' ? 'GV' : 'SV'}
                            </span>
                          )}
                          <span className="text-sm text-gray-500">
                            {formatTimeAgo(comment.createdAt)}
                          </span>
                        </div>
                        <p className="text-gray-700 leading-relaxed vietnamese-text">
                          {comment.content}
                        </p>
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </Card>
  );
};
