'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Post, CreateCommentRequest, UpdatePostRequest } from '@/types';
import { postService } from '@/services/postService';
import { Button } from '@/components/ui/Button';
import  Card  from '@/components/ui/Card';
import { Textarea } from '@/components/ui/Textarea';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ReactionButton } from '@/components/ui/ReactionButton';
import { PostMenu } from '@/components/post/PostMenu';
import { PostEditModal } from '@/components/post/PostEditModal';
import { EditIndicator } from '@/components/post/EditIndicator';
import { CommentItem } from '@/components/post/CommentItem';
import { t, formatTimeAgo } from '@/utils/localization';
import {
  MessageCircle,
  Share,
  Eye,
  Globe,
  Users,
  Lock, ThumbsUp, Heart, MoreHorizontal, Flag, Trash2, EyeOff
} from 'lucide-react';
import Avatar from "@/components/ui/Avatar";
import {useAuth} from "@/contexts/AuthContext";

interface PostCardProps {
  post: any;
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
  const { user } = useAuth();
  const [isLiked, setIsLiked] = useState(false);
  const [isBookmarked, setIsBookmarked] = useState(false);
  const [showComments, setShowComments] = useState(false);
  const [commentText, setCommentText] = useState('');
  const [comments, setComments] = useState<any[]>([]);
  const [isLoadingComments, setIsLoadingComments] = useState(false);
  const [isSubmittingComment, setIsSubmittingComment] = useState(false);
  const [isLoadingInteraction, setIsLoadingInteraction] = useState(false);
  const [actionFeedback, setActionFeedback] = useState<string | null>(null);
  const [showEditModal, setShowEditModal] = useState(false);
  const [currentReaction, setCurrentReaction] = useState<string | null>(null);
  const [reactionCounts, setReactionCounts] = useState<{[key: string]: number}>({});
  const [commentMenus, setCommentMenus] = useState<{[key: string]: boolean}>({});

  const isOwnPost = user?.id === post.authorId || user?.id === post.author?.id;

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
    setTimeout(() => setActionFeedback(null), 1500);
  };

  // Toggle phần comment
  const toggleComments = useCallback(async () => {
    if (!showComments && comments.length === 0) {
      setIsLoadingComments(true);
      try {
        const response = await postService.getComments(post.id);
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
      showFeedback('Không th�� gửi bình luận');
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
        return <Globe className="h-3 w-3 text-gray-500" />;
      case 'FRIENDS':
        return <Users className="h-3 w-3 text-gray-500" />;
      case 'PRIVATE':
        return <Lock className="h-3 w-3 text-gray-500" />;
      default:
        return <Globe className="h-3 w-3 text-gray-500" />;
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

  // Function to toggle comment menu
  const toggleCommentMenu = (commentId: string) => {
    setCommentMenus(prev => ({
      ...prev,
      [commentId]: !prev[commentId]
    }));
  };

  // New enhanced handlers for reactions and post actions
  const handleReactionClick = useCallback(async (reactionId: string) => {
    if (isLoadingInteraction) return;
    setIsLoadingInteraction(true);

    try {
      // TODO: Implement reaction API call
      // await postService.addReaction(post.id, reactionId);
      setCurrentReaction(reactionId);

      // Update reaction counts
      setReactionCounts(prev => ({
        ...prev,
        [reactionId]: (prev[reactionId] || 0) + 1
      }));

      showFeedback(`Đã ${reactionId === 'LIKE' ? 'thích' : 'phản ứng'} bài viết`);
    } catch (error) {
      console.error('Error adding reaction:', error);
      showFeedback('Không thể thêm phản ứng');
    } finally {
      setIsLoadingInteraction(false);
    }
  }, [isLoadingInteraction, post.id]);

  const handleReactionRemove = useCallback(async () => {
    if (isLoadingInteraction || !currentReaction) return;
    setIsLoadingInteraction(true);

    try {
      // TODO: Implement remove reaction API call
      // await postService.removeReaction(post.id);

      // Update reaction counts
      setReactionCounts(prev => ({
        ...prev,
        [currentReaction]: Math.max((prev[currentReaction] || 0) - 1, 0)
      }));

      setCurrentReaction(null);
      showFeedback('Đã bỏ phản ứng');
    } catch (error) {
      console.error('Error removing reaction:', error);
      showFeedback('Không thể bỏ phản ứng');
    } finally {
      setIsLoadingInteraction(false);
    }
  }, [isLoadingInteraction, currentReaction, post.id]);

  // Post menu handlers
  const handlePostEdit = () => {
    setShowEditModal(true);
  };

  const handlePostDelete = async () => {
    if (window.confirm('Bạn có chắc chắn muốn xóa bài viết này?')) {
      try {
        await postService.deletePost(post.id);
        onPostDelete?.(post.id);
        showFeedback('Đã xóa bài viết');
      } catch (error) {
        console.error('Error deleting post:', error);
        showFeedback('Không thể xóa bài viết');
      }
    }
  };

  const handlePostSave = async (updatedPost: UpdatePostRequest) => {
    try {
      const result = await postService.updatePost(post.id, updatedPost);
      onPostUpdate?.(result);
      showFeedback('Đã cập nhật bài viết');
    } catch (error) {
      console.error('Error updating post:', error);
      throw error;
    }
  };

  const handlePostReport = () => {
    // TODO: Implement report functionality
    showFeedback('Đã báo cáo bài viết');
  };

  const handlePostHide = () => {
    // TODO: Implement hide functionality
    showFeedback('Đã ẩn bài viết');
  };

  const handlePostBlock = () => {
    // TODO: Implement block functionality
    showFeedback(`Đã chặn bài viết từ ${post.author?.fullName || post.author?.name || 'người dùng này'}`);
  };

  const handleCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(`${window.location.origin}/posts/${post.id}`);
      showFeedback('Đã sao chép liên kết');
    } catch (error) {
      showFeedback('Không thể sao chép liên kết');
    }
  };

  return (
    <Card className={`bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow duration-200 border border-gray-200 mb-4 ${className}`}>
      {/* Feedback Toast */}
      {actionFeedback && (
        <div className="absolute top-3 right-3 z-10 bg-gray-800 text-white px-3 py-1 rounded text-xs animate-fade-in">
          {actionFeedback}
        </div>
      )}
      
      {/* Header - Facebook Style */}
      <div className="p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {/* Avatar */}

                <Avatar
                    id={post.author?.id}
                    src={post.author?.avatarUrl || '/default-avatar.png'}
                    alt={ post.author?.fullName ||  post.author?.username || 'Avatar'}
                    size="md"
                />

            
            {/* User Info */}
            <div className="flex-1">
              <div className="flex items-center space-x-2">
                <h3 className="font-semibold text-sm text-gray-900 hover:underline cursor-pointer vietnamese-text">
                  {post.author?.fullName || post.author?.name || post.authorName || 'Người dùng'}
                </h3>
                {post.author?.role && (
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                    post.author.role === 'LECTURER' 
                      ? 'bg-blue-100 text-blue-700' 
                      : 'bg-green-100 text-green-700'
                  }`}>
                    {post.author.role === 'LECTURER' ? 'GV' : 'SV'}
                  </span>
                )}
              </div>
              
              <div className="flex items-center space-x-1 text-xs text-gray-500 mt-0.5">
                <time dateTime={post.createdAt}>
                  {formatTimeAgo(post.createdAt)}
                </time>
                <span>•</span>
                {getPrivacyIcon()}
              </div>
            </div>
          </div>
          
          {/* More Options */}
          <PostMenu
            post={post}
            onEdit={isOwnPost ? handlePostEdit : undefined}
            onDelete={isOwnPost ? handlePostDelete : undefined}
            onReport={!isOwnPost ? handlePostReport : undefined}
            onHide={!isOwnPost ? handlePostHide : undefined}
            onBlock={!isOwnPost ? handlePostBlock : undefined}
            onBookmark={() => handleInteraction('bookmark')}
            onShare={() => handleInteraction('share')}
            onCopyLink={handleCopyLink}
          />
        </div>

        {/* Edit Indicator */}
        <EditIndicator
          isEdited={post.isEdited}
          className="mt-1 ml-12"
        />
      </div>

      {/* Content */}
      <div className="px-3 pb-3">
        {/* Title */}
        {post.title && (
          <h2 className="font-semibold text-gray-900 mb-2 vietnamese-text">
            {post.title}
          </h2>
        )}
        
        {/* Text Content */}
        <div className="text-gray-800 text-sm leading-relaxed vietnamese-text mb-3">
          {post.content}
        </div>
        
        {/* Tags */}
        {post.tags && post.tags.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-3">
            {post.tags.map((tag: string, index: number) => (
              <span
                key={index}
                className="text-blue-600 hover:underline cursor-pointer text-sm"
              >
                #{tag}
              </span>
            ))}
          </div>
        )}

        {/* Media - Images */}
        {post.images && post.images.length > 0 && (
          <div className="mb-3 -mx-3">
            <div className={`grid gap-0.5 ${
              post.images.length === 1 ? 'grid-cols-1' :
              post.images.length === 2 ? 'grid-cols-2' :
              post.images.length === 3 ? 'grid-cols-2' : 'grid-cols-2'
            }`}>
              {post.images.slice(0, 4).map((imageUrl: string, index: number) => (
                <div
                  key={index}
                  className={`relative bg-gray-100 ${
                    post.images.length === 3 && index === 0 ? 'row-span-2' : ''
                  }`}
                >
                  <img
                    src={imageUrl}
                    alt={`Ảnh bài viết ${index + 1}`}
                    className="w-full h-full object-cover cursor-pointer hover:opacity-95 transition-opacity min-h-[200px] max-h-[400px]"
                    onClick={() => window.open(imageUrl, '_blank')}
                  />
                  {post.images.length > 4 && index === 3 && (
                    <div className="absolute inset-0 bg-black bg-opacity-60 flex items-center justify-center cursor-pointer">
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
          <div className="mb-3 -mx-3">
            {post.videos.map((videoUrl: string, index: number) => (
              <div key={index} className="bg-black">
                <video
                  src={videoUrl}
                  controls
                  className="w-full h-auto max-h-[500px]"
                  preload="metadata"
                >
                  Trình duyệt của bạn không hỗ trợ video.
                </video>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Stats */}
      {(post.stats?.likes > 0 || post.stats?.comments > 0 || post.stats?.shares > 0) && (
        <div className="px-3 py-2 border-t border-gray-100">
          <div className="flex items-center justify-between text-xs text-gray-500">
            <div className="flex items-center space-x-1">
              {post.stats?.likes > 0 && (
                <>
                  <div className="flex -space-x-1">
                    <div className="w-4 h-4 bg-blue-500 rounded-full flex items-center justify-center">
                      <ThumbsUp className="w-2.5 h-2.5 text-white fill-current" />
                    </div>
                    <div className="w-4 h-4 bg-red-500 rounded-full flex items-center justify-center">
                      <Heart className="w-2.5 h-2.5 text-white fill-current" />
                    </div>
                  </div>
                  <span>{formatStats(post.stats.likes)}</span>
                </>
              )}
            </div>
            <div className="flex items-center space-x-3">
              {post.stats?.comments > 0 && (
                <button
                  onClick={toggleComments}
                  className="hover:underline"
                >
                  {formatStats(post.stats.comments)} bình luận
                </button>
              )}
              {post.stats?.shares > 0 && (
                <span>{formatStats(post.stats.shares)} chia sẻ</span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Action Buttons - Enhanced with Reactions */}
      <div className="border-t border-gray-100">
        <div className="flex">
          {/* Enhanced Reaction Button */}
          <div className="flex-1 flex justify-center">
            <ReactionButton
              onReactionClick={handleReactionClick}
              onReactionRemove={handleReactionRemove}
              currentReaction={currentReaction}
              reactionCounts={reactionCounts}
              disabled={isLoadingInteraction}
              size="md"
              showPicker={true}
            />
          </div>

          <button
            onClick={toggleComments}
            className="flex-1 flex items-center justify-center py-2 px-3 text-gray-600 hover:bg-gray-50 transition-colors"
          >
            <MessageCircle className="h-4 w-4 mr-2" />
            <span className="text-sm font-medium vietnamese-text">Bình luận</span>
          </button>

          <button
            onClick={() => handleInteraction('share')}
            disabled={isLoadingInteraction}
            className="flex-1 flex items-center justify-center py-2 px-3 text-gray-600 hover:bg-gray-50 transition-colors"
          >
            <Share className="h-4 w-4 mr-2" />
            <span className="text-sm font-medium vietnamese-text">Chia sẻ</span>
          </button>
        </div>
      </div>

      {/* Comments Section */}
      {showComments && (
        <div className="border-t border-gray-100 bg-gray-50">
          <div className="p-3">
            {/* Comment Form */}
            <form onSubmit={handleSubmitComment} className="mb-3">
              <div className="flex space-x-2">

                    <Avatar
                        id={ user?.id}
                        src={ user?.avatarUrl || '/default-avatar.png'}
                        alt={ user?.fullName ||  user?.username || 'Avatar'}
                        size="md"
                        />

                <div className="flex-1">
                  <Textarea
                    value={commentText}
                    onChange={(e) => setCommentText(e.target.value)}
                    placeholder="Viết bình luận..."
                    className="min-h-[32px] text-sm bg-gray-300 text-black border-0 rounded-full px-3 py-2 resize-none vietnamese-text"
                    disabled={isSubmittingComment}
                  />
                  {commentText.trim() && (
                    <div className="flex justify-end mt-1">
                      <Button
                        type="submit"
                        size="sm"
                        disabled={isSubmittingComment}
                        className="text-xs px-3 py-1"
                      >
                        {isSubmittingComment ? <LoadingSpinner size="sm" /> : 'Gửi'}
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            </form>

            {/* Comments List - Enhanced with CommentItem */}
            {isLoadingComments ? (
              <div className="flex justify-center py-4">
                <LoadingSpinner size="sm" />
              </div>
            ) : (
              <div className="max-h-80 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-gray-100">
                <div className="space-y-3 pr-2">
                  {comments.length === 0 ? (
                    <p className="text-gray-500 text-sm text-center py-8 vietnamese-text">
                      Chưa có bình luận nào. Hãy là người đầu tiên bình luận!
                    </p>
                  ) : (
                    comments.map((comment) => (
                      <CommentItem
                        key={comment.id}
                        comment={comment}
                        postId={post.id}
                        onCommentUpdate={(updatedComment) => {
                          setComments(prev =>
                            prev.map(c => c.id === updatedComment.id ? updatedComment : c)
                          );
                        }}
                        onCommentDelete={(commentId) => {
                          setComments(prev => prev.filter(c => c.id !== commentId));
                          onPostUpdate?.({
                            ...post,
                            stats: { ...post.stats, comments: Math.max(post.stats.comments - 1, 0) }
                          });
                        }}
                        depth={0}
                      />
                    ))
                  )}
                </div>

                {/* Load More Comments Button */}
                {comments.length > 0 && (
                  <div className="text-center pt-3 mt-3 border-t border-gray-200">
                    <button className="text-sm text-blue-600 hover:text-blue-700 font-medium transition-colors vietnamese-text">
                      Xem thêm bình luận
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Action Feedback */}
      {actionFeedback && (
        <div className="fixed bottom-4 left-1/2 transform -translate-x-1/2 bg-gray-800 text-white px-4 py-2 rounded-lg shadow-lg z-50 vietnamese-text">
          {actionFeedback}
        </div>
      )}

      {/* Edit Post Modal */}
      {showEditModal && (
        <PostEditModal
          isOpen={showEditModal}
          onClose={() => setShowEditModal(false)}
          post={post}
          onSave={handlePostSave}
        />
      )}
    </Card>
  );
};
