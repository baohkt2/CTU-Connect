/* eslint-disable @typescript-eslint/no-explicit-any */
'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { CreateCommentRequest, UpdatePostRequest } from '@/types';
import { postService } from '@/services/postService';
import { Button } from '@/components/ui/Button';
import Card from '@/components/ui/Card';
import { Textarea } from '@/components/ui/Textarea';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { PostMenu } from '@/components/post/PostMenu';
import { PostEditModal } from '@/components/post/PostEditModal';
import { EditIndicator } from '@/components/post/EditIndicator';
import { CommentItem } from '@/components/post/CommentItem';
import { CommentManager } from '@/utils/commentManager';
import { formatTimeAgo } from '@/utils/localization';
import { sanitizeHtml, prepareHtmlForDisplay } from '@/utils/richTextUtils';
import { REACTIONS } from '@/components/ui/ReactionPicker';
import {
  MessageCircle,
  Share,
  Globe,
  Users,
  Lock,
  ThumbsUp,
  Heart,
  MoreHorizontal,
  Flag,
  Trash2,
  EyeOff,
  FileText,
  Download,
  BookmarkIcon,
  Send,
  Image as ImageIcon,
  Smile
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
  const [isLiked, setIsLiked] = useState<boolean | null>(null);
  const [isBookmarked, setIsBookmarked] = useState<boolean | null>(null);
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
  const [showReactionPicker, setShowReactionPicker] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);

  const isOwnPost = user?.id === post.authorId || user?.id === post.author?.id;
  const shouldTruncate = post.content && post.content.length > 300;
  const displayContent = shouldTruncate && !isExpanded
    ? post.content.substring(0, 300) + '...'
    : post.content;

  // Load trạng thái like, bookmark khi mount
  useEffect(() => {
    let mounted = true;

    const loadInteractionStatus = async () => {
      if (!user?.id || !post.id) return;

      try {
        const status = await postService.getInteractionStatus(post.id);
        if (mounted) {
          setIsLiked(status.hasLiked);
          setIsBookmarked(status.hasBookmarked);
        }
      } catch (error) {
        console.debug('Không thể tải trạng thái tương tác:', error);
        if (mounted) {
          setIsLiked(false);
          setIsBookmarked(false);
        }
      }
    };

    loadInteractionStatus();
    return () => { mounted = false; };
  }, [post.id, user?.id]);

  // Show feedback message with improved styling
  const showFeedback = (message: string) => {
    setActionFeedback(message);
    setTimeout(() => setActionFeedback(null), 3000);
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
        try {
          const legacyResponse = await postService.getCommentsLegacy(post.id);
          setComments(legacyResponse.content || []);
        } catch (legacyError) {
          console.error('Cả hai endpoint đều thất bại:', legacyError);
          showFeedback('Không thể tải bình luận');
        }
      } finally {
        setIsLoadingComments(false);
      }
    }
    setShowComments(v => !v);
  }, [showComments, comments.length, post.id]);

  // Chức năng tương tác
  const handleInteraction = useCallback(async (type: 'like' | 'bookmark' | 'share') => {
    if (isLoadingInteraction) return;
    setIsLoadingInteraction(true);

    try {
      if (type === 'like') {
        const previousLiked = isLiked;
        const newLiked = !isLiked;
        setIsLiked(newLiked);

        try {
          await postService.toggleLike(post.id);
          const newLikes = newLiked ? (post.stats?.likes || 0) + 1 : Math.max((post.stats?.likes || 0) - 1, 0);
          onPostUpdate?.({
            ...post,
            stats: { ...post.stats, likes: newLikes }
          });
          showFeedback(newLiked ? 'Đã thích bài viết' : 'Đã bỏ thích');
        } catch (error) {
          setIsLiked(previousLiked);
          throw error;
        }
      } else if (type === 'bookmark') {
        const previousBookmarked = isBookmarked;
        const newBookmarked = !isBookmarked;
        setIsBookmarked(newBookmarked);

        try {
          await postService.toggleBookmark(post.id);
          const newBookmarks = newBookmarked ? (post.stats?.bookmarks || 0) + 1 : Math.max((post.stats?.bookmarks || 0) - 1, 0);
          onPostUpdate?.({
            ...post,
            stats: { ...post.stats, bookmarks: newBookmarks }
          });
          showFeedback(newBookmarked ? 'Đã lưu bài viết' : 'Đã bỏ lưu bài viết');
        } catch (error) {
          setIsBookmarked(previousBookmarked);
          throw error;
        }
      } else if (type === 'share') {
        await postService.sharePost(post.id);
        onPostUpdate?.({
          ...post,
          stats: { ...post.stats, shares: (post.stats?.shares || 0) + 1 }
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
  }, [isLoadingInteraction, isLiked, isBookmarked, onPostUpdate, post]);

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
        stats: { ...post.stats, comments: (post.stats.comments || 0) + 1 }
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

  // Handle comment actions (report, delete, hide)
  const handleCommentAction = useCallback(async (action: 'report' | 'delete' | 'hide', commentId: string) => {
    try {
      switch (action) {
        case 'delete':
          if (window.confirm('Bạn có chắc chắn muốn xóa bình luận này?')) {
            await postService.deleteComment(commentId, post.id);
            setComments(prev => prev.filter(comment => comment.id !== commentId));
            onPostUpdate?.({
              ...post,
              stats: { ...post.stats, comments: Math.max((post.stats.comments || 1) - 1, 0) }
            });
            showFeedback('Đã xóa bình luận');
          }
          break;
        case 'report':
          showFeedback('Đã báo cáo bình luận');
          break;
        case 'hide':
          setComments(prev => prev.filter(comment => comment.id !== commentId));
          showFeedback('Đã ẩn bình luận');
          break;
      }
      // Close the menu after action
      setCommentMenus(prev => ({
        ...prev,
        [commentId]: false
      }));
    } catch (error) {
      console.error(`Error ${action} comment:`, error);
      showFeedback(`Không thể ${action === 'delete' ? 'xóa' : action === 'report' ? 'báo cáo' : 'ẩn'} bình luận`);
    }
  }, [onPostUpdate, post]);

  // Reaction handlers
  const handleReactionClick = useCallback(async (reactionId: string) => {
    if (isLoadingInteraction) return;
    setIsLoadingInteraction(true);

    try {
      // Call API to save reaction
      await postService.reactToPost(post.id, reactionId);
      
      // Update local state
      setCurrentReaction(reactionId);
      setIsLiked(true); // Mark as reacted
      setShowReactionPicker(false); // Close picker after selection
      
      // Update reaction counts
      setReactionCounts(prev => ({
        ...prev,
        [reactionId]: (prev[reactionId] || 0) + 1
      }));
      
      // Update post stats
      onPostUpdate?.({
        ...post,
        stats: { 
          ...post.stats, 
          likes: (post.stats?.likes || 0) + 1,
          reactions: {
            ...post.stats?.reactions,
            [reactionId]: ((post.stats?.reactions?.[reactionId] || 0) + 1)
          }
        }
      });
      
      const reactionName = REACTIONS.find(r => r.id === reactionId)?.name || 'phản ứng';
      showFeedback(`Đã ${reactionName.toLowerCase()}`);
    } catch (error) {
      console.error('Error adding reaction:', error);
      showFeedback('Không thể thêm phản ứng');
    } finally {
      setIsLoadingInteraction(false);
    }
  }, [isLoadingInteraction, post, onPostUpdate]);

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
    showFeedback('Đã báo cáo bài viết');
  };

  const handlePostHide = () => {
    showFeedback('Đã ẩn bài viết');
  };

  const handlePostBlock = () => {
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
    <Card className={`
      bg-white rounded-xl shadow-sm hover:shadow-lg 
      transition-all duration-300 ease-out
      border border-gray-100 hover:border-gray-200
      mb-6 overflow-hidden group
      transform hover:scale-[1.01]
      ${className}
    `}>
      {/* Enhanced Feedback Toast */}
      {actionFeedback && (
        <div className="absolute top-4 right-4 z-50 bg-green-600 text-white px-4 py-2 rounded-lg text-sm font-medium shadow-lg animate-in slide-in-from-top-2 duration-300">
          <div className="flex items-center gap-2">
            <div className="w-1.5 h-1.5 bg-white rounded-full animate-pulse"></div>
            {actionFeedback}
          </div>
        </div>
      )}
      
      {/* Enhanced Header */}
      <div className="p-4 pb-2">
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3 flex-1">
            {/* Enhanced Avatar */}
            <div className="relative">
              <Avatar
                id={post.author?.id}
                src={post.author?.avatarUrl || '/default-avatar.png'}
                alt={post.author?.fullName || post.author?.username || 'Avatar'}
                size="md"
                className="ring-2 ring-gray-100 hover:ring-blue-200 transition-all duration-200"
              />
              {/* Online status indicator */}
              <div className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-green-400 border-2 border-white rounded-full"></div>
            </div>

            {/* Enhanced User Info */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <h3 className="font-semibold text-gray-900 hover:text-blue-600 cursor-pointer transition-colors text-sm">
                  {post.author?.fullName || post.author?.username || 'Người dùng'}
                </h3>
                {post.author?.verified && (
                  <div className="w-4 h-4 bg-blue-500 rounded-full flex items-center justify-center">
                    <svg className="w-2.5 h-2.5 text-white" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  </div>
                )}
              </div>
              <div className="flex items-center gap-2 text-xs text-gray-500 mt-0.5">
                <time className="hover:text-gray-700 cursor-pointer transition-colors">
                  {formatTimeAgo(post.createdAt)}
                </time>
                <span>•</span>
                <div className="flex items-center gap-1 hover:text-gray-700 cursor-pointer transition-colors">
                  {getPrivacyIcon()}
                  <span>{getPrivacyText()}</span>
                </div>
                {post.isEdited && (
                  <>
                    <span>•</span>
                    <EditIndicator />
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Enhanced Post Menu */}
          <PostMenu
            post={post}
            isOwnPost={isOwnPost}
            onEdit={handlePostEdit}
            onDelete={handlePostDelete}
            onReport={handlePostReport}
            onHide={handlePostHide}
            onBlock={handlePostBlock}
            onCopyLink={handleCopyLink}
            className="opacity-0 group-hover:opacity-100 transition-opacity duration-200"
          />
        </div>
      </div>

      {/* Enhanced Content */}
      <div className="px-4 pb-3">
        {post.content && (
          <div className="prose prose-sm max-w-none">
            <div
              className="text-gray-800 leading-relaxed whitespace-pre-wrap"
              dangerouslySetInnerHTML={{
                __html: prepareHtmlForDisplay(displayContent)
              }}
            />
            {shouldTruncate && (
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="text-blue-600 hover:text-blue-700 font-medium text-sm mt-2 transition-colors"
              >
                {isExpanded ? 'Thu gọn' : 'Xem thêm'}
              </button>
            )}
          </div>
        )}

        {/* Enhanced Media Section */}
        {(post.images?.length > 0 || post.attachments?.length > 0) && (
          <div className="mt-4 space-y-3">
            {/* Images Grid */}
            {post.images?.length > 0 && (
              <div className={`
                grid gap-2 rounded-lg overflow-hidden
                ${post.images.length === 1 ? 'grid-cols-1' : 
                  post.images.length === 2 ? 'grid-cols-2' :
                  post.images.length === 3 ? 'grid-cols-2' : 'grid-cols-2'}
              `}>
                {post.images.map((image: any, index: number) => (
                  <div
                    key={index}
                    className={`
                      relative group cursor-pointer overflow-hidden rounded-lg
                      ${post.images.length === 3 && index === 0 ? 'row-span-2' : ''}
                      hover:opacity-95 transition-opacity
                    `}
                  >
                    <img
                      src={image.url || image}
                      alt={`Hình ảnh ${index + 1}`}
                      className="w-full h-auto object-cover max-h-96 hover:scale-105 transition-transform duration-300"
                      loading="lazy"
                    />
                    <div className="absolute inset-0 bg-black opacity-0 group-hover:opacity-10 transition-opacity"></div>
                  </div>
                ))}
              </div>
            )}

            {/* Attachments */}
            {post.attachments?.length > 0 && (
              <div className="space-y-2">
                {post.attachments.map((attachment: any, index: number) => (
                  <div key={index} className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors group">
                    <FileText className="h-5 w-5 text-blue-500 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">
                        {attachment.name || attachment.filename}
                      </p>
                      {attachment.size && (
                        <p className="text-xs text-gray-500">
                          {(attachment.size / 1024 / 1024).toFixed(1)} MB
                        </p>
                      )}
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Enhanced Stats Bar */}
      {(post.stats?.likes > 0 || post.stats?.comments > 0 || post.stats?.shares > 0) && (
        <div className="px-4 py-2 border-t border-gray-100">
          <div className="flex items-center justify-between text-sm text-gray-500">
            <div className="flex items-center gap-4">
              {post.stats?.likes > 0 && (
                <div className="flex items-center gap-1 hover:text-blue-600 cursor-pointer transition-colors">
                  <div className="flex -space-x-1">
                    <div className="w-5 h-5 bg-blue-500 rounded-full flex items-center justify-center">
                      <ThumbsUp className="w-3 h-3 text-white" />
                    </div>
                    <div className="w-5 h-5 bg-red-500 rounded-full flex items-center justify-center">
                      <Heart className="w-3 h-3 text-white" />
                    </div>
                  </div>
                  <span>{formatStats(post.stats.likes)}</span>
                </div>
              )}
            </div>
            <div className="flex items-center gap-4">
              {post.stats?.comments > 0 && (
                <button
                  onClick={toggleComments}
                  className="hover:text-blue-600 transition-colors"
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

      {/* Enhanced Action Bar */}
      <div className="px-4 py-3 border-t border-gray-100">
        <div className="flex items-center justify-around">
          {/* Like Button with Reaction Picker */}
          <div 
            className="relative"
            onMouseEnter={() => setShowReactionPicker(true)}
            onMouseLeave={() => setShowReactionPicker(false)}
          >
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleInteraction('like')}
              disabled={isLiked === null || isLoadingInteraction}
              className={`
                flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200
                ${isLiked || currentReaction
                  ? 'text-blue-600 bg-blue-50 hover:bg-blue-100' 
                  : 'text-gray-700 hover:bg-gray-100 hover:text-blue-600'
                }
                ${isLoadingInteraction ? 'opacity-50 cursor-not-allowed' : ''}
              `}
            >
              {isLoadingInteraction ? (
                <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
              ) : currentReaction ? (
                // Show selected reaction
                <>
                  <span className="text-lg">{REACTIONS.find(r => r.id === currentReaction)?.emoji || '👍'}</span>
                  <span className="font-medium">{REACTIONS.find(r => r.id === currentReaction)?.name || 'Đã thích'}</span>
                </>
              ) : isLiked ? (
                // Show thumbs up if liked
                <>
                  <ThumbsUp className="h-4 w-4 fill-current" />
                  <span className="font-medium">Đã thích</span>
                </>
              ) : (
                // Default state
                <>
                  <ThumbsUp className="h-4 w-4" />
                  <span className="font-medium">Thích</span>
                </>
              )}
            </Button>

            {/* Reaction Picker */}
            {showReactionPicker && (
              <div
                className="absolute bottom-full left-0 flex items-center gap-1 bg-white border border-gray-200 rounded-full px-2 py-1 shadow-lg animate-in fade-in-50 slide-in-from-bottom-2 duration-200 z-50"
              >
                {REACTIONS.map((reaction) => (
                  <button
                    key={reaction.id}
                    className="w-8 h-8 rounded-full hover:scale-125 transition-transform duration-150"
                    onClick={() => handleReactionClick(reaction.id)}
                    title={reaction.name}
                  >
                    <span className="text-lg">{reaction.emoji}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Comment Button */}
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleComments}
            className="flex items-center gap-2 px-4 py-2 text-gray-700 hover:bg-gray-100 hover:text-blue-600 rounded-lg transition-all duration-200"
          >
            <MessageCircle className="h-4 w-4" />
            <span className="font-medium">Bình luận</span>
          </Button>

          {/* Share Button */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => handleInteraction('share')}
            disabled={isLoadingInteraction}
            className="flex items-center gap-2 px-4 py-2 text-gray-700 hover:bg-gray-100 hover:text-green-600 rounded-lg transition-all duration-200"
          >
            <Share className="h-4 w-4" />
            <span className="font-medium">Chia sẻ</span>
          </Button>

          {/* Bookmark Button */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => handleInteraction('bookmark')}
            disabled={isBookmarked === null || isLoadingInteraction}
            className={`
              flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200
              ${isBookmarked 
                ? 'text-yellow-600 bg-yellow-50 hover:bg-yellow-100' 
                : 'text-gray-700 hover:bg-gray-100 hover:text-yellow-600'
              }
            `}
          >
            <BookmarkIcon className={`h-4 w-4 ${isBookmarked ? 'fill-current' : ''}`} />
            <span className="font-medium">{isBookmarked ? 'Đã lưu' : 'Lưu'}</span>
          </Button>
        </div>
      </div>

      {/* Enhanced Comments Section */}
      {showComments && (
        <div className="border-t border-gray-100 animate-in slide-in-from-top-2 duration-300">
          {/* Comment Input */}
          <div className="p-4 bg-gray-50">
            <form onSubmit={handleSubmitComment} className="flex gap-3">
              <Avatar
                id={user?.id}
                src={user?.avatarUrl || '/default-avatar.png'}
                alt={user?.fullName || user?.username || 'Your avatar'}
                size="sm"
                className="flex-shrink-0"
              />
              <div className="flex-1">
                <div className="relative">
                  <Textarea
                    value={commentText}
                    onChange={(e) => setCommentText(e.target.value)}
                    placeholder="Viết bình luận..."
                    className="min-h-[40px] max-h-32 resize-none border-gray-200 focus:border-blue-400 focus:ring-blue-400 rounded-lg pr-12 text-sm"
                    rows={1}
                  />
                  <div className="absolute right-2 bottom-2 flex items-center gap-1">
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="p-1 h-6 w-6 text-gray-400 hover:text-gray-600"
                    >
                      <Smile className="h-4 w-4" />
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="p-1 h-6 w-6 text-gray-400 hover:text-gray-600"
                    >
                      <ImageIcon className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
                <div className="flex items-center justify-between mt-2">
                  <div className="text-xs text-gray-500">
                    Nhấn Enter để gửi
                  </div>
                  <Button
                    type="submit"
                    size="sm"
                    disabled={!commentText.trim() || isSubmittingComment}
                    className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-1.5 rounded-full text-xs font-medium transition-all duration-200 disabled:opacity-50"
                  >
                    {isSubmittingComment ? (
                      <div className="w-3 h-3 border border-white border-t-transparent rounded-full animate-spin"></div>
                    ) : (
                      <>
                        <Send className="h-3 w-3 mr-1" />
                        Gửi
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </form>
          </div>

          {/* Comments List */}
          <div className="max-h-96 overflow-y-auto">
            {isLoadingComments ? (
              <div className="flex items-center justify-center py-8">
                <LoadingSpinner size="sm" />
                <span className="ml-2 text-sm text-gray-500">Đang tải bình luận...</span>
              </div>
            ) : comments.length > 0 ? (
              <div className="space-y-1">
                {comments.map((comment) => (
                  <CommentItem
                    key={comment.id}
                    comment={comment}
                    postId={post.id}
                    isOwnComment={comment.authorId === user?.id || comment.author?.id === user?.id}
                    onDelete={() => handleCommentAction('delete', comment.id)}
                    onReport={() => handleCommentAction('report', comment.id)}
                    onHide={() => handleCommentAction('hide', comment.id)}
                    onCommentUpdate={(updatedComment) => {
                      setComments(prev => prev.map(c => c.id === updatedComment.id ? updatedComment : c));
                      // Update post comment count if reply count changed
                      const oldTotalReplies = CommentManager.getTotalRepliesCount(comment);
                      const newTotalReplies = CommentManager.getTotalRepliesCount(updatedComment);
                      const countDiff = newTotalReplies - oldTotalReplies;

                      if (countDiff !== 0) {
                        onPostUpdate?.({
                          ...post,
                          stats: { ...post.stats, comments: Math.max((post.stats.comments || 0) + countDiff, 0) }
                        });
                      }
                    }}
                    onRepliesUpdate={(parentId, replies) => {
                      setComments(prev => prev.map(c => {
                        if (c.id === parentId) {
                          return { ...c, replies, replyCount: replies.length };
                        }
                        return c;
                      }));
                    }}
                    depth={0}
                    className="hover:bg-gray-50/50 transition-colors duration-200 rounded-lg"
                  />
                ))}
              </div>
            ) : (
              <div className="py-8 text-center">
                <MessageCircle className="h-8 w-8 text-gray-300 mx-auto mb-2" />
                <p className="text-sm text-gray-500">Chưa có bình luận nào</p>
                <p className="text-xs text-gray-400 mt-1">Hãy là người đầu tiên bình luận!</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Enhanced Edit Modal */}
      {showEditModal && (
        <PostEditModal
          post={post}
          onSave={handlePostSave}
          onCancel={() => setShowEditModal(false)}
        />
      )}
    </Card>
  );
};

// Helper functions for document handling
const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

const getDocumentType = (contentType: string): string => {
  switch (contentType) {
    case 'application/pdf':
      return 'PDF';
    case 'application/msword':
      return 'Word Document';
    case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
      return 'Word Document';
    default:
      return 'Document';
  }
};

// Simplified download function - direct popup to link
const handleDocumentDownload = async (document: any) => {
  try {
    // Direct popup to Cloudinary link
    window.open(document.url, '_blank');
  } catch (error) {
    console.error('Error opening document:', error);
    alert('Không thể mở tài liệu. Vui lòng thử lại sau.');
  }
};

// PostDocumentIcon component
const PostDocumentIcon: React.FC<{ document: any; className?: string }> = ({ document, className }) => {
  const getIconAndColor = () => {
    const contentType = document.contentType;
    if (contentType?.includes('pdf')) {
      return { icon: <FileText className={className} />, color: 'text-red-600 bg-red-50' };
    } else if (contentType?.includes('word') || contentType?.includes('document')) {
      return { icon: <FileText className={className} />, color: 'text-blue-600 bg-blue-50' };
    }
    return { icon: <FileText className={className} />, color: 'text-gray-600 bg-gray-50' };
  };

  const { icon, color } = getIconAndColor();

  return (
    <div className={`flex items-center justify-center w-10 h-10 rounded-full ${color}`}>
      {icon}
    </div>
  );
};
