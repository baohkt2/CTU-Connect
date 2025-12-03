'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Comment, CreateCommentRequest } from '@/types';
import { postService } from '@/services/postService';
import { useAuth } from '@/contexts/AuthContext';
import { CommentManager } from '@/utils/commentManager';
import Avatar from '@/components/ui/Avatar';
import { Button } from '@/components/ui/Button';
import { Textarea } from '@/components/ui/Textarea';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { formatTimeAgo } from '@/utils/localization';
import {
  MoreHorizontal,
  Reply,
  Flag,
  Trash2,
  EyeOff,
  ChevronDown,
  ChevronUp,
  ThumbsUp,
  Send,
  Smile
} from 'lucide-react';

interface CommentItemProps {
  comment: Comment;
  postId?: string;
  isOwnComment?: boolean;
  onDelete?: () => void;
  onReport?: () => void;
  onHide?: () => void;
  onCommentUpdate?: (comment: Comment) => void;
  onRepliesUpdate?: (parentId: string, replies: Comment[]) => void;
  depth?: number;
  className?: string;
}

export const CommentItem: React.FC<CommentItemProps> = ({
  comment: initialComment,
  postId,
  isOwnComment = false,
  onDelete,
  onReport,
  onHide,
  onCommentUpdate,
  onRepliesUpdate,
  depth = 0,
  className = ''
}) => {
  const { user } = useAuth();
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Local state management with proper normalization
  const [comment, setComment] = useState(() => CommentManager.normalizeComment(initialComment));
  const [showReplyForm, setShowReplyForm] = useState(false);
  const [replyText, setReplyText] = useState('');
  const [isSubmittingReply, setIsSubmittingReply] = useState(false);
  const [isLoadingReplies, setIsLoadingReplies] = useState(false);
  const [showMenu, setShowMenu] = useState(false);
  const [isLiked, setIsLiked] = useState(false);
  const [isLiking, setIsLiking] = useState(false);

  // Separate state for replies visibility to avoid conflicts
  const [showReplies, setShowReplies] = useState(false);

  // Computed values
  const canReply = CommentManager.canAddReply(comment);
  const hasReplies = comment.replyCount > 0 || (comment.replies && comment.replies.length > 0);
  const indentWidth = Math.min(depth * 20, 60); // Max 60px indent
  const authorDisplayName = CommentManager.getAuthorDisplayName(comment.author);
  const authorDisplayAvatar = CommentManager.getAuthorDisplayAvatar(comment.author);

  // Update local state when prop changes but preserve showReplies state
  useEffect(() => {
    const normalizedComment = CommentManager.normalizeComment(initialComment);
    setComment(normalizedComment);
    // Don't reset showReplies here to avoid flickering
  }, [initialComment]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current && showReplyForm) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [replyText, showReplyForm]);

  // Load replies from server
  const loadReplies = async () => {
    if (!postId || isLoadingReplies || (comment.replies && comment.replies.length > 0)) {
      return;
    }

    setIsLoadingReplies(true);
    try {
      const loadedReplies = await postService.getCommentReplies(comment.id, postId);
      const normalizedReplies = loadedReplies.map(CommentManager.normalizeComment);

      const updatedComment = { ...comment, replies: normalizedReplies, showReplies: true };
      setComment(updatedComment);
      onCommentUpdate?.(updatedComment);
      onRepliesUpdate?.(comment.id, normalizedReplies);
    } catch (error) {
      console.error('Error loading replies:', error);
    } finally {
      setIsLoadingReplies(false);
    }
  };

  // Toggle replies visibility
  const handleToggleReplies = async () => {
    if (!hasReplies) return;

    // Sử dụng state riêng showReplies thay vì comment.showReplies
    if (showReplies) {
      setShowReplies(false);
      return;
    }

    // Nếu chưa có data thì load
    if (!comment.replies || comment.replies.length === 0) {
      setIsLoadingReplies(true);
      try {
        const loadedReplies = await postService.getCommentReplies(comment.id, postId!);
        const normalizedReplies = loadedReplies.map(CommentManager.normalizeComment);

        const updatedComment = { ...comment, replies: normalizedReplies };
        setComment(updatedComment);
        setShowReplies(true); // Set state riêng

        onCommentUpdate?.(updatedComment);
        onRepliesUpdate?.(comment.id, normalizedReplies);
      } catch (error) {
        console.error('Error loading replies:', error);
      } finally {
        setIsLoadingReplies(false);
      }
    } else {
      // Nếu đã có data thì chỉ cần show
      setShowReplies(true);
    }
  };

  // Handle reply submission with proper structure
  const handleSubmitReply = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!replyText.trim() || isSubmittingReply || !postId) return;

    setIsSubmittingReply(true);
    try {
      const replyData: CreateCommentRequest = {
        content: replyText.trim(),
        parentId: comment.id
      };

      const newReply = await postService.createComment(postId, replyData);
      const normalizedReply = CommentManager.normalizeComment({
        ...newReply,
        depth: (comment.depth || 0) + 1,
        replyToAuthor: authorDisplayName
      });

      // Update local comment state
      const updatedReplies = [...(comment.replies || []), normalizedReply];
      const updatedComment = {
        ...comment,
        replies: updatedReplies,
        replyCount: (comment.replyCount || 0) + 1,
        showReplies: true
      };

      setComment(updatedComment);
      setReplyText('');
      setShowReplyForm(false);

      // Notify parent components
      onCommentUpdate?.(updatedComment);
      onRepliesUpdate?.(comment.id, updatedReplies);
    } catch (error) {
      console.error('Error creating reply:', error);
    } finally {
      setIsSubmittingReply(false);
    }
  };

  // Handle comment like with optimistic updates
  const handleLike = async () => {
    if (isLiking || !postId) return;

    setIsLiking(true);
    const previousLiked = isLiked;
    const previousCount = comment.likesCount || 0;

    // Optimistic update
    const newLiked = !isLiked;
    const newCount = newLiked ? previousCount + 1 : Math.max(previousCount - 1, 0);

    setIsLiked(newLiked);
    const updatedComment = { ...comment, likesCount: newCount };
    setComment(updatedComment);

    try {
      await postService.toggleCommentLike(comment.id, postId);
      onCommentUpdate?.(updatedComment);
    } catch (error) {
      // Revert on error
      setIsLiked(previousLiked);
      const revertedComment = { ...comment, likesCount: previousCount };
      setComment(revertedComment);
      console.error('Error toggling comment like:', error);
    } finally {
      setIsLiking(false);
    }
  };

  // Handle menu actions
  const handleMenuAction = (action: 'report' | 'delete' | 'hide') => {
    setShowMenu(false);
    switch (action) {
      case 'report':
        onReport?.();
        break;
      case 'delete':
        onDelete?.();
        break;
      case 'hide':
        onHide?.();
        break;
    }
  };

  // Handle reply form toggle
  const handleReplyToggle = () => {
    setShowReplyForm(!showReplyForm);
    if (!showReplyForm) {
      setReplyText('');
      // Focus textarea after it renders
      setTimeout(() => textareaRef.current?.focus(), 100);
    }
  };

  return (
    <div
      className={`relative ${className}`}
      style={{ paddingLeft: `${indentWidth}px` }}
    >
      {/* Connecting line for nested replies */}
      {depth > 0 && (
        <div
          className="absolute left-2 top-0 bottom-0 w-0.5 bg-gray-200"
          style={{ left: `${indentWidth - 12}px` }}
        />
      )}

      <div className="flex gap-3 py-3 group">
        {/* Avatar */}
        <div className="flex-shrink-0">
          <Avatar
            id={comment.author.id}
            src={authorDisplayAvatar}
            alt={authorDisplayName}
            size="sm"
            className="ring-2 ring-gray-100 hover:ring-blue-200 transition-all duration-200"
          />
        </div>

        {/* Comment content */}
        <div className="flex-1 min-w-0">
          {/* Comment bubble */}
          <div className="bg-gray-100 hover:bg-gray-50 rounded-2xl px-4 py-3 inline-block max-w-full transition-colors duration-200">
            {/* Author name with verification badge */}
            <div className="flex items-center gap-2 mb-1">
              <span className="font-semibold text-sm text-gray-900 hover:text-blue-600 cursor-pointer transition-colors">
                {authorDisplayName}
              </span>
              {comment.author.verified && (
                <div className="w-3 h-3 bg-blue-500 rounded-full flex items-center justify-center">
                  <svg className="w-2 h-2 text-white" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                </div>
              )}
            </div>

            {/* Comment text with reply mention */}
            <div className="text-sm text-gray-800 leading-relaxed break-words whitespace-pre-wrap">
              {comment.replyToAuthor && comment.isFlattened && (
                <span className="text-blue-600 font-medium mr-1 hover:text-blue-700 cursor-pointer">
                  @{comment.replyToAuthor}
                </span>
              )}
              {comment.content}
            </div>
          </div>

          {/* Comment actions */}
          <div className="flex items-center gap-4 mt-1.5 ml-2">
            {/* Timestamp */}
            <span className="text-xs text-gray-500 hover:text-gray-700 cursor-pointer transition-colors">
              {formatTimeAgo(comment.createdAt)}
            </span>

            {/* Like button */}
            <button
              onClick={handleLike}
              disabled={isLiking}
              className={`flex items-center gap-1 text-xs font-medium transition-all duration-200 ${
                isLiked 
                  ? 'text-blue-600 hover:text-blue-700' 
                  : 'text-gray-500 hover:text-blue-600'
              } disabled:opacity-50`}
            >
              <ThumbsUp className={`h-3 w-3 transition-all duration-200 ${
                isLiked ? 'fill-current scale-110' : 'hover:scale-110'
              }`} />
              {comment.likesCount > 0 && (
                <span className="font-medium">{comment.likesCount}</span>
              )}
              <span>{isLiked ? 'Đã thích' : 'Thích'}</span>
            </button>

            {/* Reply button */}
            {canReply && (
              <button
                onClick={handleReplyToggle}
                className="flex items-center gap-1 text-xs font-medium text-gray-500 hover:text-blue-600 transition-colors duration-200"
              >
                <Reply className="h-3 w-3" />
                Trả lời
              </button>
            )}

            {/* Menu button */}
            <div className="relative">
              <button
                onClick={() => setShowMenu(!showMenu)}
                className="p-1 rounded-full hover:bg-gray-200 text-gray-400 hover:text-gray-600 transition-all duration-200 opacity-0 group-hover:opacity-100"
              >
                <MoreHorizontal className="h-3 w-3" />
              </button>

              {/* Dropdown menu */}
              {showMenu && (
                <div className="absolute right-0 top-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg py-1 z-50 min-w-[120px]">
                  {isOwnComment ? (
                    <button
                      onClick={() => handleMenuAction('delete')}
                      className="flex items-center gap-2 w-full px-3 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors"
                    >
                      <Trash2 className="h-3 w-3" />
                      Xóa
                    </button>
                  ) : (
                    <>
                      <button
                        onClick={() => handleMenuAction('report')}
                        className="flex items-center gap-2 w-full px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                      >
                        <Flag className="h-3 w-3" />
                        Báo cáo
                      </button>
                      <button
                        onClick={() => handleMenuAction('hide')}
                        className="flex items-center gap-2 w-full px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                      >
                        <EyeOff className="h-3 w-3" />
                        Ẩn
                      </button>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Reply form */}
          {showReplyForm && canReply && (
            <div className="mt-3 ml-2 animate-in slide-in-from-top-2 duration-200">
              <form onSubmit={handleSubmitReply} className="flex gap-3">
                <Avatar
                  id={user?.id}
                  src={user?.avatarUrl || '/default-avatar.png'}
                  alt={user?.fullName || user?.username || 'Your avatar'}
                  size="sm"
                  className="flex-shrink-0"
                />
                <div className="flex-1">
                  <Textarea
                    ref={textareaRef}
                    value={replyText}
                    onChange={(e) => setReplyText(e.target.value)}
                    placeholder={`Trả lời ${authorDisplayName}...`}
                    className="min-h-[40px] max-h-32 text-sm bg-white border border-gray-200 focus:border-blue-400 focus:ring-blue-400 rounded-lg px-3 py-2 resize-none transition-all duration-200"
                    disabled={isSubmittingReply}
                    rows={1}
                  />
                  <div className="flex justify-between items-center mt-2">
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        className="p-1.5 rounded-full hover:bg-gray-100 text-gray-400 hover:text-gray-600 transition-all duration-200"
                      >
                        <Smile className="h-4 w-4" />
                      </button>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={handleReplyToggle}
                        className="text-xs px-3 py-1 h-7"
                      >
                        Hủy
                      </Button>
                      <Button
                        type="submit"
                        size="sm"
                        disabled={!replyText.trim() || isSubmittingReply}
                        className="text-xs px-3 py-1 h-7 bg-blue-600 hover:bg-blue-700 disabled:opacity-50"
                      >
                        {isSubmittingReply ? (
                          <LoadingSpinner size="sm" />
                        ) : (
                          <>
                            <Send className="h-3 w-3 mr-1" />
                            Gửi
                          </>
                        )}
                      </Button>
                    </div>
                  </div>
                </div>
              </form>
            </div>
          )}

          {/* Replies toggle */}
          {hasReplies && (
            <div className="mt-2 ml-2">
              <button
                onClick={handleToggleReplies}
                disabled={isLoadingReplies}
                className="flex items-center gap-1 text-xs font-medium text-blue-600 hover:text-blue-700 transition-colors duration-200 disabled:opacity-50"
              >
                {isLoadingReplies ? (
                  <LoadingSpinner size="sm" />
                ) : comment.showReplies ? (
                  <ChevronUp className="h-3 w-3" />
                ) : (
                  <ChevronDown className="h-3 w-3" />
                )}
                <span>
                  {comment.showReplies
                    ? 'Ẩn phản hồi'
                    : `Xem ${comment.replyCount} phản hồi`
                  }
                </span>
              </button>
            </div>
          )}

          {/* Nested replies with improved structure */}
          {showReplies && comment.replies && comment.replies.length > 0 && (
            <div className="mt-2 space-y-1 animate-in slide-in-from-top-2 duration-300">
              {comment.replies.map((reply) => (
                <CommentItem
                  key={reply.id}
                  comment={reply}
                  postId={postId}
                  isOwnComment={user?.id === reply.author.id}
                  onDelete={() => {
                    const updatedReplies = comment.replies!.filter(r => r.id !== reply.id);
                    const updatedComment = {
                      ...comment,
                      replies: updatedReplies,
                      replyCount: Math.max((comment.replyCount || 1) - 1, 0)
                    };
                    setComment(updatedComment);
                    onCommentUpdate?.(updatedComment);
                    onRepliesUpdate?.(comment.id, updatedReplies);
                  }}
                  onReport={() => console.log('Report reply:', reply.id)}
                  onHide={() => {
                    const updatedReplies = comment.replies!.filter(r => r.id !== reply.id);
                    const updatedComment = { ...comment, replies: updatedReplies };
                    setComment(updatedComment);
                    onCommentUpdate?.(updatedComment);
                  }}
                  onCommentUpdate={(updatedReply) => {
                    const updatedReplies = comment.replies!.map(r =>
                      r.id === updatedReply.id ? updatedReply : r
                    );
                    const updatedComment = { ...comment, replies: updatedReplies };
                    setComment(updatedComment);
                    onCommentUpdate?.(updatedComment);
                    onRepliesUpdate?.(comment.id, updatedReplies);
                  }}
                  onRepliesUpdate={onRepliesUpdate}
                  depth={depth + 1}
                  className="border-l-2 border-gray-100 hover:border-blue-200 transition-colors duration-200"
                />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Click outside to close menu */}
      {showMenu && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setShowMenu(false)}
        />
      )}
    </div>
  );
};
