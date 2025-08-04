'use client';

import React, { useState } from 'react';
import { Comment, CreateCommentRequest } from '@/types';
import { postService } from '@/services/postService';
import { useAuth } from '@/contexts/AuthContext';
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
  ChevronUp
} from 'lucide-react';
import { ReactionPicker } from '@/components/ui/ReactionPicker';

interface CommentItemProps {
  comment: Comment;
  postId: string;
  onCommentUpdate?: (comment: Comment) => void;
  onCommentDelete?: (commentId: string) => void;
  depth?: number;
}

export const CommentItem: React.FC<CommentItemProps> = ({
  comment,
  postId,
  onCommentUpdate,
  onCommentDelete,
  depth = 0
}) => {
  const { user } = useAuth();
  const [showReplyForm, setShowReplyForm] = useState(false);
  const [replyText, setReplyText] = useState('');
  const [isSubmittingReply, setIsSubmittingReply] = useState(false);
  const [showReplies, setShowReplies] = useState(false);
  const [replies, setReplies] = useState<Comment[]>([]);
  const [isLoadingReplies, setIsLoadingReplies] = useState(false);
  const [showMenu, setShowMenu] = useState(false);
  const [showReactionPicker, setShowReactionPicker] = useState(false);
  const [userReaction, setUserReaction] = useState<string | null>(null);

  const isOwnComment = user?.id === comment.author?.id;
  const hasReplies = comment.stats?.replies > 0;
  const maxDepth = 3; // Limit nesting depth

  const loadReplies = async () => {
    if (isLoadingReplies || replies.length > 0) return;

    setIsLoadingReplies(true);
    try {
      // TODO: Implement getReplies API
      // const response = await postService.getCommentReplies(comment.id);
      // setReplies(response.content);
      setReplies([]); // Temporary
    } catch (error) {
      console.error('Error loading replies:', error);
    } finally {
      setIsLoadingReplies(false);
    }
  };

  const handleShowReplies = async () => {
    if (!showReplies && hasReplies) {
      await loadReplies();
    }
    setShowReplies(!showReplies);
  };

  const handleSubmitReply = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!replyText.trim() || isSubmittingReply) return;

    setIsSubmittingReply(true);
    try {
      const replyData: CreateCommentRequest = {
        content: replyText.trim(),
        parentId: comment.id
      };

      const newReply = await postService.createComment(postId, replyData);
      setReplies(prev => [...prev, newReply]);
      setReplyText('');
      setShowReplyForm(false);

      // Update parent comment reply count
      if (onCommentUpdate) {
        onCommentUpdate({
          ...comment,
          stats: {
            ...comment.stats,
            replies: (comment.stats?.replies || 0) + 1
          }
        });
      }
    } catch (error) {
      console.error('Error creating reply:', error);
    } finally {
      setIsSubmittingReply(false);
    }
  };

  const handleCommentAction = async (action: 'report' | 'delete' | 'hide') => {
    try {
      switch (action) {
        case 'delete':
          if (onCommentDelete) {
            onCommentDelete(comment.id);
          }
          break;
        case 'report':
          // TODO: Implement report functionality
          console.log('Report comment:', comment.id);
          break;
        case 'hide':
          // TODO: Implement hide functionality
          console.log('Hide comment:', comment.id);
          break;
      }
    } catch (error) {
      console.error('Error handling comment action:', error);
    }
    setShowMenu(false);
  };

  const handleReaction = (reactionType: string) => {
    // TODO: Implement comment reactions
    setUserReaction(userReaction === reactionType ? null : reactionType);
    setShowReactionPicker(false);
  };

  return (
    <div className={`${depth > 0 ? 'ml-8 border-l-2 border-gray-100 pl-4' : ''}`}>
      <div className="flex space-x-3 group">
        {/* Avatar */}
        <div className="flex-shrink-0">
          {comment.author?.avatarUrl ? (
            <Avatar
              src={comment.author.avatarUrl}
              alt={comment.author.fullName || comment.author.name || 'User'}
              size="sm"
              className="ring-2 ring-white shadow-sm"
            />
          ) : (
            <div className="w-8 h-8 bg-gradient-to-br from-blue-400 to-purple-600 rounded-full flex items-center justify-center text-white text-xs font-medium shadow-sm">
              {comment.author?.fullName?.charAt(0) || comment.author?.name?.charAt(0) || 'A'}
            </div>
          )}
        </div>

        <div className="flex-1 min-w-0">
          {/* Comment Content */}
          <div className="bg-gray-100 rounded-2xl px-4 py-3 relative">
            {/* Menu Button */}
            <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
              <div className="relative">
                <button
                  onClick={() => setShowMenu(!showMenu)}
                  className="p-1 hover:bg-gray-200 rounded-full transition-colors"
                >
                  <MoreHorizontal className="h-3 w-3 text-gray-500" />
                </button>

                {showMenu && (
                  <div className="absolute right-0 top-full mt-1 bg-white rounded-lg shadow-lg border border-gray-200 py-1 z-50 min-w-[140px]">
                    <button
                      onClick={() => handleCommentAction('report')}
                      className="flex items-center space-x-2 w-full px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                    >
                      <Flag className="h-4 w-4 text-red-500" />
                      <span>Báo cáo</span>
                    </button>

                    {isOwnComment && (
                      <button
                        onClick={() => handleCommentAction('delete')}
                        className="flex items-center space-x-2 w-full px-3 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors"
                      >
                        <Trash2 className="h-4 w-4" />
                        <span>Xóa</span>
                      </button>
                    )}

                    <button
                      onClick={() => handleCommentAction('hide')}
                      className="flex items-center space-x-2 w-full px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 transition-colors"
                    >
                      <EyeOff className="h-4 w-4" />
                      <span>Ẩn bình luận</span>
                    </button>
                  </div>
                )}
              </div>
            </div>

            {/* Author Info */}
            <div className="flex items-center space-x-2 mb-1">
              <span className="font-semibold text-sm text-gray-900 vietnamese-text truncate">
                {comment.author?.fullName || comment.author?.name || 'Người dùng ẩn danh'}
              </span>
              {comment.author?.role && (
                <span className={`px-2 py-0.5 rounded-full text-xs font-medium flex-shrink-0 ${
                  comment.author.role === 'LECTURER' 
                    ? 'bg-blue-100 text-blue-700' 
                    : 'bg-green-100 text-green-700'
                }`}>
                  {comment.author.role === 'LECTURER' ? 'Giảng viên' : 'Sinh viên'}
                </span>
              )}
            </div>

            {/* Comment Text */}
            <p className="text-sm text-gray-800 vietnamese-text leading-relaxed break-words">
              {comment.content}
            </p>
          </div>

          {/* Comment Actions */}
          <div className="flex items-center space-x-4 mt-2 text-xs text-gray-500">
            <time dateTime={comment.createdAt} className="flex-shrink-0">
              {formatTimeAgo(comment.createdAt)}
            </time>

            {/* Reaction Button */}
            <div className="relative">
              <button
                onClick={() => setShowReactionPicker(!showReactionPicker)}
                className={`hover:underline font-medium transition-colors ${
                  userReaction ? 'text-blue-600' : 'hover:text-blue-600'
                }`}
              >
                {userReaction ? `${userReaction} Đã thích` : 'Thích'}
              </button>

              {showReactionPicker && (
                <div className="absolute bottom-full left-0 mb-2 z-50">
                  <ReactionPicker
                    onReactionClick={handleReaction}
                    currentReaction={userReaction}
                  />
                </div>
              )}
            </div>

            {/* Reply Button */}
            {depth < maxDepth && (
              <button
                onClick={() => setShowReplyForm(!showReplyForm)}
                className="hover:underline font-medium transition-colors hover:text-blue-600 flex items-center space-x-1"
              >
                <Reply className="h-3 w-3" />
                <span>Trả lời</span>
              </button>
            )}

            {/* Show Replies Button */}
            {hasReplies && (
              <button
                onClick={handleShowReplies}
                className="hover:underline font-medium transition-colors hover:text-blue-600 flex items-center space-x-1"
              >
                {showReplies ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                <span>{showReplies ? 'Ẩn' : 'Xem'} {comment.stats?.replies} phản hồi</span>
              </button>
            )}
          </div>

          {/* Reply Form */}
          {showReplyForm && (
            <form onSubmit={handleSubmitReply} className="mt-3">
              <div className="flex space-x-2">
                {user?.avatarUrl ? (
                  <Avatar
                    src={user.avatarUrl}
                    alt={user.fullName || user.username || 'Your avatar'}
                    size="sm"
                    className="ring-2 ring-white shadow-sm flex-shrink-0"
                  />
                ) : (
                  <div className="w-8 h-8 bg-gray-300 rounded-full flex items-center justify-center text-white text-xs font-medium flex-shrink-0">
                    {user?.fullName?.charAt(0) || user?.name?.charAt(0) || 'A'}
                  </div>
                )}

                <div className="flex-1">
                  <Textarea
                    value={replyText}
                    onChange={(e) => setReplyText(e.target.value)}
                    placeholder={`Trả lời ${comment.author?.fullName || comment.author?.name || 'bình luận này'}...`}
                    className="min-h-[80px] text-sm bg-white border border-gray-200 rounded-lg px-3 py-2 resize-none vietnamese-text"
                    disabled={isSubmittingReply}
                  />

                  <div className="flex justify-end space-x-2 mt-2">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setShowReplyForm(false);
                        setReplyText('');
                      }}
                      className="text-xs px-3 py-1"
                    >
                      Hủy
                    </Button>
                    <Button
                      type="submit"
                      size="sm"
                      disabled={!replyText.trim() || isSubmittingReply}
                      className="text-xs px-3 py-1"
                    >
                      {isSubmittingReply ? <LoadingSpinner size="sm" /> : 'Gửi'}
                    </Button>
                  </div>
                </div>
              </div>
            </form>
          )}

          {/* Replies */}
          {showReplies && (
            <div className="mt-4 space-y-3">
              {isLoadingReplies ? (
                <div className="flex justify-center py-2">
                  <LoadingSpinner size="sm" />
                </div>
              ) : (
                replies.map((reply) => (
                  <CommentItem
                    key={reply.id}
                    comment={reply}
                    postId={postId}
                    onCommentUpdate={onCommentUpdate}
                    onCommentDelete={onCommentDelete}
                    depth={depth + 1}
                  />
                ))
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
