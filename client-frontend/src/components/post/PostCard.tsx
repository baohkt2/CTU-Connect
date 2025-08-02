'use client';

import React, { useState, useEffect } from 'react';
import { Post, CreateCommentRequest } from '@/types';
import { postService } from '@/services/postService';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Textarea } from '@/components/ui/Textarea';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import {
  Heart,
  MessageCircle,
  Share,
  Bookmark,
  MoreHorizontal,
  Send,
  Eye
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';


interface PostCardProps {
  post: Post;
  onPostUpdate?: (updatedPost: Post) => void;
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

  // Load interaction status on component mount to fix state persistence
  useEffect(() => {
    const loadInteractionStatus = async () => {
      try {
        const status = await postService.getInteractionStatus(post.id);
        setIsLiked(status.hasLiked);
        setIsBookmarked(status.hasBookmarked);
      } catch (error) {
        // Silently fail - user might not be authenticated
        // This prevents errors when viewing posts without login
        console.debug('Could not load interaction status:', error);
      }
    };

    loadInteractionStatus();
  }, [post.id]);

  const handleShowComments = async () => {
    if (!showComments && comments.length === 0) {
      setIsLoadingComments(true);
      try {
        const response = await postService.getComments(post.id);
        setComments(response.content);
      } catch (error) {
        console.error('Failed to load comments:', error);
      } finally {
        setIsLoadingComments(false);
      }
    }
    setShowComments(!showComments);
  };

  const handleLike = async () => {
    if (isLoadingInteraction) return;

    setIsLoadingInteraction(true);
    try {
      await postService.toggleLike(post.id);
      setIsLiked(!isLiked);

      const updatedPost = {
        ...post,
        stats: {
          ...post.stats,
          likes: isLiked ? post.stats.likes - 1 : post.stats.likes + 1
        }
      };
      onPostUpdate?.(updatedPost);
    } catch (error) {
      console.error('Failed to toggle like:', error);
    } finally {
      setIsLoadingInteraction(false);
    }
  };

  const handleBookmark = async () => {
    if (isLoadingInteraction) return;

    setIsLoadingInteraction(true);
    try {
      await postService.toggleBookmark(post.id);
      setIsBookmarked(!isBookmarked);

      const updatedPost = {
        ...post,
        stats: {
          ...post.stats,
          bookmarks: isBookmarked ? post.stats.bookmarks - 1 : post.stats.bookmarks + 1
        }
      };
      onPostUpdate?.(updatedPost);
    } catch (error) {
      console.error('Failed to toggle bookmark:', error);
    } finally {
      setIsLoadingInteraction(false);
    }
  };

  const handleShare = async () => {
    if (isLoadingInteraction) return;

    setIsLoadingInteraction(true);
    try {
      await postService.sharePost(post.id);

      const updatedPost = {
        ...post,
        stats: {
          ...post.stats,
          shares: post.stats.shares + 1
        }
      };
      onPostUpdate?.(updatedPost);

      // Copy link to clipboard
      await navigator.clipboard.writeText(`${window.location.origin}/posts/${post.id}`);
      alert('Post link copied to clipboard!');
    } catch (error) {
      console.error('Failed to share post:', error);
    } finally {
      setIsLoadingInteraction(false);
    }
  };

  const handleSubmitComment = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!commentText.trim() || isSubmittingComment) return;

    setIsSubmittingComment(true);
    try {
      const commentData: CreateCommentRequest = {
        content: commentText.trim()
      };

      const newComment = await postService.createComment(post.id, commentData);
      setComments(prev => [...prev, newComment]);
      setCommentText('');

      const updatedPost = {
        ...post,
        stats: {
          ...post.stats,
          comments: post.stats.comments + 1
        }
      };
      onPostUpdate?.(updatedPost);
    } catch (error) {
      console.error('Failed to create comment:', error);
    } finally {
      setIsSubmittingComment(false);
    }
  };

  const formatStats = (count: number): string => {
    if (count >= 1000000) return `${(count / 1000000).toFixed(1)}M`;
    if (count >= 1000) return `${(count / 1000).toFixed(1)}K`;
    return count.toString();
  };



  return (
    <Card className={`post-card bg-white rounded-lg shadow ${className}`}>
      {/* Post Header */}
      <div className="flex items-start justify-between p-4">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gray-300 rounded-full flex items-center justify-center">
            {post.authorAvatar ? (
              <img
                src={post.authorAvatar}
                alt={post.authorName || 'Author'}
                className="w-full h-full rounded-full object-cover"
              />
            ) : (
              <span className="text-sm font-medium text-gray-600">
                {(post.authorName || 'Anonymous').charAt(0).toUpperCase()}
              </span>
            )}
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">
              {post.authorName || 'Anonymous'}
            </h3>
            <p className="text-sm text-gray-500">
              {formatDistanceToNow(new Date(post.createdAt), { addSuffix: true })}
              {post.visibility && post.visibility !== 'PUBLIC' && (
                <span className="ml-2 px-2 py-0.5 bg-gray-100 rounded text-xs">
                  {post.visibility.toLowerCase()}
                </span>
              )}
            </p>
          </div>
        </div>
        <Button variant="ghost" size="sm">
          <MoreHorizontal className="w-4 h-4" />
        </Button>
      </div>

      {/* Post Content */}
      <div className="px-4 pb-3">
        {post.title && (
          <h2 className="text-lg font-semibold mb-2">{post.title}</h2>
        )}
        <p className="text-gray-900 whitespace-pre-wrap">{post.content}</p>

        {/* Tags */}
        {post.tags && post.tags.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-3">
            {post.tags.map((tag, index) => (
              <span
                key={index}
                className="inline-block px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
              >
                #{tag}
              </span>
            ))}
          </div>
        )}

        {/* Category */}
        {post.category && (
          <div className="mt-2">
            <span className="inline-block px-2 py-1 bg-green-100 text-green-800 rounded text-sm">
              {post.category}
            </span>
          </div>
        )}
      </div>

      {/* Post Images */}
      {post.images && post.images.length > 0 && (
        <div className="px-4 pb-3">
          <div className={`grid gap-2 ${
            post.images.length === 1 ? 'grid-cols-1' : 
            post.images.length === 2 ? 'grid-cols-2' : 
            'grid-cols-2'
          }`}>
            {post.images.slice(0, 4).map((image, index) => (
              <div key={index} className="relative">
                <img
                  src={image}
                  alt={`Post image ${index + 1}`}
                  className="w-full h-48 object-cover rounded-lg"
                />
                {index === 3 &&  post.images && post.images.length > 4 && (
                  <div className="absolute inset-0 bg-black bg-opacity-50 rounded-lg flex items-center justify-center">
                    <span className="text-white text-lg font-semibold">
                      +{post.images.length - 4} more
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Post Stats */}
      <div className="px-4 py-2 border-t border-gray-100">
        <div className="flex items-center justify-between text-sm text-gray-500">
          <div className="flex items-center space-x-4">
            <span className="flex items-center space-x-1">
              <Eye className="w-4 h-4" />
              <span>{formatStats(post.stats.views)}</span>
            </span>
            <span className="flex items-center space-x-1">
              <Heart className="w-4 h-4" />
              <span>{formatStats(post.stats.likes)}</span>
            </span>
          </div>
          <div className="flex items-center space-x-4">
            <span>{formatStats(post.stats.comments)} comments</span>
            <span>{formatStats(post.stats.shares)} shares</span>
          </div>
        </div>
      </div>

      {/* Post Actions */}
      <div className="px-4 py-2 border-t border-gray-100">
        <div className="flex items-center justify-between">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleLike}
            disabled={isLoadingInteraction}
            className={`flex items-center space-x-2 ${isLiked ? 'text-red-500' : 'text-gray-500'}`}
          >
            <Heart className={`w-5 h-5 ${isLiked ? 'fill-current' : ''}`} />
            <span>Like</span>
          </Button>

          <Button
            variant="ghost"
            size="sm"
            onClick={handleShowComments}
            className="flex items-center space-x-2 text-gray-500"
          >
            <MessageCircle className="w-5 h-5" />
            <span>Comment</span>
          </Button>

          <Button
            variant="ghost"
            size="sm"
            onClick={handleShare}
            disabled={isLoadingInteraction}
            className="flex items-center space-x-2 text-gray-500"
          >
            <Share className="w-5 h-5" />
            <span>Share</span>
          </Button>

          <Button
            variant="ghost"
            size="sm"
            onClick={handleBookmark}
            disabled={isLoadingInteraction}
            className={`flex items-center space-x-2 ${isBookmarked ? 'text-blue-500' : 'text-gray-500'}`}
          >
            <Bookmark className={`w-5 h-5 ${isBookmarked ? 'fill-current' : ''}`} />
            <span>Save</span>
          </Button>
        </div>
      </div>

      {/* Comments Section */}
      {showComments && (
        <div className="border-t border-gray-100">
          {/* Comment Form */}
          <form onSubmit={handleSubmitComment} className="p-4 border-b border-gray-100">
            <div className="flex space-x-3">
              <div className="w-8 h-8 bg-gray-300 rounded-full flex-shrink-0"></div>
              <div className="flex-1">
                <Textarea
                  placeholder="Write a comment..."
                  value={commentText}
                  onChange={(e) => setCommentText(e.target.value)}
                  className="w-full resize-none min-h-[60px]"
                />
                <div className="flex justify-end mt-2">
                  <Button
                    type="submit"
                    size="sm"
                    disabled={!commentText.trim() || isSubmittingComment}
                    className="flex items-center space-x-2"
                  >
                    {isSubmittingComment ? (
                      <LoadingSpinner />
                    ) : (
                      <Send className="w-4 h-4" />
                    )}
                    <span>Post</span>
                  </Button>
                </div>
              </div>
            </div>
          </form>

          {/* Comments List */}
          <div className="p-4">
            {isLoadingComments ? (
              <div className="flex justify-center py-4">
                <LoadingSpinner />
              </div>
            ) : comments.length === 0 ? (
              <p className="text-gray-500 text-center py-4">No comments yet</p>
            ) : (
              <div className="space-y-4">
                {comments.map((comment) => (
                  <div key={comment.author.id} className="flex space-x-3">
                    <div className="w-8 h-8 bg-gray-300 rounded-full flex-shrink-0">
                      {comment.author.avatar ? (
                        <img
                          src={comment.author.avatar}
                          alt={comment.author.name}
                          className="w-full h-full rounded-full object-cover"
                        />
                      ) : (
                        <span className="text-xs font-medium text-gray-600 flex items-center justify-center w-full h-full">
                          {(comment.author.name || 'A').charAt(0).toUpperCase()}
                        </span>
                      )}
                    </div>
                    <div className="flex-1">
                      <div className="bg-gray-100 rounded-lg px-3 py-2">
                        <p className="font-semibold text-sm">{comment.author.name || 'Anonymous'}</p>
                        <p className="text-gray-900">{comment.content}</p>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        {formatDistanceToNow(new Date(comment.createdAt), { addSuffix: true })}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </Card>
  );
};
