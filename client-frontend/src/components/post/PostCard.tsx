'use client';

import React, { useState, useEffect, useCallback } from 'react';
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
        console.debug('Could not load interaction status:', error);
      }
    })();
    return () => { mounted = false; };
  }, [post.id]);

  // Toggle phần comment
  const toggleComments = useCallback(async () => {
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
          return newBookmarked;
        });
      } else if (type === 'share') {
        await postService.sharePost(post.id);
        onPostUpdate?.({
          ...post,
          stats: { ...post.stats, shares: post.stats.shares + 1 }
        });
        await navigator.clipboard.writeText(`${window.location.origin}/posts/${post.id}`);
        alert('Post link copied to clipboard!');
      }
    } catch (error) {
      console.error(`Failed to handle ${type}:`, error);
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
    } catch (error) {
      console.error('Failed to create comment:', error);
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

  return (
      <Card className={`post-card bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300 ${className}`}>
        <header className="flex items-center justify-between p-4">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 rounded-full bg-gray-200 flex items-center justify-center overflow-hidden select-none flex-shrink-0">
              {post.authorAvatar ? (
                  <img
                      src={post.authorAvatar}
                      alt={`${post.authorName ?? 'Anonymous'}'s Avatar`}
                      className="object-cover w-full h-full"
                      loading="lazy"
                      decoding="async"
                  />
              ) : (
                  <span className="text-lg font-semibold text-gray-600">
                {(post.authorName ?? 'Anonymous').charAt(0).toUpperCase()}
              </span>
              )}
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 leading-tight">{post.authorName ?? 'Anonymous'}</h3>
              <time
                  dateTime={new Date(post.createdAt).toISOString()}
                  className="text-sm text-gray-500"
                  title={new Date(post.createdAt).toLocaleString()}
              >
                {formatDistanceToNow(new Date(post.createdAt), { addSuffix: true })}
                {post.visibility && post.visibility !== 'PUBLIC' && (
                    <span className="ml-2 px-2 py-0.5 bg-gray-100 rounded text-xs capitalize font-medium text-gray-600 select-none">
                  {post.visibility.toLowerCase()}
                </span>
                )}
              </time>
            </div>
          </div>
          <Button variant="ghost" size="sm" aria-label="More options" disabled>
            <MoreHorizontal className="w-5 h-5 text-gray-400" />
          </Button>
        </header>

        <article className="px-4 pb-4">
          {post.title && <h2 className="text-xl font-semibold mb-2 text-gray-900 break-words">{post.title}</h2>}
          <p className="whitespace-pre-wrap text-gray-800 mb-3 break-words">{post.content}</p>

          {post.tags && post.tags.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-3">
                {post.tags.map((tag, i) => (
                    <span
                        key={i}
                        className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-semibold cursor-pointer select-none hover:bg-blue-200 transition-colors duration-200"
                        title={`Tag: ${tag}`}
                    >
                #{tag}
              </span>
                ))}
              </div>
          )}

          {post.category && (
              <span className="inline-block px-3 py-1 bg-green-100 text-green-800 rounded text-sm font-semibold select-none">
            {post.category}
          </span>
          )}
        </article>

        {/* Images */}
        {post.images && post.images.length > 0 && (
            <section className="px-4 pb-4">
              <div className={`grid gap-2 ${post.images.length === 1 ? 'grid-cols-1' : 'grid-cols-2'}`}>
                {post.images.slice(0, 4).map((image, index) => (
                    <div key={index} className="relative rounded-lg overflow-hidden group cursor-pointer">
                      <img
                          src={image}
                          alt={`Post image ${index + 1}`}
                          className="object-cover w-full h-48 transition-transform duration-300 group-hover:scale-105"
                          loading="lazy"
                      />
                      {index === 3 && post.images &&  post.images.length > 4 && (
                          <div className="absolute inset-0 bg-black bg-opacity-60 flex items-center justify-center text-white font-bold text-lg">
                            +{post.images && post.images.length - 4}
                          </div>
                      )}
                    </div>
                ))}
              </div>
            </section>
        )}

        {/* Videos */}
        {post.videos && post.videos.length > 0 && (
            <section className="px-4 pb-4">
              <div className={`grid gap-2 ${post.videos && post.videos.length === 1 ? 'grid-cols-1' : 'grid-cols-2'}`}>
                {post.videos.slice(0, 2).map((video, idx) => (
                    <div key={idx} className="relative rounded-lg overflow-hidden">
                      <video
                          src={video}
                          controls
                          preload="metadata"
                          className="w-full h-48 object-cover rounded-lg shadow-md"
                      >
                        Your browser does not support the video tag.
                      </video>
                      {idx === 1 && post.videos && post.videos.length > 2 && (
                          <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center pointer-events-none rounded-lg">
                    <span className="text-white font-semibold text-lg">
                      +{post.videos && post.videos.length - 2} videos
                    </span>
                          </div>
                      )}
                    </div>
                ))}
              </div>
            </section>
        )}

        {/* Stats */}
        <section className="px-4 py-2 border-t border-gray-100 flex justify-between text-sm text-gray-600 select-none">
          <div className="flex space-x-6" aria-label={`${post.stats.views} views`}>
          <span className="flex items-center space-x-1">
            <Eye className="w-5 h-5 text-gray-500" />
            <span>{formatStats(post.stats.views)}</span>
          </span>
            <span className="flex items-center space-x-1" aria-label={`${post.stats.likes} likes`}>
            <Heart className="w-5 h-5 text-red-500" />
            <span>{formatStats(post.stats.likes)}</span>
          </span>
          </div>
          <div className="flex space-x-6">
            <span>{formatStats(post.stats.comments)} comments</span>
            <span>{formatStats(post.stats.shares)} shares</span>
          </div>
        </section>

        {/* Actions */}
        <section className="px-4 py-2 border-t border-gray-100 flex justify-between items-center">
          <Button
              variant="ghost"
              size="sm"
              onClick={() => handleInteraction('like')}
              disabled={isLoadingInteraction}
              className={`flex items-center space-x-2 transition-colors duration-150 ${
                  isLiked ? 'text-red-600' : 'text-gray-600 hover:text-red-600'
              }`}
              aria-pressed={isLiked}
              aria-label={isLiked ? 'Unlike post' : 'Like post'}
          >
            <Heart className={`w-6 h-6 ${isLiked ? 'fill-current' : ''}`} />
            <span>Like</span>
          </Button>

          <Button
              variant="ghost"
              size="sm"
              onClick={toggleComments}
              className="flex items-center space-x-2 text-gray-600 hover:text-blue-600 transition-colors duration-150"
              aria-expanded={showComments}
              aria-controls={`comments-section-${post.id}`}
              aria-label={showComments ? 'Hide comments' : 'Show comments'}
          >
            <MessageCircle className="w-6 h-6" />
            <span>Comment</span>
          </Button>

          <Button
              variant="ghost"
              size="sm"
              onClick={() => handleInteraction('share')}
              disabled={isLoadingInteraction}
              className="flex items-center space-x-2 text-gray-600 hover:text-green-600 transition-colors duration-150"
              aria-label="Share post"
          >
            <Share className="w-6 h-6" />
            <span>Share</span>
          </Button>

          <Button
              variant="ghost"
              size="sm"
              onClick={() => handleInteraction('bookmark')}
              disabled={isLoadingInteraction}
              className={`flex items-center space-x-2 transition-colors duration-150 ${
                  isBookmarked ? 'text-blue-600' : 'text-gray-600 hover:text-blue-600'
              }`}
              aria-pressed={isBookmarked}
              aria-label={isBookmarked ? 'Remove bookmark' : 'Save post'}
          >
            <Bookmark className={`w-6 h-6 ${isBookmarked ? 'fill-current' : ''}`} />
            <span>Save</span>
          </Button>
        </section>

        {/* Comment Section */}
        {showComments && (
            <section
                id={`comments-section-${post.id}`}
                className="border-t border-gray-100"
                aria-live="polite"
            >
              <form onSubmit={handleSubmitComment} className="p-4 border-b border-gray-100 flex items-start space-x-3">
                <div className="w-10 h-10 rounded-full bg-gray-300 flex-shrink-0 flex justify-center items-center select-none text-gray-600 font-semibold">
                  {/* TODO: if user avatar available, show avatar */}
                  <span>U</span>
                </div>
                <div className="flex-1">
                  <Textarea
                      placeholder="Write a comment..."
                      value={commentText}
                      onChange={(e) => setCommentText(e.target.value)}
                      className="w-full resize-none min-h-[60px] rounded-md border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                      aria-label="Write a comment"
                      rows={3}
                      required
                  />
                  <div className="flex justify-end mt-2">
                    <Button
                        type="submit"
                        size="sm"
                        disabled={!commentText.trim() || isSubmittingComment}
                        className="flex items-center space-x-2"
                        aria-busy={isSubmittingComment}
                    >
                      {isSubmittingComment ? (
                          <LoadingSpinner size="sm" />
                      ) : (
                          <Send className="w-4 h-4" />
                      )}
                      <span>Post</span>
                    </Button>
                  </div>
                </div>
              </form>

              <div className="p-4 max-h-96 overflow-y-auto space-y-4">
                {isLoadingComments ? (
                    <div className="flex justify-center py-6">
                      <LoadingSpinner />
                    </div>
                ) : comments.length === 0 ? (
                    <p className="text-center text-gray-500 py-6 select-none">No comments yet</p>
                ) : (
                    comments.map((comment) => (
                        <article key={comment.id} className="flex items-start space-x-3">
                          <div className="w-10 h-10 rounded-full bg-gray-300 flex-shrink-0 overflow-hidden select-none">
                            {comment.author.avatar ? (
                                <img
                                    src={comment.author.avatar}
                                    alt={`${comment.author.name ?? 'User'} avatar`}
                                    className="w-full h-full object-cover"
                                    loading="lazy"
                                />
                            ) : (
                                <span className="flex justify-center items-center h-full text-xs font-semibold text-gray-600">
                        {(comment.author.name ?? 'A').charAt(0).toUpperCase()}
                      </span>
                            )}
                          </div>
                          <div className="flex-1">
                            <div className="bg-gray-100 rounded-lg p-3 shadow-sm">
                              <h4 className="font-semibold text-gray-900 text-sm mb-1">{comment.author.name ?? 'Anonymous'}</h4>
                              <p className="text-gray-800 whitespace-pre-wrap break-words">{comment.content}</p>
                            </div>
                            <time
                                dateTime={new Date(comment.createdAt).toISOString()}
                                className="text-xs text-gray-500 mt-1 block select-none"
                                title={new Date(comment.createdAt).toLocaleString()}
                            >
                              {formatDistanceToNow(new Date(comment.createdAt), { addSuffix: true })}
                            </time>
                          </div>
                        </article>
                    ))
                )}
              </div>
            </section>
        )}
      </Card>
  );
};
