'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Post, PaginatedResponse } from '@/types';
import { postService } from '@/services/postService';
import { PostCard } from './PostCard';
import { CreatePost } from './CreatePost';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorAlert } from '@/components/ui/ErrorAlert';
import { Button } from '@/components/ui/Button';
import { RefreshCw, Plus, TrendingUp, Eye, Heart } from 'lucide-react';

interface PostFeedProps {
  authorId?: string;
  category?: string;
  search?: string;
  className?: string;
}

export const PostFeed: React.FC<PostFeedProps> = ({
  authorId,
  category,
  search,
  className = ''
}) => {
  const [posts, setPosts] = useState<Post[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(true);
  const [currentPage, setCurrentPage] = useState(0);
  const [showCreatePost, setShowCreatePost] = useState(false);
  const [activeTab, setActiveTab] = useState<'latest' | 'trending' | 'top-liked'>('latest');

  const loadPosts = useCallback(async (page = 0, append = false) => {
    try {
      if (!append) {
        setIsLoading(true);
        setError(null);
      } else {
        setIsLoadingMore(true);
      }

      let response: PaginatedResponse<Post>;

      if (activeTab === 'trending') {
        const trendingPosts = await postService.getTopViewedPosts();
        response = {
          content: trendingPosts,
          totalElements: trendingPosts.length,
          totalPages: 1,
          size: trendingPosts.length,
          number: 0,
          first: true,
          last: true
        };
      } else if (activeTab === 'top-liked') {
        const topLikedPosts = await postService.getTopLikedPosts();
        response = {
          content: topLikedPosts,
          totalElements: topLikedPosts.length,
          totalPages: 1,
          size: topLikedPosts.length,
          number: 0,
          first: true,
          last: true
        };
      } else {
        response = await postService.getPosts(
          page,
          10,
          'createdAt',
          'desc',
          authorId,
          category,
          search
        );
      }

      if (append) {
        setPosts(prev => [...prev, ...response.content]);
      } else {
        setPosts(response.content);
      }

      setHasMore(!response.last && response.content.length > 0);
      setCurrentPage(page);
    } catch (err: any) {
      console.error('Failed to load posts:', err);
      setError(err.response?.data?.message || err.message || 'Failed to load posts');
    } finally {
      setIsLoading(false);
      setIsLoadingMore(false);
    }
  }, [authorId, category, search, activeTab]);

  // Initial load and reload when filters change
  useEffect(() => {
    setCurrentPage(0);
    loadPosts(0, false);
  }, [loadPosts]);

  const handleLoadMore = () => {
    if (hasMore && !isLoadingMore) {
      loadPosts(currentPage + 1, true);
    }
  };

  const handleRefresh = () => {
    setCurrentPage(0);
    loadPosts(0, false);
  };

  const handlePostCreated = (newPost: Post) => {
    setPosts(prev => [newPost, ...prev]);
    setShowCreatePost(false);
  };

  const handlePostUpdate = (updatedPost: Post) => {
    setPosts(prev => prev.map(post =>
      post.id === updatedPost.id ? updatedPost : post
    ));
  };

  const handlePostDelete = (postId: string) => {
    setPosts(prev => prev.filter(post => post.id !== postId));
  };

  const handleTabChange = (tab: 'latest' | 'trending' | 'top-liked') => {
    setActiveTab(tab);
    setCurrentPage(0);
  };

  if (isLoading) {
    return (
      <div className="flex justify-center py-8">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <div className={`post-feed ${className}`}>
      {/* Feed Header */}
      <div className="bg-white rounded-lg shadow-sm border p-4 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">
            {search ? `Search results for "${search}"` :
             category ? `Posts in ${category}` :
             authorId ? 'User Posts' : 'Latest Posts'}
          </h2>

          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={handleRefresh}
              disabled={isLoading}
              className="flex items-center gap-2"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>

            {!authorId && (
              <Button
                size="sm"
                onClick={() => setShowCreatePost(true)}
                className="flex items-center gap-2"
              >
                <Plus className="w-4 h-4" />
                Create Post
              </Button>
            )}
          </div>
        </div>

        {/* Feed Tabs */}
        {!search && !category && !authorId && (
          <div className="flex gap-2">
            <Button
              variant={activeTab === 'latest' ? 'primary' : 'secondary'}
              size="sm"
              onClick={() => handleTabChange('latest')}
              className="flex items-center gap-2"
            >
              <RefreshCw className="w-4 h-4" />
              Latest
            </Button>
            <Button
              variant={activeTab === 'trending' ? 'primary' : 'secondary'}
              size="sm"
              onClick={() => handleTabChange('trending')}
              className="flex items-center gap-2"
            >
              <Eye className="w-4 h-4" />
              Trending
            </Button>
            <Button
              variant={activeTab === 'top-liked' ? 'primary' : 'secondary'}
              size="sm"
              onClick={() => handleTabChange('top-liked')}
              className="flex items-center gap-2"
            >
              <Heart className="w-4 h-4" />
              Most Liked
            </Button>
          </div>
        )}
      </div>

      {/* Create Post Modal/Inline */}
      {showCreatePost && (
        <div className="bg-white rounded-lg shadow-sm border p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Create New Post</h3>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowCreatePost(false)}
            >
              Cancel
            </Button>
          </div>
          <CreatePost
            onPostCreated={handlePostCreated}
            onCancel={() => setShowCreatePost(false)}
          />
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="mb-6">
          <ErrorAlert message={error} />
        </div>
      )}

      {/* Posts List */}
      {posts.length === 0 ? (
        <div className="text-center py-12">
          <div className="text-gray-400 mb-4">
            <TrendingUp className="w-16 h-16 mx-auto" />
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">No posts found</h3>
          <p className="text-gray-500 mb-4">
            {search ? 'Try adjusting your search terms' :
             category ? 'No posts in this category yet' :
             'Be the first to create a post!'}
          </p>
          {!authorId && !search && !category && (
            <Button onClick={() => setShowCreatePost(true)}>
              Create First Post
            </Button>
          )}
        </div>
      ) : (
        <div className="space-y-6">
          {posts.map((post) => (
            <PostCard
              key={post.id}
              post={post}
              onPostUpdate={handlePostUpdate}
              onPostDelete={handlePostDelete}
            />
          ))}

          {/* Load More Button */}
          {hasMore && (
            <div className="text-center py-6">
              <Button
                variant="secondary"
                onClick={handleLoadMore}
                disabled={isLoadingMore}
                className="flex items-center gap-2"
              >
                {isLoadingMore && <LoadingSpinner size="sm" />}
                {isLoadingMore ? 'Loading...' : 'Load More Posts'}
              </Button>
            </div>
          )}

          {/* End of Feed Message */}
          {!hasMore && posts.length > 0 && (
            <div className="text-center py-6 text-gray-500">
              <p>You've reached the end of the feed</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
