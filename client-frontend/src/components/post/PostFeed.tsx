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
        // Latest posts
        response = await postService.getPosts(
            page,           // number
            10,             // size
            'createdAt',    // sortBy
            'desc',         // sortDir
            authorId,       // string | undefined
            category,       // string | undefined
            search          // string | undefined
        );

      }

      if (append) {
        setPosts(prev => [...prev, ...response.content]);
      } else {
        setPosts(response.content);
      }

      setHasMore(!response.last && response.content.length > 0);
      setCurrentPage(response.number);
    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to load posts');
    } finally {
      setIsLoading(false);
      setIsLoadingMore(false);
    }
  }, [authorId, category, search, activeTab]);

  useEffect(() => {
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

  const handleTabChange = (tab: 'latest' | 'trending' | 'top-liked') => {
    setActiveTab(tab);
    setCurrentPage(0);
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

  const tabs = [
    { key: 'latest', label: 'Latest', icon: RefreshCw },
    { key: 'trending', label: 'Trending', icon: TrendingUp },
    { key: 'top-liked', label: 'Most Liked', icon: Heart }
  ];

  if (isLoading && posts.length === 0) {
    return (
      <div className="flex justify-center items-center min-h-64">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header with tabs and create button */}
      <div className="bg-white rounded-lg shadow-sm p-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4 gap-2">
          <div className="flex flex-wrap gap-2">
            {tabs.map(({ key, label, icon: Icon }) => (
                <button
                    key={key}
                    onClick={() => handleTabChange(key as any)}
                    className={`flex items-center px-3 py-2 sm:px-4 rounded-md transition-colors ${
                        activeTab === key
                            ? 'bg-blue-100 text-blue-700 border border-blue-200'
                            : 'text-gray-600 hover:bg-gray-100'
                    }`}
                >
                  <Icon className="h-5 w-5 sm:h-4 sm:w-4 mr-2" />
                  {label}
                </button>
            ))}
          </div>

          <div className="flex flex-wrap gap-2 justify-start sm:justify-end">
            <Button
                variant="outline"
                size="sm"
                onClick={handleRefresh}
                disabled={isLoading}
            >
              <RefreshCw className={`h-5 w-5 sm:h-4 sm:w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>

            <Button
                size="sm"
                onClick={() => setShowCreatePost(!showCreatePost)}
            >
              <Plus className="h-5 w-5 sm:h-4 sm:w-4 mr-2" />
              Create Post
            </Button>
          </div>
        </div>



      {/* Search/Filter Info */}
        {(search || category || authorId) && (
          <div className="text-sm text-gray-600 mb-2">
            Showing posts
            {search && ` matching "${search}"`}
            {category && ` in category "${category}"`}
            {authorId && ` by author`}
          </div>
        )}
      </div>

      {/* Create Post Form */}
      {showCreatePost && (
        <CreatePost
          onPostCreated={handlePostCreated}
          onCancel={() => setShowCreatePost(false)}
        />
      )}

      {/* Error Alert */}
      {error && (
        <ErrorAlert 
          message={error} 
          onClose={() => setError(null)} 
        />
      )}

      {/* Posts List */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {posts.length === 0 && !isLoading ? (
            <div className="text-center py-12 bg-white rounded-lg shadow-sm col-span-full">
              {/* Message no posts */}
            </div>
        ) : (
            posts.map((post) => (
                <PostCard
                    key={post.id}
                    post={post}
                    onPostUpdate={handlePostUpdate}
                    onPostDelete={handlePostDelete}
                />
            ))
        )}
      </div>


      {/* Load More Button */}
      {hasMore && posts.length > 0 && (
        <div className="flex justify-center pt-4">
          <Button
            variant="outline"
            onClick={handleLoadMore}
            disabled={isLoadingMore}
            loading={isLoadingMore}
          >
            {isLoadingMore ? 'Loading...' : 'Load More Posts'}
          </Button>
        </div>
      )}

      {/* Loading indicator for infinite scroll */}
      {isLoadingMore && (
        <div className="flex justify-center py-4">
          <LoadingSpinner />
        </div>
      )}
    </div>
  );
};
