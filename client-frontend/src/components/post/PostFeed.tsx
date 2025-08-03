'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Post, PaginatedResponse } from '@/types';
import { postService } from '@/services/postService';
import { PostCard } from './PostCard';
import { CreatePost } from './CreatePost';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorAlert } from '@/components/ui/ErrorAlert';
import { Button } from '@/components/ui/Button';
import { RefreshCw, Plus, TrendingUp, Heart } from 'lucide-react';

interface PostFeedProps {
  authorId?: string;
  authorName?: string; // dùng để hiển thị author name trong filter info
  category?: string;
  search?: string;
  className?: string;
}

export const PostFeed: React.FC<PostFeedProps> = ({
                                                    authorId,
                                                    authorName,
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

  const loadPosts = useCallback(
      async (page = 0, append = false) => {
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
              last: true,
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
              last: true,
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
            setPosts((prev) => [...prev, ...response.content]);
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
      },
      [activeTab, authorId, category, search]
  );

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
    if (tab !== activeTab) {
      setActiveTab(tab);
      setCurrentPage(0);
      setPosts([]); // reset posts khi đổi tab để chuyển sạch
    }
  };

  const handlePostCreated = (newPost: Post) => {
    setPosts((prev) => [newPost, ...prev]);
    setShowCreatePost(false);
  };

  const handlePostUpdate = (updatedPost: Post) => {
    setPosts((prev) => prev.map((p) => (p.id === updatedPost.id ? updatedPost : p)));
  };

  const handlePostDelete = (postId: string) => {
    setPosts((prev) => prev.filter((p) => p.id !== postId));
  };

  const tabs = [
    { key: 'latest', label: 'Latest', icon: RefreshCw },
    { key: 'trending', label: 'Trending', icon: TrendingUp },
    { key: 'top-liked', label: 'Most Liked', icon: Heart }
  ];

  if (isLoading && posts.length === 0) {
    return (
        <main className="flex justify-center items-center min-h-[16rem]">
          <LoadingSpinner size="lg" />
        </main>
    );
  }

  return (
      <main className={`space-y-6 ${className}`}>
        {/* Header: tabs và nút tạo bài */}
        <header className="bg-white rounded-lg shadow-sm p-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
            <nav aria-label="Post sorting tabs" className="flex flex-wrap gap-2">
              {tabs.map(({ key, label, icon: Icon }) => {
                const isActive = activeTab === key;
                return (
                    <button
                        key={key}
                        type="button"
                        aria-current={isActive ? 'page' : undefined}
                        onClick={() => handleTabChange(key as any)}
                        className={`flex items-center px-4 py-2 rounded-md font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
                            isActive
                                ? 'bg-blue-100 text-blue-700 border border-blue-200'
                                : 'text-gray-600 hover:bg-gray-100'
                        }`}
                    >
                      <Icon className="mr-2 h-5 w-5" aria-hidden="true" />
                      <span>{label}</span>
                    </button>
                );
              })}
            </nav>

            <div className="flex gap-2 justify-start sm:justify-end">
              <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={handleRefresh}
                  disabled={isLoading || isLoadingMore}
                  aria-label="Refresh posts list"
              >
                <RefreshCw
                    className={`mr-2 h-5 w-5 transition-transform ${
                        isLoading ? 'animate-spin' : ''
                    }`}
                    aria-hidden="true"
                />
                Refresh
              </Button>

              <Button
                  type="button"
                  size="sm"
                  onClick={() => setShowCreatePost((v) => !v)}
                  aria-expanded={showCreatePost}
                  aria-controls="create-post-form"
              >
                <Plus className="mr-2 h-5 w-5" aria-hidden="true" />
                Create Post
              </Button>
            </div>
          </div>

          {(search || category || authorId) && (
              <p className="text-xs text-gray-600 select-none" aria-live="polite">
                Showing posts
                {search && ` matching "${search}"`}
                {category && ` in category "${category}"`}
                {authorId && ` by author "${authorName ?? authorId}"`}
              </p>
          )}
        </header>

        {/* Form tạo bài viết */}
        {showCreatePost && (
            <section id="create-post-form" aria-label="Create a new post">
              <CreatePost onPostCreated={handlePostCreated} onCancel={() => setShowCreatePost(false)} />
            </section>
        )}

        {/* Thông báo lỗi */}
        {error && <ErrorAlert message={error} onClose={() => setError(null)} />}

        {/* Danh sách bài viết cuộn dọc */}
        <section
            aria-live="polite"
            aria-busy={isLoading}
            className="max-h-[70vh] overflow-y-auto space-y-4"
            aria-label="List of posts"
        >
          {!isLoading && posts.length === 0 ? (
              <p className="text-center text-gray-500 py-12 select-none w-full">No posts found.</p>
          ) : (
              posts.map((post) => (
                  <div key={post.id} className="w-full max-w-lg mx-auto">
                    <PostCard post={post} onPostUpdate={handlePostUpdate} onPostDelete={handlePostDelete} />
                  </div>
              ))
          )}
        </section>

        {/* Load More Button */}
        {hasMore && posts.length > 0 && (
            <div className="flex justify-center pt-4">
              <Button
                  variant="outline"
                  onClick={handleLoadMore}
                  disabled={isLoadingMore || isLoading}
                  loading={isLoadingMore}
                  aria-label="Load more posts"
              >
                {isLoadingMore ? 'Loading...' : 'Load More Posts'}
              </Button>
            </div>
        )}

        {/* Loading spinner ở cuối khi tải thêm */}
        {isLoadingMore && (
            <div className="flex justify-center py-4" aria-hidden="true">
              <LoadingSpinner />
            </div>
        )}
      </main>
  );
};
