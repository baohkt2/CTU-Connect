'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Post, PaginatedResponse } from '@/types';
import { postService } from '@/services/postService';
import { PostCard } from './PostCard';
import { CreatePost } from './CreatePost';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorAlert } from '@/components/ui/ErrorAlert';
import { Button } from '@/components/ui/Button';
import { RefreshCw, Plus, TrendingUp, Heart, Filter, MessageCircle } from 'lucide-react';

interface PostFeedProps {
  authorId?: string;
  authorName?: string;
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
      setPosts([]);
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
    { key: 'latest', label: 'Mới nhất', icon: RefreshCw },
    { key: 'trending', label: 'Thịnh hành', icon: TrendingUp },
    { key: 'top-liked', label: 'Yêu thích', icon: Heart }
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
        {/* Header */}
        <header className="bg-white rounded-xl shadow-md border border-gray-200 overflow-hidden">
          {/* Hero top banner */}
          <div className="bg-gradient-to-r from-indigo-600 to-purple-700 px-6 py-5 flex flex-col sm:flex-row sm:justify-between sm:items-center gap-3 sm:gap-0">
            <div>
              <h1 className="text-white text-2xl font-extrabold tracking-tight">
                Bảng tin CTU Connect
              </h1>
              <p className="mt-1 text-indigo-200 text-sm max-w-md">
                Khám phá, chia sẻ những điều thú vị với cộng đồng CTU
              </p>
            </div>

              <Button
                  onClick={() => setShowCreatePost((v) => !v)}
                  aria-expanded={showCreatePost}
                  aria-controls="create-post-form"
                  variant="especially"
                  size="sm"
              >
                <Plus className={`w-5 h-5 ${showCreatePost ? 'rotate-45' : ''}`} />
                <span>{showCreatePost ? 'Hủy bài viết' : 'Tạo bài viết'}</span>
              </Button>
            </div>


          {/* Tabs + Control */}
          <div className="px-6 py-4 border-t border-gray-100 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            {/* Tabs */}
            <nav aria-label="Bộ lọc bài viết" className="flex flex-wrap gap-3">
              {tabs.map(({ key, label, icon: Icon }) => {
                const isActive = activeTab === key;
                return (
                    <button
                        key={key}
                        type="button"
                        aria-current={isActive ? 'page' : undefined}
                        onClick={() => handleTabChange(key as any)}
                        className={`flex items-center gap-2 px-4 py-2 rounded-lg font-semibold text-sm transition-all duration-200
                    ${
                            isActive
                                ? 'bg-indigo-100 text-indigo-700 shadow-sm border border-indigo-300'
                                : 'text-gray-600 hover:bg-gray-50 hover:text-indigo-600 border border-transparent'
                        }
                  `}
                        title={label}
                    >
                      <Icon
                          className={`w-5 h-5 transition-transform ${
                              isActive ? 'text-indigo-600' : 'text-gray-400 group-hover:text-indigo-600'
                          }`}
                          aria-hidden="true"
                      />
                      <span>{label}</span>
                    </button>
                );
              })}
            </nav>

            {/* Filter info + Refresh */}
            <div className="flex flex-wrap gap-3 items-center">
              {(search || category || authorId) && (
                  <div className="flex items-center gap-2 bg-indigo-50 text-indigo-700 px-3 py-1 rounded-lg text-xs">
                    <Filter className="w-4 h-4" />
                    <span>
                  Đang lọc:
                      {search && ` "${search}"`}
                      {category && ` ${category}`}
                      {authorId && ` ${authorName ?? authorId}`}
                </span>
                  </div>
              )}

              <Button
                  variant="outline"
                  size="sm"
                  onClick={handleRefresh}
                  disabled={isLoading || isLoadingMore}
                  aria-label="Làm mới danh sách bài viết"
                  className="flex items-center gap-2"
              >
                <RefreshCw
                    className={`w-4 h-4 transition-transform ${isLoading ? 'animate-spin' : 'hover:rotate-180'}`}
                    aria-hidden="true"
                />
                <span className="hidden sm:inline">Làm mới</span>
              </Button>
            </div>
          </div>
        </header>

        {/* Form tạo bài viết */}
        {showCreatePost && (
            <section
                id="create-post-form"
                aria-label="Tạo bài viết mới"
                className="animate-slide-up"
            >
              <CreatePost onPostCreated={handlePostCreated} onCancel={() => setShowCreatePost(false)} />
            </section>
        )}

        {/* Error */}
        {error && <ErrorAlert message={error} onClose={() => setError(null)} />}

        {/* Enhanced Post List - Full Width */}
        <section
            aria-live="polite"
            aria-busy={isLoading}
            className="space-y-6"
            aria-label="Danh sách bài viết"
        >
          {!isLoading && posts.length === 0 ? (
              <div className="text-center py-16">
                <div className="w-24 h-24 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-6">
                  <MessageCircle className="w-12 h-12 text-gray-400" />
                </div>
                <p className="text-gray-500 text-xl vietnamese-text font-medium">Không có bài viết nào</p>
                <p className="text-gray-400 text-sm vietnamese-text mt-2">
                  {activeTab === 'latest' && 'Hãy tạo bài viết đầu tiên của bạn!'}
                  {activeTab === 'trending' && 'Chưa có bài viết thịnh hành nào.'}
                  {activeTab === 'top-liked' && 'Chưa có bài viết được yêu thích nào.'}
                </p>
              </div>
          ) : (
              posts.map((post) => (
                  <PostCard
                    key={post.id}
                    post={post}
                    onPostUpdate={handlePostUpdate}
                    onPostDelete={handlePostDelete}
                    className="w-full"
                  />
              ))
          )}
        </section>

        {/* Load More */}
        {hasMore && posts.length > 0 && (
            <div className="flex justify-center pt-4">
              <Button
                  variant="outline"
                  onClick={handleLoadMore}
                  disabled={isLoadingMore || isLoading}
                  loading={isLoadingMore}
                  aria-label="Xem thêm bài viết"
              >
                {isLoadingMore ? 'Đang tải...' : 'Xem thêm bài viết'}
              </Button>
            </div>
        )}

        {/* Loading Spinner */}
        {isLoadingMore && (
            <div className="flex justify-center py-4" aria-hidden="true">
              <LoadingSpinner />
            </div>
        )}
      </main>
  );
};
