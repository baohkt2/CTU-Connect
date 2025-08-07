'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Post, PaginatedResponse } from '@/types';
import { postService } from '@/services/postService';
import { PostCard } from '@/components/post/PostCard';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorAlert } from '@/components/ui/ErrorAlert';
import { Button } from '@/components/ui/Button';
import { RefreshCw, FileText, Image, Video, Clock } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';

interface ProfilePostFeedProps {
  userId: string;
  userName?: string;
  isOwnProfile?: boolean;
  className?: string;
}

export const ProfilePostFeed: React.FC<ProfilePostFeedProps> = ({
  userId,
  userName,
  isOwnProfile = false,
  className = ''
}) => {
  const [posts, setPosts] = useState<Post[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(true);
  const [currentPage, setCurrentPage] = useState(0);
  const [totalPosts, setTotalPosts] = useState(0);
  const [activeFilter, setActiveFilter] = useState<'all' | 'text' | 'image' | 'video'>('all');
  const {user} = useAuth();
  const loadUserPosts = useCallback(async (page = 0, append = false, filter = 'all') => {
    try {
      if (!append) {
        setIsLoading(true);
        setError(null);
      } else {
        setIsLoadingMore(true);
      }

      // Call the correct API endpoint based on whether it's own profile or not
      let response: PaginatedResponse<Post>;
      
      if (isOwnProfile) {
        // For own profile, use getMyPosts which calls /posts/me
        response = await postService.getMyPosts(page, 10);
      } else {
        // For other users, we need to implement getUserPosts in postService
        // For now, let's create a temporary implementation
        const apiResponse = await fetch(`/api/posts/user/${userId}?page=${page}&size=10`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`,
            'Content-Type': 'application/json'
          }
        });
        
        if (!apiResponse.ok) {
          throw new Error('Failed to fetch user posts');
        }
        
        response = await apiResponse.json();
      }

      if (append) {
        setPosts(prev => [...prev, ...response.content]);
      } else {
        setPosts(response.content);
      }

      setTotalPosts(response.totalElements);
      setHasMore(!response.last);
      setCurrentPage(response.number);

    } catch (err: any) {
      console.error('Error loading user posts:', err);
      setError('Không thể tải bài viết của người dùng');
    } finally {
      setIsLoading(false);
      setIsLoadingMore(false);
    }
  }, [userId, isOwnProfile]);

  useEffect(() => {
    loadUserPosts(0, false, activeFilter);
  }, [loadUserPosts, activeFilter]);

  const handleLoadMore = () => {
    if (hasMore && !isLoadingMore) {
      loadUserPosts(currentPage + 1, true, activeFilter);
    }
  };

  const handleRefresh = () => {
    loadUserPosts(0, false, activeFilter);
  };

  const handleFilterChange = (filter: 'all' | 'text' | 'image' | 'video') => {
    setActiveFilter(filter);
    setCurrentPage(0);
  };

  const handlePostUpdate = (updatedPost: Post) => {
    setPosts(prev => prev.map(post =>
      post.id === updatedPost.id ? updatedPost : post
    ));
  };

  const handlePostDelete = (postId: string) => {
    setPosts(prev => prev.filter(post => post.id !== postId));
    setTotalPosts(prev => prev - 1);
  };

  const getFilterIcon = (filter: string) => {
    switch (filter) {
      case 'text': return <FileText className="h-4 w-4" />;
      case 'image': return <Image className="h-4 w-4" />;
      case 'video': return <Video className="h-4 w-4" />;
      default: return <Clock className="h-4 w-4" />;
    }
  };

  const getFilterLabel = (filter: string) => {
    switch (filter) {
      case 'text': return 'Văn bản';
      case 'image': return 'Hình ảnh';
      case 'video': return 'Video';
      default: return 'Tất cả';
    }
  };

  if (isLoading) {
    return (
      <div className={`space-y-6 ${className}`}>
        <div className="flex justify-center py-12">
          <div className="text-center">
            <LoadingSpinner size="lg" />
            <p className="text-gray-600 mt-4 vietnamese-text">
              Đang tải bài viết...
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className={`space-y-6 ${className}`}>
        <ErrorAlert
          message={error}
          onRetry={handleRefresh}
        />
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header with Stats and Filters */}
      <div className="bg-white rounded-lg shadow-sm p-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          {/* Post Count */}
          <div>
            <h2 className="text-xl font-bold text-gray-900 vietnamese-text">
              Bài viết của {isOwnProfile ? 'bạn' : (userName || 'người dùng')}
            </h2>
            <p className="text-sm text-gray-600 vietnamese-text">
              {totalPosts} bài viết
            </p>
          </div>

          {/* Refresh Button */}
          <Button
            variant="outline"
            onClick={handleRefresh}
            className="flex items-center space-x-2"
            disabled={isLoading}
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            <span className="vietnamese-text">Làm mới</span>
          </Button>
        </div>

        {/* Filter Tabs */}
        <div className="mt-6 border-b border-gray-200">
          <nav className="flex space-x-8">
            {[
              { key: 'all', label: 'Tất cả' },
              { key: 'text', label: 'Văn bản' },
              { key: 'image', label: 'Hình ảnh' },
              { key: 'video', label: 'Video' }
            ].map((filter) => (
              <button
                key={filter.key}
                onClick={() => handleFilterChange(filter.key as any)}
                className={`py-2 px-1 border-b-2 font-medium text-sm vietnamese-text transition-colors flex items-center space-x-2 ${
                  activeFilter === filter.key
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {getFilterIcon(filter.key)}
                <span>{filter.label}</span>
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Posts List */}
      {posts.length === 0 ? (
        <div className="bg-white rounded-lg shadow-sm p-12 text-center">
          <div className="text-gray-400 mb-4">
            <FileText className="h-16 w-16 mx-auto" />
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-2 vietnamese-text">
            {isOwnProfile ? 'Bạn chưa có bài viết nào' : 'Người dùng này chưa có bài viết nào'}
          </h3>
          <p className="text-gray-600 vietnamese-text">
            {isOwnProfile
              ? 'Hãy tạo bài viết đầu tiên để chia sẻ với mọi người!'
              : 'Hãy quay lại sau để xem bài viết mới nhất.'
            }
          </p>
          {isOwnProfile && (
            <Button
              className="mt-4"
              onClick={() => window.location.href = '/posts/create'}
            >
              Tạo bài viết đầu tiên
            </Button>
          )}
        </div>
      ) : (
        <>
          {/* Posts */}
          <div className="space-y-6">
            {posts.map((post) => (
              <PostCard
                key={post.id}
                post={post}
                onPostUpdate={handlePostUpdate}
                onPostDelete={handlePostDelete}
                className="shadow-sm hover:shadow-md transition-shadow"
              />
            ))}
          </div>

          {/* Load More Button */}
          {hasMore && (
            <div className="text-center pt-6">
              <Button
                variant="outline"
                onClick={handleLoadMore}
                disabled={isLoadingMore}
                className="px-8"
              >
                {isLoadingMore ? (
                  <>
                    <LoadingSpinner size="sm" className="mr-2" />
                    <span className="vietnamese-text">Đang tải...</span>
                  </>
                ) : (
                  <span className="vietnamese-text">Xem thêm bài viết</span>
                )}
              </Button>
            </div>
          )}

          {/* End Message */}
          {!hasMore && posts.length > 0 && (
            <div className="text-center py-8">
              <p className="text-gray-500 vietnamese-text">
                Bạn đã xem hết tất cả bài viết
              </p>
            </div>
          )}
        </>
      )}
    </div>
  );
};
