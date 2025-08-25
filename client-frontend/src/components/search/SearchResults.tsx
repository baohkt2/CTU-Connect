'use client';

import React from 'react';
import { Clock, Heart, MessageCircle, Eye, Share2, User } from 'lucide-react';
import { SearchResponse, PostResponse } from '@/services/searchService';
import Link from 'next/link';

interface SearchResultsProps {
  searchResults: SearchResponse;
  loading?: boolean;
  onLoadMore?: () => void;
  onPostClick?: (post: PostResponse) => void;
}

const SearchResults: React.FC<SearchResultsProps> = ({
  searchResults,
  loading = false,
  onLoadMore,
  onPostClick
}) => {
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const days = Math.floor(diff / (1000 * 60 * 60 * 24));

    if (days === 0) {
      const hours = Math.floor(diff / (1000 * 60 * 60));
      if (hours === 0) {
        const minutes = Math.floor(diff / (1000 * 60));
        return `${minutes} phút trước`;
      }
      return `${hours} giờ trước`;
    } else if (days === 1) {
      return 'Hôm qua';
    } else if (days < 7) {
      return `${days} ngày trước`;
    } else {
      return date.toLocaleDateString('vi-VN');
    }
  };

  const truncateContent = (content: string, maxLength: number = 200) => {
    if (content.length <= maxLength) return content;
    return content.substring(0, maxLength) + '...';
  };

  const highlightSearchTerm = (text: string, searchQuery: string) => {
    if (!searchQuery) return text;

    const regex = new RegExp(`(${searchQuery})`, 'gi');
    const parts = text.split(regex);

    return parts.map((part, index) =>
      regex.test(part) ? (
        <mark key={index} className="bg-yellow-200 px-1 rounded">
          {part}
        </mark>
      ) : (
        part
      )
    );
  };

  if (loading) {
    return (
      <div className="space-y-4">
        {[...Array(5)].map((_, index) => (
          <div key={index} className="bg-white rounded-lg border border-gray-200 p-4 animate-pulse">
            <div className="flex items-center space-x-3 mb-3">
              <div className="w-10 h-10 bg-gray-300 rounded-full"></div>
              <div className="flex-1">
                <div className="h-4 bg-gray-300 rounded w-1/4 mb-2"></div>
                <div className="h-3 bg-gray-300 rounded w-1/6"></div>
              </div>
            </div>
            <div className="space-y-2">
              <div className="h-4 bg-gray-300 rounded w-3/4"></div>
              <div className="h-4 bg-gray-300 rounded w-1/2"></div>
              <div className="h-20 bg-gray-300 rounded"></div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (!searchResults.posts || searchResults.posts.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="text-gray-400 mb-4">
          <svg className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 15c-2.34 0-4.54-.94-6.14-2.6C4.46 11 4 9.78 4 8.5c0-3.314 2.686-6 6-6s6 2.686 6 6c0 1.28-.46 2.5-1.86 3.9A7.962 7.962 0 0112 15z" />
          </svg>
        </div>
        <h3 className="text-lg font-medium text-gray-900 mb-2">Không tìm thấy kết quả</h3>
        <p className="text-gray-500">
          Thử sử dụng từ khóa khác hoặc điều chỉnh bộ lọc tìm kiếm của bạn.
        </p>
        {searchResults.suggestedQueries && searchResults.suggestedQueries.length > 0 && (
          <div className="mt-4">
            <p className="text-sm text-gray-600 mb-2">Có thể bạn muốn tìm:</p>
            <div className="flex flex-wrap gap-2 justify-center">
              {searchResults.suggestedQueries.map((query, index) => (
                <button
                  key={index}
                  className="px-3 py-1 text-sm bg-blue-100 text-blue-700 rounded-full hover:bg-blue-200"
                >
                  {query}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Search Metadata */}
      <div className="bg-gray-50 rounded-lg p-4">
        <div className="flex items-center justify-between text-sm text-gray-600">
          <div>
            <span className="font-medium">{searchResults.totalElements.toLocaleString()}</span> kết quả
            {searchResults.searchQuery && (
              <span> cho "<span className="font-medium">{searchResults.searchQuery}</span>"</span>
            )}
          </div>
          <div>
            Thời gian tìm kiếm: {searchResults.searchTimeMs}ms
          </div>
        </div>

        {/* Applied Filters */}
        {searchResults.filtersApplied && Object.keys(searchResults.filtersApplied).length > 0 && (
          <div className="mt-2 flex flex-wrap gap-2">
            {Object.entries(searchResults.filtersApplied).map(([key, value]) => (
              <span
                key={key}
                className="inline-flex items-center px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded-full"
              >
                {key}: {Array.isArray(value) ? value.join(', ') : value.toString()}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Search Results */}
      <div className="space-y-4">
        {searchResults.posts.map((post) => (
          <div
            key={post.id}
            className="bg-white rounded-lg border border-gray-200 hover:shadow-md transition-shadow p-4 cursor-pointer"
            onClick={() => onPostClick?.(post)}
          >
            {/* Author Info */}
            <div className="flex items-center space-x-3 mb-3">
              <div className="w-10 h-10 bg-gray-300 rounded-full flex items-center justify-center">
                {post.author.avatarUrl ? (
                  <img
                    src={post.author.avatarUrl}
                    alt={post.author.name}
                    className="w-full h-full rounded-full object-cover"
                  />
                ) : (
                  <User className="w-5 h-5 text-gray-500" />
                )}
              </div>
              <div className="flex-1">
                <div className="flex items-center space-x-2">
                  <Link
                    href={`/profile/${post.author.id}`}
                    className="font-medium text-gray-900 hover:text-blue-600"
                    onClick={(e) => e.stopPropagation()}
                  >
                    {highlightSearchTerm(post.author.name, searchResults.searchQuery)}
                  </Link>
                  {post.category && (
                    <span className="px-2 py-1 text-xs bg-gray-100 text-gray-600 rounded-full">
                      {post.category}
                    </span>
                  )}
                </div>
                <div className="flex items-center space-x-1 text-sm text-gray-500">
                  <Clock className="w-4 h-4" />
                  <span>{formatDate(post.createdAt)}</span>
                </div>
              </div>
            </div>

            {/* Post Title */}
            {post.title && (
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                {highlightSearchTerm(post.title, searchResults.searchQuery)}
              </h3>
            )}

            {/* Post Content */}
            <div className="text-gray-700 mb-3">
              {highlightSearchTerm(truncateContent(post.content), searchResults.searchQuery)}
            </div>

            {/* Post Images */}
            {post.images && post.images.length > 0 && (
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mb-3">
                {post.images.slice(0, 3).map((image, index) => (
                  <div key={index} className="relative">
                    <img
                      src={image}
                      alt={`Post image ${index + 1}`}
                      className="w-full h-24 object-cover rounded"
                    />
                    {index === 2 && post.images!.length > 3 && (
                      <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center rounded">
                        <span className="text-white font-medium">+{post.images!.length - 3}</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Post Tags */}
            {post.tags && post.tags.length > 0 && (
              <div className="flex flex-wrap gap-1 mb-3">
                {post.tags.slice(0, 5).map((tag, index) => (
                  <span
                    key={index}
                    className="inline-flex items-center px-2 py-1 text-xs bg-blue-50 text-blue-700 rounded-full hover:bg-blue-100"
                  >
                    #{highlightSearchTerm(tag, searchResults.searchQuery)}
                  </span>
                ))}
                {post.tags.length > 5 && (
                  <span className="text-xs text-gray-500">+{post.tags.length - 5} thêm</span>
                )}
              </div>
            )}

            {/* Post Stats */}
            <div className="flex items-center justify-between text-sm text-gray-500">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-1">
                  <Heart className="w-4 h-4" />
                  <span>{post.stats.likes}</span>
                </div>
                <div className="flex items-center space-x-1">
                  <MessageCircle className="w-4 h-4" />
                  <span>{post.stats.comments}</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Eye className="w-4 h-4" />
                  <span>{post.stats.views}</span>
                </div>
                {post.stats.shares && post.stats.shares > 0 && (
                  <div className="flex items-center space-x-1">
                    <Share2 className="w-4 h-4" />
                    <span>{post.stats.shares}</span>
                  </div>
                )}
              </div>
              <Link
                href={`/posts/${post.id}`}
                className="text-blue-600 hover:text-blue-800 font-medium"
                onClick={(e) => e.stopPropagation()}
              >
                Xem chi tiết →
              </Link>
            </div>
          </div>
        ))}
      </div>

      {/* Load More Button */}
      {searchResults.currentPage + 1 < searchResults.totalPages && (
        <div className="text-center">
          <button
            onClick={onLoadMore}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Tải thêm kết quả
          </button>
        </div>
      )}

      {/* Related Searches */}
      {searchResults.relatedSearches && searchResults.relatedSearches.length > 0 && (
        <div className="bg-gray-50 rounded-lg p-4">
          <h3 className="text-sm font-medium text-gray-700 mb-2">Tìm kiếm liên quan:</h3>
          <div className="flex flex-wrap gap-2">
            {searchResults.relatedSearches.map((relatedSearch, index) => (
              <button
                key={index}
                className="px-3 py-1 text-sm bg-white text-gray-700 border border-gray-200 rounded-full hover:bg-gray-100"
              >
                {relatedSearch}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default SearchResults;
