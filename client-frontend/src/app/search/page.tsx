'use client';

import React, { useState, useCallback } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import SearchBar from '@/components/search/SearchBar';
import SearchResults from '@/components/search/SearchResults';
import TrendingSearches from '@/components/search/TrendingSearches';
import { SearchResponse, PostResponse } from '@/services/searchService';

const SearchPage: React.FC = () => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [searchResults, setSearchResults] = useState<SearchResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(0);

  // Get initial query from URL params
  const initialQuery = searchParams.get('q') || '';
  const initialCategory = searchParams.get('category') || '';

  const handleSearchResults = useCallback((results: SearchResponse) => {
    setSearchResults(results);
    setCurrentPage(results.currentPage);

    // Update URL with search parameters
    const params = new URLSearchParams();
    if (results.searchQuery) {
      params.set('q', results.searchQuery);
    }
    if (results.filtersApplied?.category) {
      params.set('category', results.filtersApplied.category);
    }

    const newUrl = `/search${params.toString() ? `?${params.toString()}` : ''}`;
    router.push(newUrl, { scroll: false });
  }, [router]);

  const handleSearchLoading = useCallback((loading: boolean) => {
    setIsLoading(loading);
  }, []);

  const handleLoadMore = useCallback(async () => {
    if (!searchResults || isLoading) return;

    // This would typically call the search service again with the next page
    // For now, we'll just show a loading state
    setIsLoading(true);

    // Simulate loading delay
    setTimeout(() => {
      setIsLoading(false);
    }, 1000);
  }, [searchResults, isLoading]);

  const handlePostClick = useCallback((post: PostResponse) => {
    router.push(`/posts/${post.id}`);
  }, [router]);

  const handleTrendingClick = useCallback((term: string) => {
    // Trigger a new search with the trending term
    // This would be handled by the SearchBar component
    const params = new URLSearchParams();
    params.set('q', term);
    router.push(`/search?${params.toString()}`);
  }, [router]);

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Page Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Tìm kiếm bài viết</h1>
          <p className="text-gray-600">
            Khám phá và tìm kiếm nội dung trong cộng đồng CTU Connect
          </p>
        </div>

        {/* Search Bar */}
        <div className="mb-8">
          <SearchBar
            onSearchResults={handleSearchResults}
            onSearchLoading={handleSearchLoading}
            placeholder="Tìm kiếm bài viết, tác giả, thẻ..."
            showAdvancedFilters={true}
            className="max-w-4xl mx-auto"
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-3">
            {searchResults ? (
              <SearchResults
                searchResults={searchResults}
                loading={isLoading}
                onLoadMore={handleLoadMore}
                onPostClick={handlePostClick}
              />
            ) : (
              <div className="text-center py-12">
                <div className="text-gray-400 mb-4">
                  <svg className="mx-auto h-16 w-16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                </div>
                <h3 className="text-xl font-medium text-gray-900 mb-2">
                  Tìm kiếm nội dung mà bạn quan tâm
                </h3>
                <p className="text-gray-500 mb-6">
                  Sử dụng thanh tìm kiếm phía trên để khám phá bài viết, tác giả và chủ đề
                </p>

                {/* Search Examples */}
                <div className="text-left max-w-md mx-auto">
                  <h4 className="text-sm font-semibold text-gray-700 mb-3">Gợi ý tìm kiếm:</h4>
                  <div className="space-y-2">
                    <button
                      onClick={() => handleTrendingClick('lập trình')}
                      className="block w-full text-left px-3 py-2 text-sm bg-white border border-gray-200 rounded-lg hover:bg-gray-50 hover:border-blue-300 transition-colors"
                    >
                      🔍 "lập trình" - Tìm bài viết về lập trình
                    </button>
                    <button
                      onClick={() => handleTrendingClick('sinh viên')}
                      className="block w-full text-left px-3 py-2 text-sm bg-white border border-gray-200 rounded-lg hover:bg-gray-50 hover:border-blue-300 transition-colors"
                    >
                      👥 "sinh viên" - Tìm nội dung về sinh viên
                    </button>
                    <button
                      onClick={() => handleTrendingClick('học tập')}
                      className="block w-full text-left px-3 py-2 text-sm bg-white border border-gray-200 rounded-lg hover:bg-gray-50 hover:border-blue-300 transition-colors"
                    >
                      📚 "học tập" - Tìm tài liệu học tập
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="lg:col-span-1">
            <div className="space-y-6">
              {/* Trending Searches */}
              <TrendingSearches
                onTrendingClick={handleTrendingClick}
              />

              {/* Search Tips */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Mẹo tìm kiếm
                </h3>
                <div className="space-y-3 text-sm text-gray-600">
                  <div className="flex items-start space-x-2">
                    <span className="text-blue-600 font-medium">•</span>
                    lodash     <span>Sử dụng dấu ngoặc kép để tìm cụm từ chính xác: "machine learning"</span>
                  </div>
                  <div className="flex items-start space-x-2">
                    <span className="text-blue-600 font-medium">•</span>
                    <span>Sử dụng thẻ để lọc theo chủ đề: #javascript #programming</span>
                  </div>
                  <div className="flex items-start space-x-2">
                    <span className="text-blue-600 font-medium">•</span>
                    <span>Sử dụng bộ lọc nâng cao để thu hẹp kết quả</span>
                  </div>
                  <div className="flex items-start space-x-2">
                    <span className="text-blue-600 font-medium">•</span>
                    <span>Tìm kiếm theo tên tác giả để xem tất cả bài viết của họ</span>
                  </div>
                </div>
              </div>

              {/* Quick Categories */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Danh mục phổ biến
                </h3>
                <div className="grid grid-cols-1 gap-2">
                  {[
                    { name: 'Học tập', value: 'academic', icon: '📚' },
                    { name: 'Công nghệ', value: 'technology', icon: '💻' },
                    { name: 'Thể thao', value: 'sports', icon: '⚽' },
                    { name: 'Văn hóa', value: 'culture', icon: '🎨' },
                    { name: 'Xã hội', value: 'social', icon: '👥' }
                  ].map((category) => (
                    <button
                      key={category.value}
                      onClick={() => {
                        const params = new URLSearchParams();
                        params.set('category', category.value);
                        router.push(`/search?${params.toString()}`);
                      }}
                      className="flex items-center space-x-2 px-3 py-2 text-sm text-left bg-gray-50 hover:bg-blue-50 hover:text-blue-700 rounded-lg transition-colors"
                    >
                      <span>{category.icon}</span>
                      <span>{category.name}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SearchPage;
