'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Search, Filter, X, TrendingUp } from 'lucide-react';
import { searchService, SearchRequest, SearchResponse, SearchSuggestionResponse } from '@/services/searchService';
import { debounce } from 'lodash';

interface SearchBarProps {
  onSearchResults?: (results: SearchResponse) => void;
  onSearchLoading?: (loading: boolean) => void;
  placeholder?: string;
  showAdvancedFilters?: boolean;
  className?: string;
}

const SearchBar: React.FC<SearchBarProps> = ({
  onSearchResults,
  onSearchLoading,
  placeholder = "Tìm kiếm bài viết...",
  showAdvancedFilters = true,
  className = ""
}) => {
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState<SearchSuggestionResponse | null>(null);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState<Partial<SearchRequest>>({
    sortBy: 'relevance',
    sortDirection: 'desc',
    page: 0,
    size: 10
  });
  const [isLoading, setIsLoading] = useState(false);

  // Debounced function to get suggestions
  const debouncedGetSuggestions = useCallback(
    debounce(async (searchQuery: string) => {
      if (searchQuery.length >= 2) {
        try {
          const suggestionResponse = await searchService.getSearchSuggestions(searchQuery);
          setSuggestions(suggestionResponse);
          setShowSuggestions(true);
        } catch (error) {
          console.error('Error fetching suggestions:', error);
        }
      } else {
        setSuggestions(null);
        setShowSuggestions(false);
      }
    }, 300),
    []
  );

  // Handle input change
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
    debouncedGetSuggestions(value);
  };

  // Handle search submission
  const handleSearch = async (searchQuery?: string) => {
    const finalQuery = searchQuery || query;
    if (!finalQuery.trim()) return;

    setIsLoading(true);
    onSearchLoading?.(true);
    setShowSuggestions(false);

    try {
      const searchRequest: SearchRequest = {
        ...filters,
        query: finalQuery.trim()
      };

      const results = await searchService.advancedSearch(searchRequest);
      onSearchResults?.(results);
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setIsLoading(false);
      onSearchLoading?.(false);
    }
  };

  // Handle suggestion click
  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion);
    setShowSuggestions(false);
    handleSearch(suggestion);
  };

  // Handle filter change
  const handleFilterChange = (key: keyof SearchRequest, value: any) => {
    setFilters(prev => ({
      ...prev,
      [key]: value
    }));
  };

  // Handle Enter key press
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    } else if (e.key === 'Escape') {
      setShowSuggestions(false);
    }
  };

  // Close suggestions when clicking outside
  useEffect(() => {
    const handleClickOutside = () => {
      setShowSuggestions(false);
    };

    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  return (
    <div className={`relative ${className}`}>
      {/* Main Search Input */}
      <div className="relative">
        <div className="flex items-center bg-white border border-gray-300 rounded-lg shadow-sm">
          <Search className="ml-3 h-5 w-5 text-gray-400" />
          <input
            type="text"
            value={query}
            onChange={handleInputChange}
            onKeyDown={handleKeyPress}
            placeholder={placeholder}
            className="flex-1 px-3 py-2 text-gray-900 placeholder-gray-500 bg-transparent border-none outline-none"
            onClick={(e) => e.stopPropagation()}
          />
          {showAdvancedFilters && (
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={`p-2 rounded-r-lg transition-colors ${
                showFilters ? 'bg-blue-100 text-blue-600' : 'text-gray-400 hover:text-gray-600'
              }`}
            >
              <Filter className="h-5 w-5" />
            </button>
          )}
        </div>

        {/* Loading indicator */}
        {isLoading && (
          <div className="absolute right-10 top-1/2 transform -translate-y-1/2">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
          </div>
        )}
      </div>

      {/* Search Suggestions Dropdown */}
      {showSuggestions && suggestions && (
        <div className="absolute z-50 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-96 overflow-y-auto">
          {/* Title Suggestions */}
          {suggestions.titleSuggestions.length > 0 && (
            <div className="p-2 border-b border-gray-100">
              <div className="text-xs font-semibold text-gray-500 mb-1">Tiêu đề</div>
              {suggestions.titleSuggestions.map((suggestion, index) => (
                <button
                  key={`title-${index}`}
                  onClick={() => handleSuggestionClick(suggestion)}
                  className="block w-full text-left px-2 py-1 text-sm hover:bg-gray-100 rounded"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          )}

          {/* Category Suggestions */}
          {suggestions.categorySuggestions.length > 0 && (
            <div className="p-2 border-b border-gray-100">
              <div className="text-xs font-semibold text-gray-500 mb-1">Danh mục</div>
              {suggestions.categorySuggestions.map((suggestion, index) => (
                <button
                  key={`category-${index}`}
                  onClick={() => {
                    handleFilterChange('category', suggestion);
                    handleSearch();
                  }}
                  className="block w-full text-left px-2 py-1 text-sm hover:bg-gray-100 rounded text-blue-600"
                >
                  #{suggestion}
                </button>
              ))}
            </div>
          )}

          {/* Tag Suggestions */}
          {suggestions.tagSuggestions.length > 0 && (
            <div className="p-2 border-b border-gray-100">
              <div className="text-xs font-semibold text-gray-500 mb-1">Thẻ</div>
              {suggestions.tagSuggestions.map((suggestion, index) => (
                <button
                  key={`tag-${index}`}
                  onClick={() => {
                    const currentTags = filters.tags || [];
                    if (!currentTags.includes(suggestion)) {
                      handleFilterChange('tags', [...currentTags, suggestion]);
                      handleSearch();
                    }
                  }}
                  className="inline-block px-2 py-1 m-1 text-sm bg-gray-100 hover:bg-gray-200 rounded-full text-gray-700"
                >
                  #{suggestion}
                </button>
              ))}
            </div>
          )}

          {/* Trending Queries */}
          {suggestions.trendingQueries.length > 0 && (
            <div className="p-2">
              <div className="flex items-center text-xs font-semibold text-gray-500 mb-1">
                <TrendingUp className="h-3 w-3 mr-1" />
                Tìm kiếm phổ biến
              </div>
              {suggestions.trendingQueries.map((suggestion, index) => (
                <button
                  key={`trending-${index}`}
                  onClick={() => handleSuggestionClick(suggestion)}
                  className="block w-full text-left px-2 py-1 text-sm hover:bg-gray-100 rounded text-orange-600"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Advanced Filters Panel */}
      {showFilters && (
        <div className="absolute z-40 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-gray-700">Bộ lọc nâng cao</h3>
            <button
              onClick={() => setShowFilters(false)}
              className="text-gray-400 hover:text-gray-600"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Category Filter */}
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Danh mục
              </label>
              <select
                value={filters.category || ''}
                onChange={(e) => handleFilterChange('category', e.target.value || undefined)}
                className="w-full px-3 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:border-blue-500"
              >
                <option value="">Tất cả danh mục</option>
                <option value="academic">Học tập</option>
                <option value="social">Xã hội</option>
                <option value="technology">Công nghệ</option>
                <option value="sports">Thể thao</option>
                <option value="culture">Văn hóa</option>
              </select>
            </div>

            {/* Sort By */}
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Sắp xếp theo
              </label>
              <select
                value={filters.sortBy || 'relevance'}
                onChange={(e) => handleFilterChange('sortBy', e.target.value)}
                className="w-full px-3 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:border-blue-500"
              >
                <option value="relevance">Liên quan</option>
                <option value="date">Ngày tạo</option>
                <option value="likes">Lượt thích</option>
                <option value="views">Lượt xem</option>
                <option value="comments">Bình luận</option>
              </select>
            </div>

            {/* Date Range */}
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Từ ngày
              </label>
              <input
                type="date"
                value={filters.dateFrom || ''}
                onChange={(e) => handleFilterChange('dateFrom', e.target.value || undefined)}
                className="w-full px-3 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Đến ngày
              </label>
              <input
                type="date"
                value={filters.dateTo || ''}
                onChange={(e) => handleFilterChange('dateTo', e.target.value || undefined)}
                className="w-full px-3 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:border-blue-500"
              />
            </div>

            {/* Minimum Likes */}
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Tối thiểu lượt thích
              </label>
              <input
                type="number"
                min="0"
                value={filters.minLikes || ''}
                onChange={(e) => handleFilterChange('minLikes', e.target.value ? parseInt(e.target.value) : undefined)}
                className="w-full px-3 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:border-blue-500"
                placeholder="0"
              />
            </div>

            {/* Post Type */}
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Loại bài viết
              </label>
              <select
                value={filters.postType || ''}
                onChange={(e) => handleFilterChange('postType', e.target.value || undefined)}
                className="w-full px-3 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:border-blue-500"
              >
                <option value="">Tất cả</option>
                <option value="TEXT">Văn bản</option>
                <option value="IMAGE">Hình ảnh</option>
                <option value="VIDEO">Video</option>
                <option value="DOCUMENT">Tài liệu</option>
              </select>
            </div>
          </div>

          {/* Selected Tags */}
          {filters.tags && filters.tags.length > 0 && (
            <div className="mt-3">
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Thẻ đã chọn
              </label>
              <div className="flex flex-wrap gap-1">
                {filters.tags.map((tag, index) => (
                  <span
                    key={index}
                    className="inline-flex items-center px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded-full"
                  >
                    #{tag}
                    <button
                      onClick={() => {
                        const newTags = filters.tags?.filter(t => t !== tag);
                        handleFilterChange('tags', newTags?.length ? newTags : undefined);
                      }}
                      className="ml-1 text-blue-600 hover:text-blue-800"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Apply Filters Button */}
          <div className="mt-4 flex justify-end space-x-2">
            <button
              onClick={() => {
                setFilters({
                  sortBy: 'relevance',
                  sortDirection: 'desc',
                  page: 0,
                  size: 10
                });
              }}
              className="px-3 py-1 text-sm text-gray-600 hover:text-gray-800"
            >
              Đặt lại
            </button>
            <button
              onClick={() => {
                handleSearch();
                setShowFilters(false);
              }}
              className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Áp dụng bộ lọc
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default SearchBar;
