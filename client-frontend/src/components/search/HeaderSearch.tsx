'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Search, X } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { searchService, SearchSuggestionResponse } from '@/services/searchService';
import { debounce } from 'lodash';

interface HeaderSearchProps {
  className?: string;
}

const HeaderSearch: React.FC<HeaderSearchProps> = ({ className = '' }) => {
  const router = useRouter();
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState<SearchSuggestionResponse | null>(null);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Debounced function to get suggestions
  const debouncedGetSuggestions = debounce(async (searchQuery: string) => {
    if (searchQuery.length >= 2) {
      try {
        setIsLoading(true);
        const suggestionResponse = await searchService.getSearchSuggestions(searchQuery);
        setSuggestions(suggestionResponse);
        setShowSuggestions(true);
      } catch (error) {
        console.error('Error fetching suggestions:', error);
      } finally {
        setIsLoading(false);
      }
    } else {
      setSuggestions(null);
      setShowSuggestions(false);
    }
  }, 300);

  // Handle input change
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
    debouncedGetSuggestions(value);
  };

  // Handle search submission
  const handleSearch = (searchQuery?: string) => {
    const finalQuery = searchQuery || query;
    if (!finalQuery.trim()) return;

    setShowSuggestions(false);
    const params = new URLSearchParams();
    params.set('q', finalQuery.trim());
    router.push(`/search?${params.toString()}`);
  };

  // Handle suggestion click
  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion);
    setShowSuggestions(false);
    handleSearch(suggestion);
  };

  // Handle key press
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    } else if (e.key === 'Escape') {
      setShowSuggestions(false);
      inputRef.current?.blur();
    }
  };

  // Handle click outside to close suggestions
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Clear search
  const clearSearch = () => {
    setQuery('');
    setSuggestions(null);
    setShowSuggestions(false);
    inputRef.current?.focus();
  };

  return (
    <div ref={searchRef} className={`relative ${className}`}>
      <div className="relative">
        <div className="flex items-center bg-gray-100 hover:bg-gray-200 transition-colors rounded-full">
          <Search className="ml-3 h-4 w-4 text-gray-500" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={handleInputChange}
            onKeyDown={handleKeyPress}
            onFocus={() => query.length >= 2 && suggestions && setShowSuggestions(true)}
            placeholder="Tìm kiếm..."
            className="flex-1 px-3 py-2 text-sm bg-transparent placeholder-gray-500 border-none outline-none"
          />
          {query && (
            <button
              onClick={clearSearch}
              className="mr-2 p-1 text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          )}
          {isLoading && (
            <div className="mr-3">
              <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-gray-400"></div>
            </div>
          )}
        </div>
      </div>

      {/* Quick Suggestions Dropdown */}
      {showSuggestions && suggestions && (
        <div className="absolute z-50 w-full mt-2 bg-white border border-gray-200 rounded-lg shadow-lg max-h-80 overflow-y-auto">
          {/* Quick Actions */}
          {query && (
            <div className="p-2 border-b border-gray-100">
              <button
                onClick={() => handleSearch()}
                className="flex items-center w-full px-3 py-2 text-sm text-left hover:bg-gray-100 rounded"
              >
                <Search className="w-4 h-4 mr-3 text-gray-400" />
                Tìm kiếm "{query}"
              </button>
            </div>
          )}

          {/* Title Suggestions */}
          {suggestions.titleSuggestions.length > 0 && (
            <div className="p-2">
              <div className="text-xs font-semibold text-gray-500 mb-1 px-2">Bài viết</div>
              {suggestions.titleSuggestions.slice(0, 3).map((suggestion, index) => (
                <button
                  key={`title-${index}`}
                  onClick={() => handleSuggestionClick(suggestion)}
                  className="block w-full text-left px-3 py-2 text-sm hover:bg-gray-100 rounded truncate"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          )}

          {/* Category Suggestions */}
          {suggestions.categorySuggestions.length > 0 && (
            <div className="p-2 border-t border-gray-100">
              <div className="text-xs font-semibold text-gray-500 mb-1 px-2">Danh mục</div>
              <div className="flex flex-wrap gap-1 px-2">
                {suggestions.categorySuggestions.slice(0, 4).map((suggestion, index) => (
                  <button
                    key={`category-${index}`}
                    onClick={() => {
                      const params = new URLSearchParams();
                      params.set('category', suggestion);
                      router.push(`/search?${params.toString()}`);
                      setShowSuggestions(false);
                    }}
                    className="px-2 py-1 text-xs bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-full"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Trending Queries */}
          {suggestions.trendingQueries.length > 0 && (
            <div className="p-2 border-t border-gray-100">
              <div className="text-xs font-semibold text-gray-500 mb-1 px-2">Phổ biến</div>
              {suggestions.trendingQueries.slice(0, 3).map((suggestion, index) => (
                <button
                  key={`trending-${index}`}
                  onClick={() => handleSuggestionClick(suggestion)}
                  className="block w-full text-left px-3 py-2 text-sm hover:bg-gray-100 rounded text-orange-600"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          )}

          {/* View All Results Link */}
          {query && (
            <div className="p-2 border-t border-gray-100">
              <button
                onClick={() => handleSearch()}
                className="w-full px-3 py-2 text-sm text-center text-blue-600 hover:bg-blue-50 rounded font-medium"
              >
                Xem tất cả kết quả cho "{query}"
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default HeaderSearch;
