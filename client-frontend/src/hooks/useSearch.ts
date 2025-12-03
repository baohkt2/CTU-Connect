'use client';

import { useState, useCallback, useRef } from 'react';
import { searchService, SearchRequest, SearchResponse, SearchSuggestionResponse } from '@/services/searchService';

export interface UseSearchOptions {
  debounceMs?: number;
  autoSearch?: boolean;
  defaultPageSize?: number;
}

export interface UseSearchReturn {
  // State
  searchResults: SearchResponse | null;
  suggestions: SearchSuggestionResponse | null;
  loading: boolean;
  error: string | null;

  // Actions
  search: (query: string, filters?: Partial<SearchRequest>) => Promise<void>;
  advancedSearch: (searchRequest: SearchRequest) => Promise<void>;
  getSuggestions: (query: string) => Promise<void>;
  loadMore: () => Promise<void>;
  clearResults: () => void;

  // Utilities
  hasMore: boolean;
  totalResults: number;
}

export const useSearch = (options: UseSearchOptions = {}): UseSearchReturn => {
  const {
    debounceMs = 300,
    autoSearch = false,
    defaultPageSize = 10
  } = options;

  // State
  const [searchResults, setSearchResults] = useState<SearchResponse | null>(null);
  const [suggestions, setSuggestions] = useState<SearchSuggestionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentSearchRequest, setCurrentSearchRequest] = useState<SearchRequest | null>(null);

  // Refs for debouncing
  const debounceRef = useRef<NodeJS.Timeout>();

  // Clear previous search results
  const clearResults = useCallback(() => {
    setSearchResults(null);
    setSuggestions(null);
    setError(null);
    setCurrentSearchRequest(null);
  }, []);

  // Basic search function
  const search = useCallback(async (query: string, filters: Partial<SearchRequest> = {}) => {
    if (!query.trim()) {
      clearResults();
      return;
    }

    const searchRequest: SearchRequest = {
      query: query.trim(),
      page: 0,
      size: defaultPageSize,
      sortBy: 'relevance',
      sortDirection: 'desc',
      ...filters
    };

    try {
      setLoading(true);
      setError(null);
      setCurrentSearchRequest(searchRequest);

      const results = await searchService.advancedSearch(searchRequest);
      setSearchResults(results);
    } catch (err: any) {
      console.error('Search error:', err);
      setError(err.message || 'Có lỗi xảy ra khi tìm kiếm');
    } finally {
      setLoading(false);
    }
  }, [defaultPageSize]);

  // Advanced search function
  const advancedSearch = useCallback(async (searchRequest: SearchRequest) => {
    try {
      setLoading(true);
      setError(null);
      setCurrentSearchRequest(searchRequest);

      const results = await searchService.advancedSearch(searchRequest);
      setSearchResults(results);
    } catch (err: any) {
      console.error('Advanced search error:', err);
      setError(err.message || 'Có lỗi xảy ra khi tìm kiếm');
    } finally {
      setLoading(false);
    }
  }, []);

  // Get search suggestions
  const getSuggestions = useCallback(async (query: string) => {
    if (!query.trim() || query.length < 2) {
      setSuggestions(null);
      return;
    }

    // Clear previous debounce
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    // Debounce the API call
    debounceRef.current = setTimeout(async () => {
      try {
        const suggestionResults = await searchService.getSearchSuggestions(query.trim());
        setSuggestions(suggestionResults);
      } catch (err) {
        console.error('Error getting suggestions:', err);
        setSuggestions(null);
      }
    }, debounceMs);
  }, [debounceMs]);

  // Load more results (pagination)
  const loadMore = useCallback(async () => {
    if (!currentSearchRequest || !searchResults || loading) {
      return;
    }

    if (searchResults.currentPage + 1 >= searchResults.totalPages) {
      return; // No more pages
    }

    try {
      setLoading(true);
      setError(null);

      const nextPageRequest: SearchRequest = {
        ...currentSearchRequest,
        page: searchResults.currentPage + 1
      };

      const results = await searchService.advancedSearch(nextPageRequest);

      // Append new results to existing ones
      setSearchResults(prev => {
        if (!prev) return results;

        return {
          ...results,
          posts: [...prev.posts, ...results.posts]
        };
      });

      setCurrentSearchRequest(nextPageRequest);
    } catch (err: any) {
      console.error('Load more error:', err);
      setError(err.message || 'Có lỗi xảy ra khi tải thêm kết quả');
    } finally {
      setLoading(false);
    }
  }, [currentSearchRequest, searchResults, loading]);

  // Computed values
  const hasMore = searchResults ?
    (searchResults.currentPage + 1) < searchResults.totalPages :
    false;

  const totalResults = searchResults?.totalElements || 0;

  return {
    // State
    searchResults,
    suggestions,
    loading,
    error,

    // Actions
    search,
    advancedSearch,
    getSuggestions,
    loadMore,
    clearResults,

    // Utilities
    hasMore,
    totalResults
  };
};

export default useSearch;
