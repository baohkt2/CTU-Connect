// Search Components
export { default as SearchBar } from './SearchBar';
export { default as SearchResults } from './SearchResults';
export { default as RelatedPosts } from './RelatedPosts';
export { default as TrendingSearches } from './TrendingSearches';
export { default as HeaderSearch } from './HeaderSearch';

// Re-export types
export type {
  SearchRequest,
  SearchResponse,
  SearchSuggestionResponse,
  PostResponse
} from '@/services/searchService';
