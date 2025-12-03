import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8080';

export interface SearchRequest {
  query?: string;
  category?: string;
  authorId?: string;
  tags?: string[];
  dateFrom?: string;
  dateTo?: string;
  postType?: string;
  minLikes?: number;
  minViews?: number;
  minComments?: number;
  sortBy?: 'relevance' | 'date' | 'likes' | 'views' | 'comments';
  sortDirection?: 'asc' | 'desc';
  page?: number;
  size?: number;
  includeContent?: boolean;
  includeTags?: boolean;
  includeAuthor?: boolean;
  exactMatch?: boolean;
}

export interface SearchResponse {
  posts: PostResponse[];
  totalElements: number;
  totalPages: number;
  currentPage: number;
  pageSize: number;
  searchQuery: string;
  filtersApplied: Record<string, any>;
  searchTimeMs: number;
  suggestedQueries?: string[];
  relatedSearches?: string[];
}

export interface SearchSuggestionResponse {
  titleSuggestions: string[];
  categorySuggestions: string[];
  tagSuggestions: string[];
  authorSuggestions: string[];
  trendingQueries: string[];
  recentSearches: string[];
}

export interface PostResponse {
  id: string;
  title: string;
  content: string;
  author: {
    id: string;
    name: string;
    avatarUrl?: string;
  };
  category: string;
  tags: string[];
  stats: {
    likes: number;
    views: number;
    comments: number;
    shares?: number;
  };
  createdAt: string;
  updatedAt: string;
  images?: string[];
  videos?: string[];
}

class SearchService {
  private apiClient;

  constructor() {
    this.apiClient = axios.create({
      baseURL: API_BASE_URL,
      timeout: 10000,
    });

    // Add auth interceptor
    this.apiClient.interceptors.request.use((config) => {
      const token = localStorage.getItem('token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });
  }

  // Simple search with GET parameters
  async searchPosts(params: {
    query?: string;
    category?: string;
    authorId?: string;
    tags?: string;
    dateFrom?: string;
    dateTo?: string;
    postType?: string;
    minLikes?: number;
    sortBy?: string;
    sortDirection?: string;
    page?: number;
    size?: number;
  }): Promise<SearchResponse> {
    const response = await this.apiClient.get('/api/search/posts', { params });
    return response.data;
  }

  // Advanced search with POST body
  async advancedSearch(searchRequest: SearchRequest): Promise<SearchResponse> {
    const response = await this.apiClient.post('/api/search/posts', searchRequest);
    return response.data;
  }

  // Get search suggestions for autocomplete
  async getSearchSuggestions(query: string): Promise<SearchSuggestionResponse> {
    const response = await this.apiClient.get('/api/search/suggestions', {
      params: { query }
    });
    return response.data;
  }

  // Get trending search terms
  async getTrendingSearchTerms(): Promise<{ trendingTerms: string[] }> {
    const response = await this.apiClient.get('/api/search/trending');
    return response.data;
  }

  // Get related posts for a specific post
  async getRelatedPosts(postId: string, limit: number = 5): Promise<{ relatedPosts: PostResponse[] }> {
    const response = await this.apiClient.get(`/api/search/related/${postId}`, {
      params: { limit }
    });
    return response.data;
  }

  // Quick search for mobile/simplified interfaces
  async quickSearch(query: string, limit: number = 5): Promise<{ posts: PostResponse[]; total: number }> {
    const response = await this.apiClient.get('/api/search/quick', {
      params: { q: query, limit }
    });
    return response.data;
  }

  // Search using post service endpoints (backward compatibility)
  async searchPostsViaPostService(params: {
    query?: string;
    category?: string;
    authorId?: string;
    page?: number;
    size?: number;
    sortBy?: string;
    sortDir?: string;
  }): Promise<{ content: PostResponse[]; totalElements: number; totalPages: number }> {
    const response = await this.apiClient.get('/api/posts', { params });
    return response.data;
  }

  // Get search suggestions via post service
  async getPostSearchSuggestions(query: string): Promise<SearchSuggestionResponse> {
    const response = await this.apiClient.get('/api/posts/search/suggestions', {
      params: { query }
    });
    return response.data;
  }

  // Advanced search via post service
  async advancedSearchViaPostService(searchRequest: SearchRequest): Promise<SearchResponse> {
    const response = await this.apiClient.post('/api/posts/search/advanced', searchRequest);
    return response.data;
  }
}

export const searchService = new SearchService();
export default searchService;
