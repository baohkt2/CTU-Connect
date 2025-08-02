import api from '@/lib/api';
import { mediaService, MediaResponse } from './mediaService';
import {
  Post,
  Comment,
  CreatePostRequest,
  UpdatePostRequest,
  CreateCommentRequest,
  CreateInteractionRequest,
  Interaction,
  PaginatedResponse,
  InteractionType,
  ReactionType
} from '@/types';

export const postService = {
  // Create post with proper structure matching backend
  async createPost(postData: CreatePostRequest, files?: File[]): Promise<Post> {
    try {
      let mediaUrls: string[] = [];

      // Step 1: Upload files to media service first if files exist
      if (files && files.length > 0) {
        console.log('Uploading files to media service...');
        const mediaResponses: MediaResponse[] = await mediaService.uploadFiles(
          files,
          'Post media files'
        );

        // Extract cloudinary URLs from media responses
        mediaUrls = mediaResponses.map(media => media.cloudinaryUrl);
        console.log('Files uploaded successfully:', mediaUrls);
      }

      // Step 2: Create post data with media URLs
      const postRequestData = {
        ...postData,
        images: mediaUrls.length > 0 ? mediaUrls : undefined
      };
      console.log('Uploading post request...', postRequestData);
      // Step 3: Create post via post service
      const response = await api.post('/posts', postRequestData, {
        headers: {
          'Content-Type': 'application/json',
        },
      });

      return response.data;
    } catch (error) {
      console.error('Error creating post:', error);
      throw error;
    }
  },

  // Get paginated posts
  async getPosts(
    page = 0,
    size = 10,
    sortBy = 'createdAt',
    sortDir = 'desc',
    authorId?: string,
    category?: string,
    search?: string
  ): Promise<PaginatedResponse<Post>> {
    const params = new URLSearchParams({
      page: page.toString(),
      size: size.toString(),
      sortBy,
      sortDir,
    });

    if (authorId) params.append('authorId', authorId);
    if (category) params.append('category', category);
    if (search) params.append('search', search);

    const response = await api.get(`/posts?${params.toString()}`);
    return response.data;
  },

  // Get single post by ID
  async getPost(postId: string): Promise<Post> {
    const response = await api.get(`/posts/${postId}`);
    return response.data;
  },

  // Get posts by author
  async getUserPosts(authorId: string, page = 0, size = 10): Promise<PaginatedResponse<Post>> {
    return this.getPosts(page, size, 'createdAt', 'desc', authorId);
  },

  // Update post
  async updatePost(postId: string, updateData: UpdatePostRequest): Promise<Post> {
    const response = await api.put(`/posts/${postId}`, updateData);
    return response.data;
  },

  // Delete post
  async deletePost(postId: string): Promise<void> {
    await api.delete(`/posts/${postId}`);
  },

  // Search posts
  async searchPosts(query: string, page = 0, size = 10): Promise<PaginatedResponse<Post>> {
    return this.getPosts(page, size, 'createdAt', 'desc', undefined, undefined, query);
  },

  // Get posts by category
  async getPostsByCategory(category: string, page = 0, size = 10): Promise<PaginatedResponse<Post>> {
    return this.getPosts(page, size, 'createdAt', 'desc', undefined, category);
  },

  // INTERACTION METHODS

  // Create interaction (like, share, bookmark)
  async createInteraction(postId: string, interactionData: CreateInteractionRequest): Promise<Interaction | null> {
    const response = await api.post(`/posts/${postId}/interactions`, interactionData);
    return response.data;
  },

  // Like/Unlike post
  async toggleLike(postId: string): Promise<Interaction | null> {
    return this.createInteraction(postId, {
      type: InteractionType.LIKE,
      reactionType: ReactionType.LIKE
    });
  },

  // Share post
  async sharePost(postId: string): Promise<Interaction | null> {
    return this.createInteraction(postId, {
      type: InteractionType.SHARE
    });
  },

  // Bookmark post
  async toggleBookmark(postId: string): Promise<Interaction | null> {
    return this.createInteraction(postId, {
      type: InteractionType.BOOKMARK,
      reactionType: ReactionType.BOOKMARK
    });
  },

  // Check if user has liked post
  async hasUserLikedPost(postId: string): Promise<boolean> {
    const response = await api.get(`/posts/${postId}/likes/check`);
    return response.data;
  },

  // COMMENT METHODS

  // Get comments for post
  async getComments(postId: string, page = 0, size = 10): Promise<PaginatedResponse<Comment>> {
    const response = await api.get(`/posts/${postId}/comments?page=${page}&size=${size}`);
    return response.data;
  },

  // Create comment
  async createComment(postId: string, commentData: CreateCommentRequest): Promise<Comment> {
    const response = await api.post(`/posts/${postId}/comments`, commentData);
    return response.data;
  },

  // UTILITY METHODS

  // Get trending posts
  async getTrendingPosts(): Promise<Post[]> {
    const response = await api.get('/posts/trending');
    return response.data;
  },

  // Get top viewed posts
  async getTopViewedPosts(): Promise<Post[]> {
    const response = await api.get('/posts/top-viewed');
    return response.data;
  },

  // Get top liked posts
  async getTopLikedPosts(): Promise<Post[]> {
    const response = await api.get('/posts/top-liked');
    return response.data;
  }
};
