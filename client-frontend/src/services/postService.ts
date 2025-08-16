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
      let imageUrls: string[] = [];
      let videoUrls: string[] = [];

      // Step 1: Upload files to media service first if files exist
      if (files && files.length > 0) {
        console.log('Uploading files to media service...');
        const mediaResponses: MediaResponse[] = await mediaService.uploadFiles(
          files,
          'Post media files'
        );

        // Separate images and videos based on media type
        mediaResponses.forEach(media => {
          if (media.mediaType === 'IMAGE') {
            imageUrls.push(media.cloudinaryUrl);
          } else if (media.mediaType === 'VIDEO') {
            videoUrls.push(media.cloudinaryUrl);
          }
        });

        console.log('Files uploaded successfully - Images:', imageUrls, 'Videos:', videoUrls);
      }

      // Step 2: Create post data with separated media URLs
      const postRequestData = {
        ...postData,
        images: imageUrls.length > 0 ? imageUrls : undefined,
        videos: videoUrls.length > 0 ? videoUrls : undefined
      };
      console.log('Creating post request...', postRequestData);

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
    const response = await api.get(`/posts/user/${authorId}`, {
      params: {
        page,
        size
      }
    });
    console.log("Fetched user posts:", response.data);
    return response.data;
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
    console.log("Created interaction:", response.data);
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

  // Get top viewed posts
  async getTopViewedPosts(): Promise<Post[]> {
    const topViewedResponse = await api.get('/posts/top-viewed');
    return topViewedResponse.data;
  },

  // Get top liked posts
  async getTopLikedPosts(): Promise<Post[]> {
    const response = await api.get('/posts/top-liked');
    return response.data;
  },

  // INTERACTION STATUS METHODS (New - for persistent state)

  // Check interaction status for a post
  async getInteractionStatus(postId: string): Promise<{
    postId: string;
    userId: string;
    hasLiked: boolean;
    hasBookmarked: boolean;
    interactions: { LIKE: boolean; BOOKMARK: boolean };
  }> {
    try {
      const response = await api.get(`/posts/${postId}/interactions/status`);
      const data = response.data;

      // Add debugging to understand the actual response structure
      console.log('Backend interaction status response:', data);

      // Handle the actual backend response structure
      // Backend returns: { id, postId, userId, type, reactionType, hasInteraction, active, ... }

      let hasLiked = false;
      let hasBookmarked = false;

      // If data is an array of interactions
      if (Array.isArray(data)) {
        console.log('Processing array of interactions:', data.length);
        hasLiked = data.some(interaction => {
          const isLike = interaction.type === 'LIKE' && interaction.hasInteraction && interaction.active;
          console.log('Checking interaction for LIKE:', interaction, 'Result:', isLike);
          return isLike;
        });
        hasBookmarked = data.some(interaction => {
          const isBookmark = interaction.type === 'BOOKMARK' && interaction.hasInteraction && interaction.active;
          console.log('Checking interaction for BOOKMARK:', interaction, 'Result:', isBookmark);
          return isBookmark;
        });
      }
      // If data is a single interaction object
      else if (data && typeof data === 'object') {
        console.log('Processing single interaction object:', data);
        // Check if it's the structure you showed: { hasInteraction: true, type: "LIKE", ... }
        if (data.hasInteraction && data.active) {
          hasLiked = data.type === 'LIKE';
          hasBookmarked = data.type === 'BOOKMARK';
          console.log('Single interaction - hasLiked:', hasLiked, 'hasBookmarked:', hasBookmarked);
        }
        // Or if it's already in the expected format
        else if ('hasLiked' in data) {
          hasLiked = data.hasLiked;
          hasBookmarked = data.hasBookmarked;
          console.log('Expected format - hasLiked:', hasLiked, 'hasBookmarked:', hasBookmarked);
        }
      }

      console.log('Final interaction status:', { hasLiked, hasBookmarked });

      return {
        postId,
        userId: data?.userId || '',
        hasLiked,
        hasBookmarked,
        interactions: {
          LIKE: hasLiked,
          BOOKMARK: hasBookmarked
        }
      };
    } catch (error) {
      console.error('Error fetching interaction status:', error);
      // Return default values on error
      return {
        postId,
        userId: '',
        hasLiked: false,
        hasBookmarked: false,
        interactions: {
          LIKE: false,
          BOOKMARK: false
        }
      };
    }
  },

  // Check like status specifically
  async checkLikeStatus(postId: string): Promise<{
    postId: string;
    hasLiked: boolean;
    userId: string;
  }> {
    const response = await api.get(`/posts/${postId}/interactions/like/status`);
    return response.data;
  },

  // Check bookmark status specifically
  async checkBookmarkStatus(postId: string): Promise<{
    postId: string;
    hasBookmarked: boolean;
    userId: string;
  }> {
    const response = await api.get(`/posts/${postId}/interactions/bookmark/status`);
    return response.data;
  },

  // COMMENT METHODS (Missing)

  // Get comments for a post
  async getComments(postId: string, page = 0, size = 10): Promise<PaginatedResponse<Comment>> {
    const params = new URLSearchParams({
      page: page.toString(),
      size: size.toString(),
    });

    const response = await api.get(`/posts/${postId}/comments?${params.toString()}`);
    console.log("Fetched comments for post:", postId, response.data);
    return response.data;
  },

  // Create a comment
  async createComment(postId: string, commentData: CreateCommentRequest): Promise<Comment> {
    const response = await api.post(`/posts/${postId}/comments`, commentData);
    return response.data;
  },

  async getMyPosts(page = 0, size = 10): Promise<PaginatedResponse<Post>> {
    const response = await api.get(`/posts/me?page=${page}&size=${size}`);

    return response.data;
  },

};
