import { apiClient } from '@/shared/config/api-client';
import { API_ENDPOINTS } from '@/shared/constants';
import { createApiUrl } from '@/shared/utils';
import {
  Post,
  Comment,
  CreatePostRequest,
  UpdatePostRequest,
  CreateCommentRequest,
  UpdateCommentRequest,
  PaginatedResponse,
  ApiResponse,
  PostVisibility,
} from '@/shared/types';

/**
 * Post Service - Updated to sync with backend APIs
 */
export class PostService {
  /**
   * Create new post - Updated to match EnhancedPostController
   */
  async createPost(postData: CreatePostRequest): Promise<Post> {
    // Use JSON instead of FormData to match EnhancedPostController
    return apiClient.post<Post>(API_ENDPOINTS.POSTS.BASE, postData);
  }

  /**
   * Get personalized feed - Updated to match backend
   */
  async getPersonalizedFeed(page = 0, size = 10): Promise<Post[]> {
    const url = createApiUrl(API_ENDPOINTS.POSTS.FEED, undefined, { page, size });
    return apiClient.get<Post[]>(url);
  }

  /**
   * Get posts feed
   */
  async getPosts(page = 0, size = 10): Promise<PaginatedResponse<Post>> {
    const url = createApiUrl(API_ENDPOINTS.POSTS.BASE, undefined, { page, size });
    return apiClient.get<PaginatedResponse<Post>>(url);
  }

  /**
   * Get single post by ID
   */
  async getPost(postId: string): Promise<Post> {
    const url = createApiUrl(API_ENDPOINTS.POSTS.BASE + '/:id', { id: postId });
    return apiClient.get<Post>(url);
  }

  /**
   * Get posts by user
   */
  async getUserPosts(
    userId: string,
    page = 0,
    size = 10
  ): Promise<PaginatedResponse<Post>> {
    const url = createApiUrl(
      API_ENDPOINTS.POSTS.BY_USER,
      { id: userId },
      { page, size }
    );
    return apiClient.get<PaginatedResponse<Post>>(url);
  }

  /**
   * Update post
   */
  async updatePost(postId: string, updateData: UpdatePostRequest): Promise<Post> {
    const url = createApiUrl(API_ENDPOINTS.POSTS.BASE + '/:id', { id: postId });
    return apiClient.put<Post>(url, updateData);
  }

  /**
   * Delete post
   */
  async deletePost(postId: string): Promise<ApiResponse<null>> {
    const url = createApiUrl(API_ENDPOINTS.POSTS.BASE + '/:id', { id: postId });
    return apiClient.delete<ApiResponse<null>>(url);
  }

  /**
   * Like post
   */
  async likePost(postId: string): Promise<ApiResponse<null>> {
    const url = createApiUrl(API_ENDPOINTS.POSTS.LIKE, { id: postId });
    return apiClient.post<ApiResponse<null>>(url);
  }

  /**
   * Unlike post
   */
  async unlikePost(postId: string): Promise<ApiResponse<null>> {
    const url = createApiUrl(API_ENDPOINTS.POSTS.LIKE, { id: postId });
    return apiClient.delete<ApiResponse<null>>(url);
  }

  /**
   * Get post comments
   */
  async getComments(
    postId: string,
    page = 0,
    size = 10
  ): Promise<PaginatedResponse<Comment>> {
    const url = createApiUrl(
      API_ENDPOINTS.POSTS.COMMENTS,
      { id: postId },
      { page, size }
    );
    return apiClient.get<PaginatedResponse<Comment>>(url);
  }

  /**
   * Create comment
   */
  async createComment(commentData: CreateCommentRequest): Promise<Comment> {
    const url = createApiUrl(API_ENDPOINTS.POSTS.COMMENTS, { id: commentData.postId });
    return apiClient.post<Comment>(url, {
      content: commentData.content,
      parentId: commentData.parentId,
    });
  }

  /**
   * Update comment
   */
  async updateComment(
    commentId: string,
    updateData: UpdateCommentRequest
  ): Promise<Comment> {
    const url = createApiUrl('/comments/:id', { id: commentId });
    return apiClient.put<Comment>(url, updateData);
  }

  /**
   * Delete comment
   */
  async deleteComment(commentId: string): Promise<ApiResponse<null>> {
    const url = createApiUrl('/comments/:id', { id: commentId });
    return apiClient.delete<ApiResponse<null>>(url);
  }

  /**
   * Like comment
   */
  async likeComment(commentId: string): Promise<ApiResponse<null>> {
    const url = createApiUrl(API_ENDPOINTS.POSTS.COMMENT_LIKE, { id: commentId });
    return apiClient.post<ApiResponse<null>>(url);
  }

  /**
   * Unlike comment
   */
  async unlikeComment(commentId: string): Promise<ApiResponse<null>> {
    const url = createApiUrl(API_ENDPOINTS.POSTS.COMMENT_LIKE, { id: commentId });
    return apiClient.delete<ApiResponse<null>>(url);
  }

  /**
   * Share post
   */
  async sharePost(postId: string, content?: string): Promise<ApiResponse<null>> {
    const url = createApiUrl('/posts/:id/share', { id: postId });
    return apiClient.post<ApiResponse<null>>(url, { content });
  }

  /**
   * Bookmark post
   */
  async bookmarkPost(postId: string): Promise<ApiResponse<null>> {
    const url = createApiUrl('/posts/:id/bookmark', { id: postId });
    return apiClient.post<ApiResponse<null>>(url);
  }

  /**
   * Remove bookmark
   */
  async removeBookmark(postId: string): Promise<ApiResponse<null>> {
    const url = createApiUrl('/posts/:id/bookmark', { id: postId });
    return apiClient.delete<ApiResponse<null>>(url);
  }
}

// Export singleton instance
export const postService = new PostService();
