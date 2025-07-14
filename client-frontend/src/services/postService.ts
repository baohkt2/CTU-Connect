import api from '@/lib/api';
import { Post, Comment, ApiResponse, PaginatedResponse } from '@/types';

export const postService = {
  async createPost(content: string, images?: File[]): Promise<Post> {
    const formData = new FormData();
    formData.append('content', content);

    if (images) {
      images.forEach((image) => {
        formData.append('images', image);
      });
    }

    const response = await api.post('/posts', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  async getPosts(page = 0, size = 10): Promise<PaginatedResponse<Post>> {
    const response = await api.get(`/posts?page=${page}&size=${size}`);
    return response.data;
  },

  async getPost(postId: string): Promise<Post> {
    const response = await api.get(`/posts/${postId}`);
    return response.data;
  },

  async getUserPosts(userId: string, page = 0, size = 10): Promise<PaginatedResponse<Post>> {
    const response = await api.get(`/posts/user/${userId}?page=${page}&size=${size}`);
    return response.data;
  },

  async likePost(postId: string): Promise<ApiResponse<null>> {
    const response = await api.post(`/posts/${postId}/like`);
    return response.data;
  },

  async unlikePost(postId: string): Promise<ApiResponse<null>> {
    const response = await api.delete(`/posts/${postId}/like`);
    return response.data;
  },

  async deletePost(postId: string): Promise<ApiResponse<null>> {
    const response = await api.delete(`/posts/${postId}`);
    return response.data;
  },

  async createComment(postId: string, content: string): Promise<Comment> {
    const response = await api.post(`/posts/${postId}/comments`, { content });
    return response.data;
  },

  async getComments(postId: string, page = 0, size = 10): Promise<PaginatedResponse<Comment>> {
    const response = await api.get(`/posts/${postId}/comments?page=${page}&size=${size}`);
    return response.data;
  },

  async likeComment(commentId: string): Promise<ApiResponse<null>> {
    const response = await api.post(`/comments/${commentId}/like`);
    return response.data;
  },

  async unlikeComment(commentId: string): Promise<ApiResponse<null>> {
    const response = await api.delete(`/comments/${commentId}/like`);
    return response.data;
  },

  async deleteComment(commentId: string): Promise<ApiResponse<null>> {
    const response = await api.delete(`/comments/${commentId}`);
    return response.data;
  }
};
