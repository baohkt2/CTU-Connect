import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { postService } from '@/services/postService';
import { Post, PaginatedResponse } from '@/types';

export const usePostHooks = () => {
  const queryClient = useQueryClient();

  // Get posts with pagination
  const usePosts = (page = 0, size = 10) => {
    return useQuery({
      queryKey: ['posts', page, size],
      queryFn: () => postService.getPosts(page, size),
      staleTime: 5 * 60 * 1000, // 5 minutes
    });
  };

  // Get user posts
  const useUserPosts = (userId: string, page = 0, size = 10) => {
    return useQuery({
      queryKey: ['posts', 'user', userId, page, size],
      queryFn: () => postService.getUserPosts(userId, page, size),
      enabled: !!userId,
    });
  };

  // Get single post
  const usePost = (postId: string) => {
    return useQuery({
      queryKey: ['post', postId],
      queryFn: () => postService.getPost(postId),
      enabled: !!postId,
    });
  };

  // Create post mutation
  const useCreatePost = () => {
    return useMutation({
      mutationFn: ({ content, images }: { content: string; images?: File[] }) =>
        postService.createPost(content, images),
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['posts'] });
      },
    });
  };

  // Like post mutation
  const useLikePost = () => {
    return useMutation({
      mutationFn: (postId: string) => postService.likePost(postId),
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['posts'] });
      },
    });
  };

  // Unlike post mutation
  const useUnlikePost = () => {
    return useMutation({
      mutationFn: (postId: string) => postService.unlikePost(postId),
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['posts'] });
      },
    });
  };

  // Delete post mutation
  const useDeletePost = () => {
    return useMutation({
      mutationFn: (postId: string) => postService.deletePost(postId),
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['posts'] });
      },
    });
  };

  return {
    usePosts,
    useUserPosts,
    usePost,
    useCreatePost,
    useLikePost,
    useUnlikePost,
    useDeletePost,
  };
};

export const useCommentHooks = () => {
  const queryClient = useQueryClient();

  // Get comments for a post
  const useComments = (postId: string, page = 0, size = 10) => {
    return useQuery({
      queryKey: ['comments', postId, page, size],
      queryFn: () => postService.getComments(postId, page, size),
      enabled: !!postId,
    });
  };

  // Create comment mutation
  const useCreateComment = () => {
    return useMutation({
      mutationFn: ({ postId, content }: { postId: string; content: string }) =>
        postService.createComment(postId, content),
      onSuccess: (_, { postId }) => {
        queryClient.invalidateQueries({ queryKey: ['comments', postId] });
        queryClient.invalidateQueries({ queryKey: ['posts'] });
      },
    });
  };

  // Like comment mutation
  const useLikeComment = () => {
    return useMutation({
      mutationFn: (commentId: string) => postService.likeComment(commentId),
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['comments'] });
      },
    });
  };

  // Unlike comment mutation
  const useUnlikeComment = () => {
    return useMutation({
      mutationFn: (commentId: string) => postService.unlikeComment(commentId),
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['comments'] });
      },
    });
  };

  return {
    useComments,
    useCreateComment,
    useLikeComment,
    useUnlikeComment,
  };
};
