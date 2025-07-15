import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { userService } from '@/services/userService';
import { User } from '@/types';

export const useUserHooks = () => {
  const queryClient = useQueryClient();

  // Get user profile
  const useUser = (userId: string) => {
    return useQuery({
      queryKey: ['user', userId],
      queryFn: () => userService.getProfile(userId),
      enabled: !!userId,
    });
  };

  // Search users
  const useSearchUsers = (query: string, page = 0, size = 10) => {
    return useQuery({
      queryKey: ['users', 'search', query, page, size],
      queryFn: () => userService.searchUsers(query, page, size),
      enabled: !!query,
    });
  };

  // Get followers
  const useFollowers = (userId: string, page = 0, size = 10) => {
    return useQuery({
      queryKey: ['users', userId, 'followers', page, size],
      queryFn: () => userService.getFollowers(userId, page, size),
      enabled: !!userId,
    });
  };

  // Get following
  const useFollowing = (userId: string, page = 0, size = 10) => {
    return useQuery({
      queryKey: ['users', userId, 'following', page, size],
      queryFn: () => userService.getFollowing(userId, page, size),
      enabled: !!userId,
    });
  };

  // Update profile mutation
  const useUpdateProfile = () => {
    return useMutation({
      mutationFn: (userData: Partial<User>) => userService.updateProfile(userData),
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['user'] });
      },
    });
  };

  // Follow user mutation
  const useFollowUser = () => {
    return useMutation({
      mutationFn: (userId: string) => userService.followUser(userId),
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['users'] });
      },
    });
  };

  // Unfollow user mutation
  const useUnfollowUser = () => {
    return useMutation({
      mutationFn: (userId: string) => userService.unfollowUser(userId),
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['users'] });
      },
    });
  };

  // Upload avatar mutation
  const useUploadAvatar = () => {
    return useMutation({
      mutationFn: (file: File) => userService.uploadAvatar(file),
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ['user'] });
      },
    });
  };

  return {
    useUser,
    useSearchUsers,
    useFollowers,
    useFollowing,
    useUpdateProfile,
    useFollowUser,
    useUnfollowUser,
    useUploadAvatar,
  };
};
