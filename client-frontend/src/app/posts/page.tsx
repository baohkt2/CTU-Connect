'use client';

import React, { useState, useEffect } from 'react';
import { Post, PaginatedResponse } from '@/types';
import { postService } from '@/services/postService';
import { CreatePost } from '@/components/post/CreatePost';
import { PostCard } from '@/components/post/PostCard';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorAlert } from '@/components/ui/ErrorAlert';
import { Button } from '@/components/ui/Button';
import { Plus, RefreshCw } from 'lucide-react';

export default function PostsPage() {
  const [posts, setPosts] = useState<Post[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreatePost, setShowCreatePost] = useState(false);

  const loadPosts = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response: PaginatedResponse<Post> = await postService.getPosts();
      setPosts(response.content);
    } catch (err: any) {
      console.error('Failed to load posts:', err);
      setError(err.response?.data?.message || err.message || 'Failed to load posts');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadPosts();
  }, []);

  const handlePostCreated = (newPost: Post) => {
    setPosts(prev => [newPost, ...prev]);
    setShowCreatePost(false);
  };

  const handlePostUpdate = (updatedPost: Post) => {
    setPosts(prev => prev.map(post => 
      post.id === updatedPost.id ? updatedPost : post
    ));
  };

  const handlePostDelete = (postId: string) => {
    setPosts(prev => prev.filter(post => post.id !== postId));
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-2xl mx-auto py-8 px-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold text-gray-900">Posts</h1>
          <div className="flex gap-2">
            <Button
              variant="secondary"
              onClick={loadPosts}
              disabled={isLoading}
              className="flex items-center gap-2"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
            <Button
              onClick={() => setShowCreatePost(true)}
              className="flex items-center gap-2"
            >
              <Plus className="w-4 h-4" />
              Create Post
            </Button>
          </div>
        </div>

        {/* Create Post Form */}
        {showCreatePost && (
          <div className="mb-6">
            <CreatePost
              onPostCreated={handlePostCreated}
              onCancel={() => setShowCreatePost(false)}
            />
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="mb-6">
            <ErrorAlert message={error} />
          </div>
        )}

        {/* Posts List */}
        {posts.length === 0 ? (
          <div className="text-center py-12">
            <div className="text-gray-400 mb-4">
              <Plus className="w-16 h-16 mx-auto" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">No posts yet</h3>
            <p className="text-gray-500 mb-4">Be the first to create a post!</p>
            <Button onClick={() => setShowCreatePost(true)}>
              Create First Post
            </Button>
          </div>
        ) : (
          <div className="space-y-6">
            {posts.map((post) => (
              <PostCard
                key={post.id}
                post={post}
                onPostUpdate={handlePostUpdate}
                onPostDelete={handlePostDelete}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
