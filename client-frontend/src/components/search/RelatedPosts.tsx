'use client';

import React, { useState, useEffect } from 'react';
import { ExternalLink, Clock, Heart, MessageCircle } from 'lucide-react';
import { searchService, PostResponse } from '@/services/searchService';
import Link from 'next/link';

interface RelatedPostsProps {
  postId: string;
  limit?: number;
  className?: string;
}

const RelatedPosts: React.FC<RelatedPostsProps> = ({
  postId,
  limit = 5,
  className = ''
}) => {
  const [relatedPosts, setRelatedPosts] = useState<PostResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchRelatedPosts = async () => {
      if (!postId) return;

      try {
        setLoading(true);
        setError(null);
        const response = await searchService.getRelatedPosts(postId, limit);
        setRelatedPosts(response.relatedPosts || []);
      } catch (err) {
        console.error('Error fetching related posts:', err);
        setError('Không thể tải bài viết liên quan');
      } finally {
        setLoading(false);
      }
    };

    fetchRelatedPosts();
  }, [postId, limit]);

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const hours = Math.floor(diff / (1000 * 60 * 60));

    if (hours < 24) {
      return `${hours} giờ trước`;
    } else {
      const days = Math.floor(hours / 24);
      return `${days} ngày trước`;
    }
  };

  if (loading) {
    return (
      <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-4 ${className}`}>
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Bài viết liên quan</h3>
        <div className="space-y-3">
          {[...Array(3)].map((_, index) => (
            <div key={index} className="animate-pulse">
              <div className="flex space-x-3">
                <div className="w-16 h-12 bg-gray-300 rounded"></div>
                <div className="flex-1">
                  <div className="h-4 bg-gray-300 rounded w-3/4 mb-2"></div>
                  <div className="h-3 bg-gray-300 rounded w-1/2"></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error || !relatedPosts.length) {
    return null;
  }

  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-4 ${className}`}>
      <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
        <ExternalLink className="w-5 h-5 mr-2 text-blue-600" />
        Bài viết liên quan
      </h3>

      <div className="space-y-4">
        {relatedPosts.map((post) => (
          <Link
            key={post.id}
            href={`/posts/${post.id}`}
            className="block group hover:bg-gray-50 rounded-lg p-2 -m-2 transition-colors"
          >
            <div className="flex space-x-3">
              {/* Post Thumbnail */}
              <div className="flex-shrink-0">
                {post.images && post.images.length > 0 ? (
                  <img
                    src={post.images[0]}
                    alt={post.title}
                    className="w-16 h-12 object-cover rounded"
                  />
                ) : (
                  <div className="w-16 h-12 bg-gray-100 rounded flex items-center justify-center">
                    <span className="text-xs text-gray-500">No image</span>
                  </div>
                )}
              </div>

              {/* Post Info */}
              <div className="flex-1 min-w-0">
                <h4 className="text-sm font-medium text-gray-900 group-hover:text-blue-600 transition-colors line-clamp-2">
                  {post.title || post.content.substring(0, 100) + '...'}
                </h4>

                <div className="flex items-center space-x-2 mt-1">
                  <span className="text-xs text-gray-500">{post.author.name}</span>
                  <span className="text-xs text-gray-400">•</span>
                  <div className="flex items-center space-x-1 text-xs text-gray-500">
                    <Clock className="w-3 h-3" />
                    <span>{formatDate(post.createdAt)}</span>
                  </div>
                </div>

                <div className="flex items-center space-x-3 mt-1 text-xs text-gray-500">
                  <div className="flex items-center space-x-1">
                    <Heart className="w-3 h-3" />
                    <span>{post.stats.likes}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <MessageCircle className="w-3 h-3" />
                    <span>{post.stats.comments}</span>
                  </div>
                </div>

                {/* Tags */}
                {post.tags && post.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-2">
                    {post.tags.slice(0, 2).map((tag, index) => (
                      <span
                        key={index}
                        className="inline-flex items-center px-1.5 py-0.5 text-xs bg-blue-50 text-blue-600 rounded"
                      >
                        #{tag}
                      </span>
                    ))}
                    {post.tags.length > 2 && (
                      <span className="text-xs text-gray-500">+{post.tags.length - 2}</span>
                    )}
                  </div>
                )}
              </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
};

export default RelatedPosts;
