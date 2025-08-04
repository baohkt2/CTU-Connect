'use client';

import React, { useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { useUserHooks } from '@/hooks/useUserHooks';
import Layout from '@/components/layout/Layout';
import Avatar from '@/components/ui/Avatar';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import Card from '@/components/ui/Card';
import { debounce } from '@/utils/helpers';
import { MagnifyingGlassIcon, UserPlusIcon } from '@heroicons/react/24/outline';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

export default function SearchPage() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const { useSearchUsers, useFollowUser } = useUserHooks();
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');

  const { data: usersData, isLoading: searchLoading } = useSearchUsers(debouncedQuery);
  const followUserMutation = useFollowUser();

  useEffect(() => {
    if (!loading && !user) {
      router.push('/login');
    }
  }, [user, loading, router]);

  // Debounce search query
  useEffect(() => {
    const debouncedSearch = debounce((query: string) => {
      setDebouncedQuery(query);
    }, 300);

    debouncedSearch(searchQuery);
  }, [searchQuery]);

  const handleFollow = async (userId: string) => {
    try {
      await followUserMutation.mutateAsync(userId);
    } catch (error) {
      console.error('Error following user:', error);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <Layout>
      <div className="max-w-2xl mx-auto space-y-6">
        {/* Search Header */}
        <Card>
          <div className="flex items-center space-x-3">
            <MagnifyingGlassIcon className="w-5 h-5 text-gray-400" />
            <Input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Tìm kiếm người dùng..."
              className="border-none focus:ring-0 p-0"
            />
          </div>
        </Card>

        {/* Search Results */}
        {searchQuery && (
          <div>
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Kết quả tìm kiếm cho &quot;{searchQuery}&quot;
            </h2>

            {searchLoading ? (
              <div className="space-y-4">
                {[...Array(3)].map((_, i) => (
                  <Card key={i} className="animate-pulse">
                    <div className="flex items-center space-x-3">
                      <div className="w-12 h-12 bg-gray-300 rounded-full"></div>
                      <div className="flex-1 space-y-2">
                        <div className="h-4 bg-gray-300 rounded w-1/3"></div>
                        <div className="h-3 bg-gray-300 rounded w-1/4"></div>
                      </div>
                      <div className="w-20 h-8 bg-gray-300 rounded"></div>
                    </div>
                  </Card>
                ))}
              </div>
            ) : usersData?.content && usersData.content.length > 0 ? (
              <div className="space-y-4">
                {usersData.content.map((searchUser) => (
                  <Card key={searchUser.id} hover>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <Avatar
                          src={searchUser.avatarUrl || '/default-avatar.png'}
                          alt={searchUser.fullName}
                          size="md"
                          online={searchUser.isOnline}
                        />
                        <div>
                          <Link
                            href={`/profile/${searchUser.id}`}
                            className="font-semibold text-gray-900 hover:text-blue-600"
                          >
                            {searchUser.fullName}
                          </Link>
                          <p className="text-sm text-gray-500">
                            @{searchUser.username}
                          </p>
                          {searchUser.faculty && (
                            <p className="text-xs text-gray-400">
                              {searchUser.faculty.name}
                            </p>
                          )}
                        </div>
                      </div>

                      {searchUser.id !== user.id && (
                        <Button
                          onClick={() => handleFollow(searchUser.id)}
                          loading={followUserMutation.isPending}
                          size="sm"
                          className="flex items-center space-x-2"
                        >
                          <UserPlusIcon className="w-4 h-4" />
                          <span>Theo dõi</span>
                        </Button>
                      )}
                    </div>
                  </Card>
                ))}
              </div>
            ) : (
              <Card className="text-center py-8">
                <div className="text-gray-500">
                  <MagnifyingGlassIcon className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                  <p>Không tìm thấy người dùng nào</p>
                  <p className="text-sm mt-1">
                    Hãy thử tìm kiếm với từ khóa khác
                  </p>
                </div>
              </Card>
            )}
          </div>
        )}

        {/* Popular Users or Recent Searches */}
        {!searchQuery && (
          <div>
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Gợi ý kết bạn
            </h2>
            <Card className="text-center py-8">
              <div className="text-gray-500">
                <MagnifyingGlassIcon className="w-12 h-12 mx-auto mb-4 text-gray-300" />
                <p>Tìm kiếm bạn bè và kết nối</p>
                <p className="text-sm mt-1">
                  Nhập tên hoặc username để tìm kiếm người dùng
                </p>
              </div>
            </Card>
          </div>
        )}
      </div>
    </Layout>
  );
}
