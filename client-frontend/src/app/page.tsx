'use client';

import React, { useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import Layout from '@/components/layout/Layout';
import CreatePostForm from '@/components/post/CreatePostForm';
import {PostCard }from '@/components/post/PostCard';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Link from 'next/link';
import { usePostHooks } from '@/hooks/usePostHooks';
import { useRouter } from 'next/navigation';

// Landing Page Component for non-authenticated users
const LandingPage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-cyan-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <header className="pt-6 pb-16">
          <nav className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <img src="/logo.svg" alt="CTU Connect" className="h-8 w-auto" />
              <span className="text-xl font-bold text-gray-900">CTU Connect</span>
            </div>
            <div className="flex items-center space-x-4">
              <Link href="/login">
                <Button variant="outline">Đăng nhập</Button>
              </Link>
              <Link href="/register">
                <Button>Đăng ký</Button>
              </Link>
            </div>
          </nav>
        </header>

        {/* Hero Section */}
        <main className="pt-10 pb-20">
          <div className="text-center">
            <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
              Kết nối sinh viên{' '}
              <span className="text-indigo-600">Đại học Cần Thơ</span>
            </h1>
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
              Tham gia cộng đồng sinh viên CTU để chia sẻ kiến thức, kết nối bạn bè và cùng nhau phát triển
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/register">
                <Button size="lg" className="w-full sm:w-auto">
                  Tham gia ngay
                </Button>
              </Link>
              <Link href="/login">
                <Button variant="outline" size="lg" className="w-full sm:w-auto">
                  Đã có tài khoản?
                </Button>
              </Link>
            </div>
          </div>

          {/* Features */}
          <div className="mt-20 grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="bg-indigo-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8h2a2 2 0 012 2v6a2 2 0 01-2 2h-2v4l-4-4H9a2 2 0 01-2-2v-6a2 2 0 012-2h2" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Trò chuyện</h3>
              <p className="text-gray-600">Kết nối và trò chuyện với bạn bè, đồng khoa trong trường</p>
            </div>
            <div className="text-center">
              <div className="bg-indigo-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Chia sẻ</h3>
              <p className="text-gray-600">Chia sẻ thông tin học tập, kinh nghiệm và hoạt động sinh viên</p>
            </div>
            <div className="text-center">
              <div className="bg-indigo-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">Khám phá</h3>
              <p className="text-gray-600">Tìm kiếm và khám phá các hoạt động, sự kiện trong trường</p>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
};

// Home Page Component for authenticated users
const HomePage = () => {
  const { usePosts } = usePostHooks();
  const { data: postsData, isLoading: postsLoading } = usePosts(0, 10);

  return (
    <Layout>
      <div className="max-w-2xl mx-auto space-y-6">
        {/* Create Post Form */}
        <CreatePostForm />

        {/* Posts Feed */}
        <div>
          {postsLoading ? (
            <div className="space-y-4">
              {[...Array(3)].map((_, i) => (
                <Card key={i} className="animate-pulse">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="w-10 h-10 bg-gray-300 rounded-full"></div>
                    <div className="flex-1">
                      <div className="h-4 bg-gray-300 rounded w-1/4 mb-2"></div>
                      <div className="h-3 bg-gray-300 rounded w-1/6"></div>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="h-4 bg-gray-300 rounded w-full"></div>
                    <div className="h-4 bg-gray-300 rounded w-3/4"></div>
                  </div>
                </Card>
              ))}
            </div>
          ) : postsData?.content && postsData.content.length > 0 ? (
            <div className="space-y-4">
              {postsData.content.map((post) => (
                <PostCard key={post.id} post={post} />
              ))}
            </div>
          ) : (
            <Card className="text-center py-8">
              <p className="text-gray-600 mb-4">Chưa có bài viết nào.</p>
              <p className="text-sm text-gray-500">Hãy tạo bài viết đầu tiên của bạn!</p>
            </Card>
          )}
        </div>
      </div>
    </Layout>
  );
};

export default function Home() {
  const { user, loading } = useAuth();
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  return user ? <HomePage /> : <LandingPage />;
}
