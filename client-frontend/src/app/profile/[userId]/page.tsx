'use client';

import React from 'react';
import { useParams } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import Layout from '@/components/layout/Layout';
import UserProfile from '@/components/user/UserProfile';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

export default function OtherUserProfilePage() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const params = useParams();
  const userId = params.userId as string;

  useEffect(() => {
    if (!loading && !user) {
      router.push('/login');
      return;
    }

    // If the userId matches current user's ID, redirect to /profile/me
    if (user && user.id === userId) {
      router.push('/profile/me');
      return;
    }
  }, [user, loading, router, userId]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600 vietnamese-text">Đang tải trang cá nhân...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  // Don't render if this is current user's profile (will be redirected)
  if (user.id === userId) {
    return null;
  }

  return (
    <Layout>
      <div className="min-h-screen bg-gray-50">
        <UserProfile userId={userId} />
      </div>
    </Layout>
  );
}
