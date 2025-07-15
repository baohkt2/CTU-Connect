'use client';

import React from 'react';
import { useAuth } from '@/contexts/AuthContext';
import Layout from '@/components/layout/Layout';
import Card from '@/components/ui/Card';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

export default function MessagesPage() {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user) {
      router.push('/login');
    }
  }, [user, loading, router]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
      </div>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <Layout>
      <div className="max-w-6xl mx-auto">
        <Card className="min-h-[600px]">
          <div className="text-center py-16">
            <h1 className="text-2xl font-bold text-gray-900 mb-4">Tin nhắn</h1>
            <p className="text-gray-600 mb-8">Tính năng chat sẽ sớm được phát triển</p>
            <div className="bg-gray-50 rounded-lg p-8">
              <p className="text-sm text-gray-500">
                Chức năng nhắn tin đang được phát triển và sẽ có sẵn trong phiên bản tiếp theo.
              </p>
            </div>
          </div>
        </Card>
      </div>
    </Layout>
  );
}
