'use client';

import React, { useEffect } from 'react';
import LoginForm from '@/components/auth/LoginForm';
import { useAuth } from '@/contexts/AuthContext';
import { useRouter } from 'next/navigation';

export default function LoginPage() {
  const {  loading, isAuthenticated } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!loading && isAuthenticated) {
      router.replace('/'); // ✅ Chuyển hướng nếu đã đăng nhập
    }
    console.log('isAuthenticated:', isAuthenticated);
  }, [isAuthenticated, loading, router]); // ✅ Sửa dependency từ `user` → `isAuthenticated`

  if (loading) {
    return (
        <div className="min-h-screen flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
    );
  }

  if (isAuthenticated) {
    return null; // ✅ Tránh render LoginForm nếu đã đăng nhập
  }

  return <LoginForm />;
}
