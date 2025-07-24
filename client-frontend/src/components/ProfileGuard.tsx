'use client';

import { useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { userService } from '@/services/userService';
import LoadingSpinner from '@/components/ui/LoadingSpinner';

interface ProfileGuardProps {
  children: React.ReactNode;
}

export default function ProfileGuard({ children }: ProfileGuardProps) {
  const { user, loading } = useAuth(); // Fixed to use 'loading' instead of 'isLoading'
  const router = useRouter();
  const pathname = usePathname();
  const [isChecking, setIsChecking] = useState(false);
  const [profileCompleted, setProfileCompleted] = useState(false);

  // Pages that don't require profile completion
  const exemptPaths = [
    '/login',
    '/register',
    '/forgot-password',
    '/reset-password',
    '/verify-email',
    '/profile/update'
  ];

  const isExemptPath = exemptPaths.some(path => pathname.startsWith(path));

  useEffect(() => {
    const checkProfileCompletion = async () => {
      // Skip check if user is not logged in, auth is loading, or on exempt pages

      if (!user || loading || isExemptPath) {
        setProfileCompleted(true);
        return;
      }

      // Skip check for admin users
      if (user.role === 'ADMIN') {
        setProfileCompleted(true);
        return;
      }

      setIsChecking(true);
      try {
        const isCompleted = await userService.checkProfileCompletion();
        setProfileCompleted(isCompleted);
        console.log('isCompleted', isCompleted);
        // If profile is not completed, redirect to update profile page
        if (!isCompleted) {
          router.push('/profile/update');
        }
      } catch (error) {
        console.error('Error checking profile completion:', error);
        // If there's an error, assume profile needs completion
        setProfileCompleted(false);
        router.push('/profile/update');
      } finally {
        setIsChecking(false);
      }
    };

    checkProfileCompletion();
  }, [user, loading, isExemptPath, router]);

  // Show loading while checking auth or profile completion
  if (loading || (isChecking && !isExemptPath)) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <LoadingSpinner size="lg" />
          <p className="mt-4 text-gray-600">Đang kiểm tra thông tin người dùng...</p>
        </div>
      </div>
    );
  }

  // If user needs to complete profile and not on exempt page, show loading
  // (they will be redirected by the useEffect)
  if (user && !profileCompleted && !isExemptPath) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <LoadingSpinner size="lg" />
          <p className="mt-4 text-gray-600">Đang chuyển hướng đến trang cập nhật thông tin...</p>
        </div>
      </div>
    );
  }

  return <>{children}</>;
}
