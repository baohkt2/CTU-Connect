/*
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
  const { user, loading } = useAuth();
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

  console.log('DEBUG: ProfileGuard - pathname:', pathname);
  console.log('DEBUG: ProfileGuard - user:', !!user);
  console.log('DEBUG: ProfileGuard - loading:', loading);
  console.log('DEBUG: ProfileGuard - isExemptPath:', isExemptPath);
  console.log('DEBUG: ProfileGuard - profileCompleted:', profileCompleted);

  // Handle authenticated users on login page - redirect them away
  useEffect(() => {
    if (!loading && user && pathname === '/login') {
      console.log('DEBUG: Authenticated user on login page, redirecting to home');
      router.replace('/');
      return;
    }
  }, [user, loading, pathname, router]);

  useEffect(() => {
    const checkProfileCompletion = async () => {
      console.log('DEBUG: Starting profile completion check');
      console.log('DEBUG: Conditions - user:', !!user, 'loading:', loading, 'isExemptPath:', isExemptPath);

      // Skip check if user is not logged in, auth is loading, or on exempt pages
      if (!user || loading || isExemptPath) {
        console.log('DEBUG: Skipping profile check - setting profileCompleted to true');
        setProfileCompleted(true);
        return;
      }

      // Skip check for admin users
      if (user.role === 'ADMIN') {
        console.log('DEBUG: Admin user detected, skipping profile check');
        setProfileCompleted(true);
        return;
      }

      console.log('DEBUG: Starting profile completion API check for user:', user.id);
      setIsChecking(true);

      try {
        const isCompleted = await userService.checkProfileCompletion();
        console.log('DEBUG: Profile completion result:', isCompleted);
        setProfileCompleted(isCompleted);

        // If profile is not completed, redirect to update profile page
        if (!isCompleted) {
          console.log('DEBUG: Profile not completed, redirecting to /profile/update');
          router.replace('/profile/update');
        } else {
          console.log('DEBUG: Profile completed, staying on current page');
        }
      } catch (error) {
        console.error('DEBUG: Error checking profile completion:', error);
        // If there's an error checking profile, don't force redirect on login-related pages
        if (!isExemptPath) {
          console.log('DEBUG: Profile check failed, redirecting to /profile/update');
          setProfileCompleted(false);
          router.replace('/profile/update');
        } else {
          console.log('DEBUG: Profile check failed but on exempt path, allowing access');
          setProfileCompleted(true);
        }
      } finally {
        setIsChecking(false);
      }
    };

    // Only run profile check if not on login page and user exists
    if (pathname !== '/login') {
      checkProfileCompletion();
    } else {
      // On login page, always allow access
      setProfileCompleted(true);
    }
  }, [user, loading, isExemptPath, router, pathname]);

  // Show loading while checking auth or profile completion
  if (loading) {
    console.log('DEBUG: Showing loading spinner - auth loading');
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <LoadingSpinner size="lg" />
          <p className="mt-4 text-gray-600">Đang kiểm tra thông tin xác thực...</p>
        </div>
      </div>
    );
  }

  if (isChecking && !isExemptPath && user) {
    console.log('DEBUG: Showing loading spinner - profile checking');
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
    console.log('DEBUG: Showing loading spinner - redirecting for profile completion');
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <LoadingSpinner size="lg" />
          <p className="mt-4 text-gray-600">Đang chuyển hướng đến trang cập nhật thông tin...</p>
        </div>
      </div>
    );
  }

  console.log('DEBUG: ProfileGuard - rendering children');
  return <>{children}</>;
}
*/
