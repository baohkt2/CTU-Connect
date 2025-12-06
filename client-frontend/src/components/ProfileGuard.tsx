'use client';
import { useEffect, useState, useRef } from 'react';
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
  const [profileCompleted, setProfileCompleted] = useState<boolean | null>(null);
  const hasCheckedRef = useRef(false);
  const lastCheckTimeRef = useRef<number>(0);

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

  useEffect(() => {
    const checkProfileCompletion = async () => {
      // Skip if already checking or recently checked
      const now = Date.now();
      if (hasCheckedRef.current && (now - lastCheckTimeRef.current) < 5000) {
        console.log('DEBUG: Skipping check - recently checked');
        return;
      }

      // Skip check if user is not logged in, auth is loading, or on exempt pages
      if (!user || loading || isExemptPath) {
        console.log('DEBUG: Skipping profile check');
        setProfileCompleted(true);
        return;
      }

      // Skip check for admin users
      if (user.role === 'ADMIN') {
        console.log('DEBUG: Admin user detected, skipping profile check');
        setProfileCompleted(true);
        return;
      }

      // Check if user object already indicates profile is complete
      if (user.fullName && user.studentId && user.major && user.batch && user.gender) {
        console.log('DEBUG: User object shows profile is complete');
        setProfileCompleted(true);
        hasCheckedRef.current = true;
        lastCheckTimeRef.current = now;
        return;
      }

      console.log('DEBUG: Starting profile completion API check for user:', user.id);
      setIsChecking(true);
      hasCheckedRef.current = true;
      lastCheckTimeRef.current = now;

      try {
        const isCompleted = await userService.checkProfileCompletion();
        console.log('DEBUG: Profile completion result:', isCompleted);
        setProfileCompleted(isCompleted);

        // If profile is not completed and not already on update page, redirect
        if (!isCompleted && pathname !== '/profile/update') {
          console.log('DEBUG: Profile not completed, redirecting to /profile/update');
          router.replace('/profile/update');
        } else {
          console.log('DEBUG: Profile completed or already on update page');
        }
      } catch (error) {
        console.error('DEBUG: Error checking profile completion:', error);
        // On error, don't force redirect - allow user to continue
        setProfileCompleted(true);
      } finally {
        setIsChecking(false);
      }
    };

    checkProfileCompletion();
  }, [user, loading, isExemptPath, pathname]);

  // Reset check flag when user changes
  useEffect(() => {
    hasCheckedRef.current = false;
    setProfileCompleted(null);
  }, [user?.id]);

  // Show loading while checking auth
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

  // Show loading while checking profile (only on first check)
  if (isChecking && profileCompleted === null && !isExemptPath && user) {
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

  console.log('DEBUG: ProfileGuard - rendering children');
  return <>{children}</>;
}

