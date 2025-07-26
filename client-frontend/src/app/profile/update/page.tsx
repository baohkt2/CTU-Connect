'use client';

import { useEffect, useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { userService } from '@/services/userService';
import { User } from '@/types';
import StudentProfileForm from '@/components/profile/StudentProfileForm';
import FacultyProfileForm from '@/components/profile/FacultyProfileForm';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import {useRouter} from "next/navigation";

export default function UpdateProfilePage() {
  const { user } = useAuth();
  const router = useRouter();
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!user) {
      setLoading(false);
      router.push('/login');
    }
    const fetchUserProfile = async () => {
      try {
        const profile = await userService.getMyProfile();
        setCurrentUser(profile);
        console.log('Current user profile:', profile);
      } catch (err) {
        console.error('Error fetching user profile:', err);
        setError('Không thể tải thông tin người dùng');
      } finally {
        setLoading(false);
      }
    };

    fetchUserProfile();
  }, [user]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner />
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-red-600 mb-4">Lỗi</h2>
          <p className="text-gray-600">{error}</p>
        </div>
      </div>
    );
  }

  if (!currentUser) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Không tìm thấy thông tin người dùng</h2>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="bg-white rounded-lg shadow-lg p-8">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">
              Cập nhật thông tin cá nhân
            </h1>
            <p className="text-gray-600">
              Vui lòng hoàn thiện thông tin để sử dụng đầy đủ các tính năng của hệ thống
            </p>
          </div>

          {currentUser.role === 'STUDENT' && (
            <StudentProfileForm user={currentUser} />
          )}

          {currentUser.role === 'FACULTY' && (
            <FacultyProfileForm user={currentUser} />
          )}

          {!['STUDENT', 'FACULTY'].includes(currentUser.role) && (
            <div className="text-center py-8">
              <p className="text-gray-600">
                Vai trò người dùng không hợp lệ để cập nhật thông tin cá nhân.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
