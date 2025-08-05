'use client';

import { useEffect, useState } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { userService } from '@/services/userService';
import { User } from '@/types';
import StudentProfileForm from '@/components/profile/StudentProfileForm';
import LecturerProfileForm from '@/components/profile/LecturerProfileForm';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import Layout from '@/components/layout/Layout';
import { useRouter } from "next/navigation";
import { ArrowLeft, User as UserIcon } from 'lucide-react';
import { Button } from '@/components/ui/Button';

export default function UpdateProfilePage() {
  const { user } = useAuth();
  const router = useRouter();
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!user) {
      router.push('/login');
      return;
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
  }, [user, router]);

  const handleBackToProfile = () => {
    router.push('/profile/me');
  };

  if (loading) {
    return (
      <Layout>
        <div className="min-h-screen flex items-center justify-center bg-gray-50">
          <div className="text-center">
            <LoadingSpinner size="lg" />
            <p className="text-gray-600 mt-4 vietnamese-text">Đang tải thông tin cá nhân...</p>
          </div>
        </div>
      </Layout>
    );
  }

  if (error) {
    return (
      <Layout>
        <div className="min-h-screen flex items-center justify-center bg-gray-50">
          <div className="text-center bg-white rounded-lg shadow-sm p-8 max-w-md">
            <div className="text-red-500 mb-4">
              <UserIcon className="h-16 w-16 mx-auto" />
            </div>
            <h2 className="text-2xl font-bold text-red-600 mb-4 vietnamese-text">Lỗi</h2>
            <p className="text-gray-600 mb-6 vietnamese-text">{error}</p>
            <Button onClick={() => window.location.reload()} className="mr-3">
              Thử lại
            </Button>
            <Button variant="outline" onClick={() => router.push('/profile/me')}>
              Quay lại
            </Button>
          </div>
        </div>
      </Layout>
    );
  }

  if (!currentUser) {
    return (
      <Layout>
        <div className="min-h-screen flex items-center justify-center bg-gray-50">
          <div className="text-center bg-white rounded-lg shadow-sm p-8 max-w-md">
            <div className="text-gray-400 mb-4">
              <UserIcon className="h-16 w-16 mx-auto" />
            </div>
            <h2 className="text-2xl font-bold text-gray-800 mb-4 vietnamese-text">
              Không tìm thấy thông tin người dùng
            </h2>
            <Button onClick={() => router.push('/profile/me')}>
              Quay lại trang cá nhân
            </Button>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="min-h-screen bg-gray-50 py-8">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Header with Back Button */}
          <div className="mb-6">
            <Button
              variant="outline"
              onClick={handleBackToProfile}
              className="flex items-center space-x-2 mb-4"
            >
              <ArrowLeft className="h-4 w-4" />
              <span className="vietnamese-text">Quay lại trang cá nhân</span>
            </Button>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-8">
            <div className="text-center mb-8">
              <div className="mb-4">
                {currentUser.avatarUrl ? (
                  <img
                    src={currentUser.avatarUrl}
                    alt="Avatar"
                    className="w-20 h-20 rounded-full mx-auto object-cover border-4 border-white shadow-lg"
                  />
                ) : (
                  <div className="w-20 h-20 bg-gradient-to-br from-blue-400 to-purple-600 rounded-full mx-auto flex items-center justify-center text-white text-2xl font-bold shadow-lg">
                    {(currentUser.fullName || currentUser.name || 'U').charAt(0).toUpperCase()}
                  </div>
                )}
              </div>
              
              <h1 className="text-3xl font-bold text-gray-900 mb-2 vietnamese-text">
                Cập nhật thông tin cá nhân
              </h1>
              <p className="text-gray-600 vietnamese-text">
                Vui lòng hoàn thiện thông tin để sử dụng đầy đủ các tính năng của hệ thống
              </p>
              
              {/* Role Badge */}
              <div className="mt-4">
                <span className={`inline-flex px-3 py-1 rounded-full text-sm font-medium ${
                  currentUser.role === 'LECTURER' 
                    ? 'bg-blue-100 text-blue-700' 
                    : 'bg-green-100 text-green-700'
                }`}>
                  {currentUser.role === 'LECTURER' ? 'Giảng viên' : 'Sinh viên'}
                </span>
              </div>
            </div>

            {currentUser.role === 'STUDENT' && (
              <StudentProfileForm user={currentUser} />
            )}

            {currentUser.role === 'LECTURER' && (
              <LecturerProfileForm user={currentUser} />
            )}

            {!['STUDENT', 'LECTURER'].includes(currentUser.role) && (
              <div className="text-center py-8">
                <div className="text-gray-400 mb-4">
                  <UserIcon className="h-12 w-12 mx-auto" />
                </div>
                <p className="text-gray-600 vietnamese-text">
                  Vai trò người dùng không hợp lệ để cập nhật thông tin cá nhân.
                </p>
                <Button 
                  variant="outline" 
                  onClick={() => router.push('/profile/me')} 
                  className="mt-4"
                >
                  Quay lại trang cá nhân
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
}
