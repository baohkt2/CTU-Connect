'use client';

import React, { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { User, Post } from '@/types';
import { userService } from '@/services/userService';
import { postService } from '@/services/postService';
import { ProfileHeader } from '@/components/profile/ProfileHeader';
import { ProfileStats } from '@/components/profile/ProfileStats';
import { StudentProfileInfo } from '@/components/profile/StudentProfileInfo';
import { LecturerProfileInfo } from '@/components/profile/LecturerProfileInfo';
import { ProfilePostFeed } from '@/components/profile/ProfilePostFeed';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { ErrorAlert } from '@/components/ui/ErrorAlert';

interface UserProfileProps {
  userId: string;
}

const UserProfile: React.FC<UserProfileProps> = ({ userId }) => {
  const { user: currentUser } = useAuth();
  const [profileUser, setProfileUser] = useState<User | null>(null);
  const [userPosts, setUserPosts] = useState<Post[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isFollowing, setIsFollowing] = useState(false);
  const [activeTab, setActiveTab] = useState<'posts' | 'about' | 'photos' | 'videos'>('posts');
  const [stats, setStats] = useState({
    posts: 0,
    followers: 0,
    following: 0,
    likes: 0,
    views: 0
  });

  const isOwnProfile = currentUser?.id === userId;

  useEffect(() => {
    loadUserProfile();
    if (userId) {
      loadUserPosts();
      loadUserStats();
    }
  }, [userId]);

  const loadUserProfile = async () => {
    try {
      setIsLoading(true);
      // Use the correct method based on whether it's own profile or not
      const user = isOwnProfile
        ? await userService.getMyProfile()
        : await userService.getProfile(userId);
      setProfileUser(user);

      // Check if current user is following this user
      if (!isOwnProfile && currentUser) {
        // TODO: Implement follow status check
        // const followStatus = await userService.getFollowStatus(userId);
        // setIsFollowing(followStatus.isFollowing);
      }
    } catch (err) {
      setError('Không thể tải thông tin người dùng');
      console.error('Error loading user profile:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const loadUserPosts = async () => {
    try {
      // TODO: Implement user-specific posts loading
      //  const posts = await postService.getMyPosts();
      // setUserPosts(posts);
      setUserPosts([]); // Temporary empty array
    } catch (err) {
      console.error('Error loading user posts:', err);
    }
  };

  const loadUserStats = async () => {
    try {
      // TODO: Implement stats loading from backend
      // For now, using mock data
      setStats({
        posts: Math.floor(Math.random() * 100),
        followers: Math.floor(Math.random() * 1000),
        following: Math.floor(Math.random() * 500),
        likes: Math.floor(Math.random() * 5000),
        views: Math.floor(Math.random() * 10000)
      });
    } catch (err) {
      console.error('Error loading user stats:', err);
    }
  };

  const handleFollow = async () => {
    try {
      if (isFollowing) {
        await userService.unfollowUser(userId);
        setIsFollowing(false);
        setStats(prev => ({ ...prev, followers: prev.followers - 1 }));
      } else {
        await userService.followUser(userId);
        setIsFollowing(true);
        setStats(prev => ({ ...prev, followers: prev.followers + 1 }));
      }
    } catch (err) {
      console.error('Error updating follow status:', err);
    }
  };

  const handleMessage = () => {
    // TODO: Implement messaging functionality
    console.log('Open chat with user:', userId);
  };

  const handleEditProfile = () => {
    // TODO: Implement profile editing
    console.log('Edit profile');
  };

  const handleEditCover = () => {
    // TODO: Implement cover photo editing
    console.log('Edit cover photo');
  };

  const handleEditAvatar = () => {
    // TODO: Implement avatar editing
    console.log('Edit avatar');
  };

  const handleStatsClick = (type: 'posts' | 'followers' | 'following') => {
    switch (type) {
      case 'posts':
        setActiveTab('posts');
        break;
      case 'followers':
        // TODO: Show followers modal/page
        console.log('Show followers');
        break;
      case 'following':
        // TODO: Show following modal/page
        console.log('Show following');
        break;
    }
  };

  if (isLoading) {
    return (
      <div className="max-w-4xl mx-auto p-4">
        <div className="flex justify-center items-center h-64">
          <LoadingSpinner size="lg" />
        </div>
      </div>
    );
  }

  if (error || !profileUser) {
    return (
      <div className="max-w-4xl mx-auto p-4">
        <ErrorAlert message={error || 'Không tìm thấy người dùng'} />
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-4 space-y-6">
      {/* Profile Header */}
      <ProfileHeader
        user={profileUser}
        isOwnProfile={isOwnProfile}
        isFollowing={isFollowing}
        onFollow={handleFollow}
        onMessage={handleMessage}
        onEditProfile={handleEditProfile}
        onEditCover={handleEditCover}
        onEditAvatar={handleEditAvatar}
      />

      {/* Profile Stats */}
      <ProfileStats
        stats={stats}
        onStatsClick={handleStatsClick}
      />

      {/* Navigation Tabs */}
      <div className="bg-white rounded-lg shadow-sm">
        <div className="border-b border-gray-200">
          <nav className="flex space-x-8 px-6">
            {[
              { key: 'posts', label: 'Bài viết' },
              { key: 'about', label: 'Giới thiệu' },
              { key: 'photos', label: 'Ảnh' },
              { key: 'videos', label: 'Video' }
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => setActiveTab(tab.key as any)}
                className={`py-4 px-1 border-b-2 font-medium text-sm vietnamese-text transition-colors ${
                  activeTab === tab.key
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Tab Content */}
        <div className="p-6">
          {activeTab === 'posts' && (
            <ProfilePostFeed
              userId={userId}
              userName={profileUser.fullName || profileUser.name}
              isOwnProfile={isOwnProfile}
            />
          )}

          {activeTab === 'about' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {profileUser.role === 'STUDENT' ? (
                <StudentProfileInfo user={profileUser} />
              ) : (
                <LecturerProfileInfo user={profileUser} />
              )}

              {/* Additional Info Section */}
              <div className="space-y-6">
                {profileUser.bio && (
                  <div className="bg-white rounded-lg shadow-sm p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4 vietnamese-text">
                      Giới thiệu bản thân
                    </h3>
                    <p className="text-gray-700 leading-relaxed vietnamese-text">
                      {profileUser.bio}
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'photos' && (
            <div className="text-center py-12">
              <div className="text-gray-400 mb-4">
                <svg className="w-16 h-16 mx-auto" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
                </svg>
              </div>
              <p className="text-gray-500 vietnamese-text">Chức năng xem ảnh đang được phát triển</p>
            </div>
          )}

          {activeTab === 'videos' && (
            <div className="text-center py-12">
              <div className="text-gray-400 mb-4">
                <svg className="w-16 h-16 mx-auto" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M2 6a2 2 0 012-2h6a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6zM14.553 7.106A1 1 0 0014 8v4a1 1 0 00.553.894l2 1A1 1 0 0018 13V7a1 1 0 00-1.447-.894l-2 1z" clipRule="evenodd" />
                </svg>
              </div>
              <p className="text-gray-500 vietnamese-text">Chức năng xem video đang được phát triển</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default UserProfile;
