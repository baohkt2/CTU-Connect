import React, { ReactNode } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { useChatHooks } from '@/hooks/useChatHooks';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useRouter } from 'next/navigation';
import Avatar from '@/components/ui/Avatar';
import LeftSidebar from './LeftSidebar';
import RightSidebar from './RightSidebar';
import { t } from '@/utils/localization';
import {
  HomeIcon,
  UserIcon,
  ChatBubbleLeftRightIcon,
  MagnifyingGlassIcon,
  BellIcon,
  Cog6ToothIcon,
  ArrowRightOnRectangleIcon
} from '@heroicons/react/24/outline';
import {
  HomeIcon as HomeIconSolid,
  UserIcon as UserIconSolid,
  ChatBubbleLeftRightIcon as ChatIconSolid,
  MagnifyingGlassIcon as SearchIconSolid,
  BellIcon as BellIconSolid
} from '@heroicons/react/24/solid';

interface LayoutProps {
  children: ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const { user, logout } = useAuth();
  const router = useRouter();
  const { useUnreadCount } = useChatHooks();
  const { data: unreadCount } = useUnreadCount();

  const navigation = [
    {
      name: 'Trang chủ',
      href: '/',
      icon: HomeIcon,
      iconSolid: HomeIconSolid
    },
    {
      name: 'Tìm kiếm',
      href: '/search',
      icon: MagnifyingGlassIcon,
      iconSolid: SearchIconSolid
    },
    {
      name: 'Tin nhắn',
      href: '/messages',
      icon: ChatBubbleLeftRightIcon,
      iconSolid: ChatIconSolid,
      badge: unreadCount
    },
    {
      name: 'Thông báo',
      href: '/notifications',
      icon: BellIcon,
      iconSolid: BellIconSolid
    },
    {
      name: 'Hồ sơ',
      href: `/profile/me`,
      icon: UserIcon,
      iconSolid: UserIconSolid
    },
  ];

  const pathname = usePathname();

  const isActive = (href: string) => {
    if (href === '/') {
      return pathname === '/';
    }
    return pathname.startsWith(href);
  };

  const handleLogout = async () => {
    try {
      await logout();
      router.push('/login');
    } catch (error) {
      console.error('Lỗi đăng xuất:', error);
    }
  };


  return (
      <div className="flex flex-col min-h-screen bg-gray-50">
        {/* Header: thanh bar nằm trên cùng */}
        <header className="bg-white shadow-sm border-b sticky top-0 z-50 backdrop-blur-sm bg-white/95">
          <div className="max-w-full mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between items-center h-16">
              {/* Logo */}
              <Link href="/" className="flex items-center space-x-2 hover:opacity-80 transition-opacity">
                <div className="w-8 h-8 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-lg flex items-center justify-center shadow-sm">
                  <span className="text-white font-bold text-sm">CTU</span>
                </div>
                <span className="text-xl font-bold text-gray-900">Connect</span>
              </Link>

              {/* Navigation - Chỉ hiển thị trên desktop vì đã có sidebar */}
              <nav className="hidden xl:flex space-x-1">
                {navigation.slice(0, 3).map((item) => {
                  const IconComponent = isActive(item.href) ? item.iconSolid : item.icon;
                  return (
                    <Link
                      key={item.name}
                      href={item.href}
                      className={`relative flex items-center px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 ${
                        isActive(item.href)
                          ? 'bg-indigo-50 text-indigo-600 shadow-sm'
                          : 'text-gray-600 hover:text-indigo-600 hover:bg-gray-50'
                      }`}
                    >
                      <IconComponent className="h-5 w-5 mr-2" />
                      <span>{item.name}</span>
                      {item.badge && item.badge > 0 && (
                        <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full h-5 w-5 flex items-center justify-center font-medium animate-pulse">
                          {item.badge > 99 ? '99+' : item.badge}
                        </span>
                      )}
                    </Link>
                  );
                })}
              </nav>

              {/* User Menu */}
              <div className="flex items-center space-x-4">
                <div className="hidden xl:flex items-center space-x-2">
                  {/* Notifications */}
                  <button className="relative p-2 text-gray-600 hover:text-indigo-600 hover:bg-gray-50 rounded-lg transition-all duration-200">
                    <BellIcon className="h-5 w-5" />
                    <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full h-4 w-4 flex items-center justify-center">3</span>
                  </button>

                  {/* Settings */}
                  <button className="p-2 text-gray-600 hover:text-indigo-600 hover:bg-gray-50 rounded-lg transition-all duration-200">
                    <Cog6ToothIcon className="h-5 w-5" />
                  </button>
                </div>

                {/* User Avatar & Info */}
                <div className="flex items-center space-x-3">
                  <div className="hidden sm:block text-right">
                    <div className="text-sm font-medium text-gray-900 vietnamese-text">
                      {user?.fullName || user?.name || 'Người dùng'}
                    </div>
                    <div className="text-xs text-gray-500">
                      {user?.email || 'Email không xác định'}
                    </div>
                  </div>
                  <div className="relative">
                    <Avatar
                      src={user?.avatarUrl || '/default-avatar.png'}
                      alt={user?.fullName || user?.username || 'Avatar'}
                      size="md"
                     />
                    <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-green-500 border-2 border-white rounded-full"></div>
                  </div>
                </div>

                {/* Logout */}
                <button
                  onClick={handleLogout}
                  className="flex items-center px-3 py-2 text-sm font-medium text-gray-600 hover:text-red-600 hover:bg-red-50 rounded-lg transition-all duration-200"
                  title="Đăng xuất"
                >
                  <ArrowRightOnRectangleIcon className="h-5 w-5 mr-1" />
                  <span className="hidden sm:inline">{t('auth.logout')}</span>
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* Main Layout với Sidebars */}
        <div className="flex flex-1">
          {/* Left Sidebar */}
          <LeftSidebar />

          {/* Main Content */}
          <main className="flex-1 min-w-0 px-4 py-6 sm:px-6 lg:px-8">
            <div className="min-h-full">
              {children}
            </div>
          </main>

          {/* Right Sidebar */}
          <RightSidebar />
        </div>

        {/* Mobile Navigation */}
        <div className="lg:hidden bg-white border-t border-gray-200 fixed bottom-0 left-0 right-0 z-50 shadow-lg backdrop-blur-sm bg-white/95">
          <div className="flex justify-around py-2">
            {navigation.map((item) => {
              const IconComponent = isActive(item.href) ? item.iconSolid : item.icon;
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`relative flex flex-col items-center px-3 py-2 rounded-lg transition-all duration-200 ${
                    isActive(item.href)
                      ? 'text-indigo-600'
                      : 'text-gray-600 hover:text-indigo-600'
                  }`}
                >
                  <IconComponent className="h-6 w-6" />
                  <span className="text-xs mt-1 font-medium vietnamese-text">{item.name}</span>
                  {item.badge && item.badge > 0 && (
                    <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full h-4 w-4 flex items-center justify-center font-medium animate-pulse">
                      {item.badge > 9 ? '9+' : item.badge}
                    </span>
                  )}
                </Link>
              );
            })}
          </div>
        </div>

        {/* Footer - Chỉ hiển thị khi không có sidebar */}
        <footer className="lg:hidden bg-white border-t border-gray-200 py-8 mt-auto">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex flex-col md:flex-row justify-between items-center">
              <div className="flex items-center space-x-2 mb-4 md:mb-0">
                <div className="w-6 h-6 bg-gradient-to-br from-indigo-600 to-purple-600 rounded flex items-center justify-center">
                  <span className="text-white font-bold text-xs">CTU</span>
                </div>
                <span className="text-gray-900 font-medium">CTU Connect</span>
              </div>
              <div className="text-sm text-gray-500 text-center md:text-right">
                <p>&copy; 2025 Đại học Cần Thơ. Phát triển bởi sinh viên CTU.</p>
                <p className="mt-1">Kết nối - Chia sẻ - Phát triển</p>
              </div>
            </div>
          </div>
        </footer>
      </div>
  );
};

export default Layout;
