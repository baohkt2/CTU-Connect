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
  ArrowRightOnRectangleIcon,
  UserGroupIcon
} from '@heroicons/react/24/outline';
import {
  HomeIcon as HomeIconSolid,
  UserIcon as UserIconSolid,
  ChatBubbleLeftRightIcon as ChatIconSolid,
  MagnifyingGlassIcon as SearchIconSolid,
  BellIcon as BellIconSolid,
  UserGroupIcon as UserGroupIconSolid
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
      iconSolid: HomeIconSolid,
      badge: undefined
    },
    {
      name: 'Bạn bè',
      href: '/friends',
      icon: UserGroupIcon,
      iconSolid: UserGroupIconSolid,
      badge: undefined
    },
    {
      name: 'Tìm kiếm',
      href: '/search',
      icon: MagnifyingGlassIcon,
      iconSolid: SearchIconSolid,
      badge: undefined
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
      iconSolid: BellIconSolid,
      badge: undefined
    },
    {
      name: 'Hồ sơ',
      href: `/profile/me`,
      icon: UserIcon,
      iconSolid: UserIconSolid,
      badge: undefined
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
        {/* Header: thanh bar nằm trên cùng - Enhanced responsive */}
        <header className="bg-white shadow-sm border-b sticky top-0 z-50 backdrop-blur-sm bg-white/95">
          <div className="max-w-full mx-auto px-3 sm:px-4 lg:px-6 xl:px-8">
            <div className="flex justify-between items-center h-14 sm:h-16">
              {/* Logo - Responsive sizing */}
              <Link href="/" className="flex items-center space-x-2 hover:opacity-80 transition-opacity">
                <div className="w-7 h-7 sm:w-8 sm:h-8 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-lg flex items-center justify-center shadow-sm">
                  <span className="text-white font-bold text-xs sm:text-sm">CTU</span>
                </div>
                <span className="text-lg sm:text-xl font-bold text-gray-900 hidden xs:block">Connect</span>
              </Link>

              {/* Navigation - Enhanced breakpoints */}
              <nav className="hidden lg:flex space-x-1 xl:space-x-2">
                {navigation.slice(0, 3).map((item) => {
                  const IconComponent = isActive(item.href) ? item.iconSolid : item.icon;
                  return (
                    <Link
                      key={item.name}
                      href={item.href}
                      className={`relative flex items-center px-2 lg:px-3 xl:px-4 py-2 rounded-xl text-xs lg:text-sm font-medium transition-all duration-200 ${
                        isActive(item.href)
                          ? 'bg-indigo-50 text-indigo-600 shadow-sm'
                          : 'text-gray-600 hover:text-indigo-600 hover:bg-gray-50'
                      }`}
                    >
                      <IconComponent className="h-4 w-4 lg:h-5 lg:w-5 mr-1 lg:mr-2" />
                      <span className="hidden xl:inline">{item.name}</span>
                      {item.badge && item.badge > 0 && (
                        <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full h-4 w-4 lg:h-5 lg:w-5 flex items-center justify-center font-medium animate-pulse">
                          {item.badge > 99 ? '99+' : item.badge}
                        </span>
                      )}
                    </Link>
                  );
                })}
              </nav>

              {/* User Menu - Enhanced responsive */}
              <div className="flex items-center space-x-2 sm:space-x-3 lg:space-x-4">
                {/* Desktop notifications and settings */}
                <div className="hidden xl:flex items-center space-x-1">
                  <button className="relative p-2 text-gray-600 hover:text-indigo-600 hover:bg-gray-50 rounded-lg transition-all duration-200">
                    <BellIcon className="h-5 w-5" />
                    <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full h-4 w-4 flex items-center justify-center">3</span>
                  </button>

                  <button className="p-2 text-gray-600 hover:text-indigo-600 hover:bg-gray-50 rounded-lg transition-all duration-200">
                    <Cog6ToothIcon className="h-5 w-5" />
                  </button>
                </div>

                {/* User Avatar & Info - Enhanced responsive */}
                <div className="flex items-center space-x-2 sm:space-x-3">
                  <div className="hidden md:block text-right">
                    <div className="text-sm font-medium text-gray-900 vietnamese-text truncate max-w-24 lg:max-w-32">
                      {user?.fullName || user?.name || 'Người dùng'}
                    </div>
                    <div className="text-xs text-gray-500 truncate max-w-24 lg:max-w-32">
                      {user?.email || 'Email không xác định'}
                    </div>
                  </div>
                  <div className="relative">
                    <Avatar
                        id={user?.id}
                      src={user?.avatarUrl || '/default-avatar.png'}
                      alt={user?.fullName || user?.username || 'Avatar'}
                      size="sm"
                      className="sm:w-10 sm:h-10"
                     />
                    <div className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 sm:w-3 sm:h-3 bg-green-500 border-2 border-white rounded-full"></div>
                  </div>
                </div>

                {/* Logout - Enhanced responsive */}
                <button
                  onClick={handleLogout}
                  className="flex items-center px-2 sm:px-3 py-2 text-xs sm:text-sm font-medium text-gray-600 hover:text-red-600 hover:bg-red-50 rounded-lg transition-all duration-200"
                  title="Đăng xuất"
                >
                  <ArrowRightOnRectangleIcon className="h-4 w-4 sm:h-5 sm:w-5 mr-0 sm:mr-1" />
                  <span className="hidden md:inline">{t('auth.logout')}</span>
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* Main Layout với Sidebars - Enhanced responsive */}
        <div className="flex flex-1">
          {/* Left Sidebar */}
          <LeftSidebar />

          {/* Main Content - Enhanced responsive padding */}
          <main className="flex-1 min-w-0 px-2 py-3 sm:px-4 sm:py-4 lg:px-6 lg:py-6 xl:px-8 xl:py-8 pb-20 lg:pb-6">
            <div className="min-h-full">
              {children}
            </div>
          </main>

          {/* Right Sidebar */}
          <RightSidebar />
        </div>

        {/* Mobile Navigation - Enhanced */}
        <div className="lg:hidden bg-white border-t border-gray-200 fixed bottom-0 left-0 right-0 z-50 shadow-lg backdrop-blur-sm bg-white/95 safe-area-inset-bottom">
          <div className="flex justify-around py-1 sm:py-2 px-1">
            {navigation.map((item) => {
              const IconComponent = isActive(item.href) ? item.iconSolid : item.icon;
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={`relative flex flex-col items-center px-2 sm:px-3 py-1.5 sm:py-2 rounded-lg transition-all duration-200 min-w-0 ${
                    isActive(item.href)
                      ? 'text-indigo-600'
                      : 'text-gray-600 hover:text-indigo-600'
                  }`}
                >
                  <IconComponent className="h-5 w-5 sm:h-6 sm:w-6 flex-shrink-0" />
                  <span className="text-xs mt-0.5 sm:mt-1 font-medium vietnamese-text truncate max-w-full">{item.name}</span>
                  {item.badge && item.badge > 0 && (
                    <span className="absolute -top-0.5 -right-0.5 sm:-top-1 sm:-right-1 bg-red-500 text-white text-xs rounded-full h-4 w-4 flex items-center justify-center font-medium animate-pulse">
                      {item.badge > 9 ? '9+' : item.badge}
                    </span>
                  )}
                </Link>
              );
            })}
          </div>
        </div>

        {/* Footer - Enhanced responsive */}
        <footer className="lg:hidden bg-white border-t border-gray-200 py-6 sm:py-8 mt-auto">
          <div className="max-w-7xl mx-auto px-3 sm:px-4 lg:px-6">
            <div className="flex flex-col sm:flex-row justify-between items-center text-center sm:text-left">
              <div className="flex items-center space-x-2 mb-3 sm:mb-0">
                <div className="w-6 h-6 bg-gradient-to-br from-indigo-600 to-purple-600 rounded flex items-center justify-center">
                  <span className="text-white font-bold text-xs">CTU</span>
                </div>
                <span className="text-gray-900 font-medium">CTU Connect</span>
              </div>
              <div className="text-sm text-gray-500">
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
