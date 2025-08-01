import React, { ReactNode } from 'react';
import { useAuth } from '@/contexts/AuthContext';
import { useChatHooks } from '@/hooks/useChatHooks';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useRouter } from 'next/navigation';
import Avatar from '@/components/ui/Avatar';
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
    await logout();
    router.push('/login');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo */}
            <Link href="/" className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">CTU</span>
              </div>
              <span className="text-xl font-bold text-gray-900">Connect</span>
            </Link>

            {/* Navigation */}
            <nav className="hidden md:flex space-x-8">
              {navigation.map((item) => {
                const Icon = isActive(item.href) ? item.iconSolid : item.icon;
                return (
                  <Link
                    key={item.name}
                    href={item.href}
                    className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium transition-colors relative ${
                      isActive(item.href)
                        ? 'text-blue-600 bg-blue-50'
                        : 'text-gray-700 hover:text-blue-600 hover:bg-gray-50'
                    }`}
                  >
                    <Icon className="w-5 h-5" />
                    <span>{item.name}</span>
                    {item.badge && item.badge > 0 && (
                      <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                        {item.badge > 99 ? '99+' : item.badge}
                      </span>
                    )}
                  </Link>
                );
              })}
            </nav>

            {/* User Menu */}
            <div className="flex items-center space-x-4">
              <div className="relative group">
                <button className="flex items-center space-x-2 p-2 rounded-md hover:bg-gray-100">
                  <Avatar
                    src={user?.avatarUrl || '/default-avatar.png'}
                    alt={user?.fullName || 'User'}
                    size="sm"
                    online={user?.isOnline}
                  />
                  <span className="hidden md:block text-sm font-medium text-gray-700">
                    {user?.fullName}
                  </span>
                </button>

                {/* Dropdown Menu */}
                <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 z-50 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all">
                  <Link
                    href={`/profile/${user?.id}`}
                    className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                  >
                    <UserIcon className="w-4 h-4 inline mr-2" />
                    Hồ sơ của tôi
                  </Link>
                  <Link
                    href="/settings"
                    className="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                  >
                    <Cog6ToothIcon className="w-4 h-4 inline mr-2" />
                    Cài đặt
                  </Link>
                  <button
                    onClick={handleLogout}
                    className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
                  >
                    <ArrowRightOnRectangleIcon className="w-4 h-4 inline mr-2" />
                    Đăng xuất
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>

      {/* Mobile Navigation */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 px-4 py-2 z-40">
        <div className="flex justify-around">
          {navigation.map((item) => {
            const Icon = isActive(item.href) ? item.iconSolid : item.icon;
            return (
              <Link
                key={item.name}
                href={item.href}
                className={`flex flex-col items-center space-y-1 p-2 rounded-md relative ${
                  isActive(item.href)
                    ? 'text-blue-600'
                    : 'text-gray-700'
                }`}
              >
                <Icon className="w-6 h-6" />
                <span className="text-xs">{item.name}</span>
                {item.badge && item.badge > 0 && (
                  <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
                    {item.badge > 99 ? '99+' : item.badge}
                  </span>
                )}
              </Link>
            );
          })}
        </div>
      </nav>
    </div>
  );
};

export default Layout;
