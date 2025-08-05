import React from 'react';
import { useChat } from '../../contexts/ChatContext';
import { UserPresence } from '../../services/chatService';
import { formatDistanceToNow } from 'date-fns';
import { vi } from 'date-fns/locale';

const UserPresenceBar: React.FC = () => {
  const { state } = useChat();

  const getStatusColor = (status: UserPresence['status']) => {
    switch (status) {
      case 'ONLINE':
        return 'bg-green-500';
      case 'AWAY':
        return 'bg-yellow-500';
      case 'OFFLINE':
        return 'bg-gray-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getStatusText = (user: UserPresence) => {
    if (user.status === 'ONLINE') {
      return user.currentActivity ? 'Đang gõ...' : 'Đang hoạt động';
    } else if (user.status === 'AWAY') {
      return 'Vắng mặt';
    } else {
      try {
        return `Hoạt động ${formatDistanceToNow(new Date(user.lastSeenAt), {
          addSuffix: true,
          locale: vi
        })}`;
      } catch {
        return 'Không hoạt động';
      }
    }
  };

  return (
    <div className="h-full bg-gray-50 overflow-y-auto">
      {/* Header */}
      <div className="p-4 border-b bg-white">
        <h3 className="font-medium text-gray-900">Người dùng</h3>
        <p className="text-sm text-gray-500">
          {state.onlineUsers.filter(u => u.status === 'ONLINE').length} đang online
        </p>
      </div>

      {/* Online Users */}
      <div className="p-4">
        <h4 className="text-sm font-medium text-gray-700 mb-3">Đang hoạt động</h4>
        <div className="space-y-3">
          {state.onlineUsers
            .filter(user => user.status === 'ONLINE')
            .map(user => (
              <div key={user.userId} className="flex items-center space-x-3">
                <div className="relative">
                  {user.userAvatar ? (
                    <img
                      src={user.userAvatar}
                      alt={user.userName}
                      className="w-8 h-8 rounded-full object-cover"
                    />
                  ) : (
                    <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center">
                      <span className="text-xs font-medium text-gray-600">
                        {user.userName.charAt(0).toUpperCase()}
                      </span>
                    </div>
                  )}
                  <div className={`absolute bottom-0 right-0 w-3 h-3 border-2 border-white rounded-full ${getStatusColor(user.status)}`}></div>
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">
                    {user.userName}
                  </p>
                  <p className="text-xs text-gray-500">
                    {getStatusText(user)}
                  </p>
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* Away Users */}
      {state.onlineUsers.some(u => u.status === 'AWAY') && (
        <div className="p-4 border-t">
          <h4 className="text-sm font-medium text-gray-700 mb-3">Vắng mặt</h4>
          <div className="space-y-3">
            {state.onlineUsers
              .filter(user => user.status === 'AWAY')
              .map(user => (
                <div key={user.userId} className="flex items-center space-x-3">
                  <div className="relative">
                    {user.userAvatar ? (
                      <img
                        src={user.userAvatar}
                        alt={user.userName}
                        className="w-8 h-8 rounded-full object-cover"
                      />
                    ) : (
                      <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center">
                        <span className="text-xs font-medium text-gray-600">
                          {user.userName.charAt(0).toUpperCase()}
                        </span>
                      </div>
                    )}
                    <div className={`absolute bottom-0 right-0 w-3 h-3 border-2 border-white rounded-full ${getStatusColor(user.status)}`}></div>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {user.userName}
                    </p>
                    <p className="text-xs text-gray-500">
                      {getStatusText(user)}
                    </p>
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Offline Users */}
      {state.onlineUsers.some(u => u.status === 'OFFLINE') && (
        <div className="p-4 border-t">
          <h4 className="text-sm font-medium text-gray-700 mb-3">Không hoạt động</h4>
          <div className="space-y-3">
            {state.onlineUsers
              .filter(user => user.status === 'OFFLINE')
              .slice(0, 10) // Limit to 10 offline users
              .map(user => (
                <div key={user.userId} className="flex items-center space-x-3 opacity-60">
                  <div className="relative">
                    {user.userAvatar ? (
                      <img
                        src={user.userAvatar}
                        alt={user.userName}
                        className="w-8 h-8 rounded-full object-cover"
                      />
                    ) : (
                      <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center">
                        <span className="text-xs font-medium text-gray-600">
                          {user.userName.charAt(0).toUpperCase()}
                        </span>
                      </div>
                    )}
                    <div className={`absolute bottom-0 right-0 w-3 h-3 border-2 border-white rounded-full ${getStatusColor(user.status)}`}></div>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {user.userName}
                    </p>
                    <p className="text-xs text-gray-500">
                      {getStatusText(user)}
                    </p>
                  </div>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Empty State */}
      {state.onlineUsers.length === 0 && (
        <div className="p-4 text-center text-gray-500">
          <p className="text-sm">Không có người dùng nào online</p>
        </div>
      )}
    </div>
  );
};

export default UserPresenceBar;
