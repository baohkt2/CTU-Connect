import React from 'react';
import { useChat } from '@/contexts/ChatContext';

const UserPresenceBar: React.FC = () => {
  const { onlineUsers } = useChat();

  if (onlineUsers.size === 0) {
    return null;
  }

  return (
    <div className="p-3 bg-gray-50 border-t border-gray-200">
      <div className="flex items-center space-x-2">
        <div className="flex items-center space-x-1">
          <div className="w-2 h-2 bg-green-400 rounded-full"></div>
          <span className="text-sm text-gray-600">
            {onlineUsers.size} online
          </span>
        </div>
      </div>
    </div>
  );
};

export default UserPresenceBar;
