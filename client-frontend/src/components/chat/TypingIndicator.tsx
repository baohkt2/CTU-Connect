import React from 'react';

interface TypingIndicatorProps {
  userIds: string[];
}

const TypingIndicator: React.FC<TypingIndicatorProps> = ({ userIds }) => {
  if (userIds.length === 0) return null;

  const getUserNames = () => {
    // TODO: Map userIds to actual user names
    // For now, just use generic names
    if (userIds.length === 1) {
      return 'Ai đó';
    } else if (userIds.length === 2) {
      return '2 người';
    } else {
      return `${userIds.length} người`;
    }
  };

  return (
    <div className="flex items-center space-x-2 px-4 py-2">
      <div className="flex items-center space-x-2">
        {/* Avatar placeholder */}
        <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center">
          <span className="text-xs font-medium text-gray-600">...</span>
        </div>
        
        {/* Typing animation */}
        <div className="bg-gray-200 rounded-2xl px-4 py-2 flex items-center space-x-1">
          <span className="text-sm text-gray-600">{getUserNames()} đang gõ</span>
          <div className="flex space-x-1">
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
            <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TypingIndicator;
