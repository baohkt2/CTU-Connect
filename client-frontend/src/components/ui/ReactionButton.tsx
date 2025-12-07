/* eslint-disable @typescript-eslint/no-unused-vars */
'use client';

import React, { useState, useRef, useEffect } from 'react';
import { ThumbsUp } from 'lucide-react';
import { ReactionPicker, REACTIONS } from './ReactionPicker';

interface ReactionButtonProps {
  onReactionClick: (reactionId: string) => void;
  onReactionRemove: () => void;
  currentReaction?: string | null;
  reactionCounts?: { [key: string]: number };
  disabled?: boolean;
  size?: 'sm' | 'md' | 'lg';
  showPicker?: boolean;
}

export const ReactionButton: React.FC<ReactionButtonProps> = ({
                                                                onReactionClick,
                                                                onReactionRemove,
                                                                currentReaction,
                                                                reactionCounts = {},
                                                                disabled = false,
                                                                size = 'md',
                                                                showPicker = true
                                                               }) => {
  const [showReactionPicker, setShowReactionPicker] = useState(false);
  const [countAnim, setCountAnim] = useState(false);

  const pickerRef = useRef<HTMLDivElement>(null);

  const currentReactionData = REACTIONS.find(r => r.id === currentReaction);
  const totalReactions = Object.values(reactionCounts).reduce((sum, count) => sum + count, 0);

  // Đóng picker nếu click ra ngoài
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (pickerRef.current && !pickerRef.current.contains(event.target as Node)) {
        setShowReactionPicker(false);
      }
    };

    if (showReactionPicker) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showReactionPicker]);

  // Animation khi số lượng thay đổi
  useEffect(() => {
    if (totalReactions > 0) {
      setCountAnim(true);
      const timer = setTimeout(() => setCountAnim(false), 300);
      return () => clearTimeout(timer);
    }
  }, [totalReactions]);

  // Hover to show picker
  const handleMouseEnter = () => {
    if (!showPicker) return;
    setShowReactionPicker(true);
  };

  const handleMouseLeave = () => {
    // Delay hiding picker to allow moving mouse to it
    setTimeout(() => {
      setShowReactionPicker(false);
    }, 200);
  };

  // Click handler for button
  const handleButtonClick = () => {
    if (currentReaction) {
      onReactionRemove();
    } else {
      onReactionClick('LIKE');
    }
  };

  const handleReactionSelect = (reactionId: string) => {
    onReactionClick(reactionId);
    setShowReactionPicker(false);
  };

  const buttonSizes = {
    sm: 'px-2 py-1 text-xs',
    md: 'px-3 py-2 text-sm',
    lg: 'px-4 py-2 text-base'
  };

  const iconSizes = {
    sm: 'h-3 w-3',
    md: 'h-4 w-4',
    lg: 'h-5 w-5'
  };

  return (
      <div className="relative" ref={pickerRef}>
        <button
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
            onClick={handleButtonClick}
            disabled={disabled}
            className={`
          flex items-center space-x-2 rounded-lg font-medium transition-all duration-200
          ${buttonSizes[size]}
          ${currentReaction
                ? `${currentReactionData?.color} bg-blue-50 hover:bg-blue-100`
                : 'text-gray-600 hover:bg-gray-50 hover:text-blue-600'
            }
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
        `}
        >
          {currentReactionData ? (
              <>
                <span className={iconSizes[size]}>{currentReactionData.emoji}</span>
                <span className="vietnamese-text">{currentReactionData.name}</span>
              </>
          ) : (
              <>
                <ThumbsUp className={iconSizes[size]} />
                <span className="vietnamese-text">Thích</span>
              </>
          )}

          {totalReactions > 0 && (
              <span className={`ml-1 text-xs bg-white rounded-full px-1 min-w-[20px] text-center transition-transform duration-300 ${countAnim ? 'scale-125' : 'scale-100'}`}>
            {totalReactions}
          </span>
          )}
        </button>

        {/* Picker với animation xuất hiện */}
        {showPicker && showReactionPicker && (
            <div 
              className="absolute bottom-full left-0 mb-2 z-50 animate-fadeScaleIn"
              onMouseEnter={() => setShowReactionPicker(true)}
              onMouseLeave={() => setShowReactionPicker(false)}
            >
              <ReactionPicker
                  onReactionClick={handleReactionSelect}
                  currentReaction={currentReaction}
                  size={size}
              />
            </div>
        )}

        {/* Hiển thị emoji reaction tổng kết */}
        {totalReactions > 0 && (
            <div className="absolute -top-2 -right-2 bg-white rounded-full shadow-sm border border-gray-200 px-1 min-w-[20px] text-center">
              <div className="flex items-center space-x-1">
                {Object.entries(reactionCounts)
                    .filter(([_, count]) => count > 0)
                    .slice(0, 3)
                    .map(([reactionId, count]) => {
                      const reaction = REACTIONS.find(r => r.id === reactionId);
                      return reaction ? (
                          <span key={reactionId} className="text-xs" title={`${count} ${reaction.name}`}>
                    {reaction.emoji}
                  </span>
                      ) : null;
                    })}
                <span className="text-xs text-gray-600 font-medium">
              {totalReactions}
            </span>
              </div>
            </div>
        )}
      </div>
  );
};
