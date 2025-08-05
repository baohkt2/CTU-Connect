'use client';

import React from 'react';
import { Heart, ThumbsUp, Laugh, Frown, Angry, Zap } from 'lucide-react';

export interface ReactionType {
  id: string;
  name: string;
  emoji: string;
  icon: React.ReactNode;
  color: string;
  hoverColor: string;
}

export const REACTIONS: ReactionType[] = [
  {
    id: 'LIKE',
    name: 'Thích',
    emoji: '👍',
    icon: <ThumbsUp className="w-6 h-6" />,
    color: 'text-blue-600',
    hoverColor: 'hover:bg-blue-50'
  },
  {
    id: 'LOVE',
    name: 'Yêu thích',
    emoji: '❤️',
    icon: <Heart className="w-6 h-6" />,
    color: 'text-red-600',
    hoverColor: 'hover:bg-red-50'
  },
  {
    id: 'HAHA',
    name: 'Haha',
    emoji: '😂',
    icon: <Laugh className="w-6 h-6" />,
    color: 'text-yellow-600',
    hoverColor: 'hover:bg-yellow-50'
  },
  {
    id: 'WOW',
    name: 'Wow',
    emoji: '😮',
    icon: <Zap className="w-6 h-6" />,
    color: 'text-orange-600',
    hoverColor: 'hover:bg-orange-50'
  },
  {
    id: 'SAD',
    name: 'Buồn',
    emoji: '😢',
    icon: <Frown className="w-6 h-6" />,
    color: 'text-yellow-700',
    hoverColor: 'hover:bg-yellow-50'
  },
  {
    id: 'ANGRY',
    name: 'Phẫn nộ',
    emoji: '😠',
    icon: <Angry className="w-6 h-6" />,
    color: 'text-red-700',
    hoverColor: 'hover:bg-red-50'
  }
];

interface ReactionPickerProps {
  onReactionClick: (reactionId: string) => void;
  currentReaction?: string | null;
  size?: 'sm' | 'md' | 'lg';
}

export const ReactionPicker: React.FC<ReactionPickerProps> = ({
                                                                onReactionClick,
                                                                currentReaction,
                                                                size = 'md'
                                                              }) => {
  const sizeClasses = {
    sm: 'p-1',
    md: 'p-2',
    lg: 'p-3'
  };

  const emojiSizes = {
    sm: 'text-lg',
    md: 'text-xl',
    lg: 'text-2xl'
  };

  return (
      <div className="bg-white rounded-full shadow-lg border border-gray-200 flex items-center space-x-1 px-2 py-1 animate-fadeIn">
        {REACTIONS.map((reaction) => (
            <button
                key={reaction.id}
                onClick={() => onReactionClick(reaction.id)}
                className={`
            ${sizeClasses[size]} 
            ${reaction.hoverColor} 
            rounded-full transition-transform duration-300 ease-in-out transform hover:scale-125 hover:-translate-y-1
            ${currentReaction === reaction.id ? 'bg-blue-50 ring-2 ring-blue-300' : ''}
          `}
                title={reaction.name}
            >
          <span className={`${emojiSizes[size]} block animate-popIn`}>
            {reaction.emoji}
          </span>
            </button>
        ))}
      </div>
  );
};
