'use client';

import React from 'react';
import { ThumbsUp, Lightbulb, CheckCircle, BookOpen, HelpCircle } from 'lucide-react';

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
    id: 'INSIGHTFUL',
    name: 'Sáng Suốt',
    emoji: '💡',
    icon: <Lightbulb className="w-6 h-6" />,
    color: 'text-yellow-600',
    hoverColor: 'hover:bg-yellow-50'
  },
  {
    id: 'RELEVANT',
    name: 'Phù Hợp',
    emoji: '✔️',
    icon: <CheckCircle className="w-6 h-6" />,
    color: 'text-green-600',
    hoverColor: 'hover:bg-green-50'
  },
  {
    id: 'USEFUL_SOURCE',
    name: 'Nguồn Hữu Ích',
    emoji: '📚',
    icon: <BookOpen className="w-6 h-6" />,
    color: 'text-purple-600',
    hoverColor: 'hover:bg-purple-50'
  },
  {
    id: 'QUESTION',
    name: 'Cần Thảo Luận',
    emoji: '❓',
    icon: <HelpCircle className="w-6 h-6" />,
    color: 'text-orange-600',
    hoverColor: 'hover:bg-orange-50'
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
            rounded-full transition-all duration-200 transform hover:scale-110
            ${currentReaction === reaction.id ? 'bg-blue-50 ring-2 ring-blue-300' : ''}
          `}
          title={reaction.name}
        >
          <span className={`${emojiSizes[size]} block`}>
            {reaction.emoji}
          </span>
        </button>
      ))}
    </div>
  );
};
