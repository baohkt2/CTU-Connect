'use client';

import React from 'react';
import { Edit3 } from 'lucide-react';

interface EditIndicatorProps {
  isEdited: boolean;
  className?: string;
}

export const EditIndicator: React.FC<EditIndicatorProps> = ({
 isEdited = false,
  className = ''
}) => {
  // Check if post was edited (updatedAt is different from createdAt)

  if (!isEdited) return null;

  return (
    <div className={`flex items-center space-x-1 text-xs text-gray-500 ${className}`}>
      <Edit3 className="h-3 w-3" />
      <span className="vietnamese-text">Đã chỉnh sửa</span>
    </div>
  );
};

export default EditIndicator;
