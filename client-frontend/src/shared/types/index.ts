// Export all shared types
export * from './common';
export * from './user';
export * from './post';
export * from './chat';
export * from './auth';

// Common type definitions that might be missing
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  errors?: string[];
}

export interface PaginatedResponse<T = any> {
  content: T[];
  totalElements: number;
  totalPages: number;
  size: number;
  number: number;
  first: boolean;
  last: boolean;
}

// Export specific types that services are looking for
export type {
  User,
  UserProfile,
  UpdateProfileRequest
} from './user';

export type {
  Post,
  Comment,
  CreatePostRequest,
  UpdatePostRequest,
  CreateCommentRequest,
  UpdateCommentRequest,
  PostVisibility
} from './post';

export type {
  ChatRoom,
  ChatMessage
} from './chat';
