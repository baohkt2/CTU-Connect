import { BaseEntity } from './common';
import { User } from './user';

// Post entity
export interface Post extends BaseEntity {
  content: string;
  images?: string[];
  authorId: string;
  author: User;
  likesCount: number;
  commentsCount: number;
  sharesCount: number;
  isLiked: boolean;
  isBookmarked: boolean;
  visibility: PostVisibility;
  tags?: string[];
}

// Post creation/update types
export interface CreatePostRequest {
  content: string;
  images?: File[];
  visibility?: PostVisibility;
  tags?: string[];
}

export interface UpdatePostRequest {
  content?: string;
  visibility?: PostVisibility;
  tags?: string[];
}

// Comment entity
export interface Comment extends BaseEntity {
  content: string;
  postId: string;
  authorId: string;
  author: User;
  parentId?: string; // For nested comments
  likesCount: number;
  repliesCount: number;
  isLiked: boolean;
  replies?: Comment[];
}

// Comment creation/update types
export interface CreateCommentRequest {
  content: string;
  postId: string;
  parentId?: string;
}

export interface UpdateCommentRequest {
  content: string;
}

// Post visibility
export enum PostVisibility {
  PUBLIC = 'PUBLIC',
  FRIENDS = 'FRIENDS',
  PRIVATE = 'PRIVATE',
}

// Post interaction types
export interface PostLike extends BaseEntity {
  postId: string;
  userId: string;
  user: User;
}

export interface PostShare extends BaseEntity {
  postId: string;
  userId: string;
  user: User;
  sharedContent?: string;
}
