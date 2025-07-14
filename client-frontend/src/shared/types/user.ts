import { BaseEntity } from './common';

// User entity
export interface User extends BaseEntity {
  email: string;
  username: string;
  fullName: string;
  avatar?: string;
  bio?: string;
  faculty?: string;
  yearOfStudy?: number;
  isVerified: boolean;
  isOnline: boolean;
  followersCount?: number;
  followingCount?: number;
  postsCount?: number;
}

// User creation/update types
export interface CreateUserRequest {
  email: string;
  username: string;
  fullName: string;
  password: string;
}

export interface UpdateUserRequest {
  fullName?: string;
  bio?: string;
  faculty?: string;
  yearOfStudy?: number;
}

// User relationship types
export interface UserRelationship {
  id: string;
  followerId: string;
  followingId: string;
  createdAt: string;
}

// User stats
export interface UserStats {
  postsCount: number;
  followersCount: number;
  followingCount: number;
  friendsCount: number;
}
