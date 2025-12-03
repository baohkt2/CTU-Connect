import { BaseEntity } from './common';

// ============================
// User Model
// ============================
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

// ============================
// User Profile (Extended User Info)
// ============================
export interface UserProfile extends User {
  avatarUrl?: string;
  facultyId?: string;
  facultyName?: string;
  majorId?: string;
  majorName?: string;
  batchId?: string;
  batchName?: string;
  collegeId?: string;
  collegeName?: string;
  genderId?: string;
  phoneNumber?: string;
  dateOfBirth?: string;
  address?: string;
  interests?: string[];
  socialLinks?: {
    facebook?: string;
    twitter?: string;
    linkedin?: string;
    instagram?: string;
  };
  privacySettings?: {
    profileVisibility: 'PUBLIC' | 'FRIENDS' | 'PRIVATE';
    contactInfoVisibility: 'PUBLIC' | 'FRIENDS' | 'PRIVATE';
  };
}

// ============================
// Lecturer-specific types
// ============================
export interface LecturerPosition {
  id: string;
  name: string;
}

export interface LecturerFaculty {
  id: string;
  name: string;
}

export interface LecturerCollege {
  id: string;
  name: string;
}

export interface LecturerDegree {
  id: string;
  name: string;
}

export interface LecturerAcademic {
  id: string;
  name: string;
}

export interface LecturerProfile extends User {
  staffCode?: string;
  position?: LecturerPosition;
  faculty?: LecturerFaculty;
  college?: LecturerCollege;
  degree?: LecturerDegree;
  academic?: LecturerAcademic;
}

// ============================
// Create / Update DTOs
// ============================
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

// ============================
// Update Profile Request (More comprehensive than UpdateUserRequest)
// ============================
export interface UpdateProfileRequest {
  fullName?: string;
  bio?: string;
  facultyId?: string;
  majorId?: string;
  batchId?: string;
  collegeId?: string;
  genderId?: string;
  phoneNumber?: string;
  dateOfBirth?: string;
  address?: string;
  interests?: string[];
  socialLinks?: {
    facebook?: string;
    twitter?: string;
    linkedin?: string;
    instagram?: string;
  };
  privacySettings?: {
    profileVisibility: 'PUBLIC' | 'FRIENDS' | 'PRIVATE';
    contactInfoVisibility: 'PUBLIC' | 'FRIENDS' | 'PRIVATE';
  };
}

// ============================
// Relationship
// ============================
export interface UserRelationship {
  id: string;
  followerId: string;
  followingId: string;
  createdAt: string;
}

// ============================
// Stats Object
// ============================
export interface UserStats {
  postsCount: number;
  followersCount: number;
  followingCount: number;
  friendsCount: number;
}
