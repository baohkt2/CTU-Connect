// Types for API responses and entities

export interface User {
  id: string;
  email: string;
  username: string;
  fullName?: string;
  name?: string; // Optional for compatibility with older data
  bio?: string;
  studentId?: string;
  yearOfStudy?: number;
  isVerified: boolean;
  isOnline?: boolean;
  createdAt?: string;
  updatedAt?: string;
  // Enhanced profile fields
  role: 'STUDENT' | 'LECTURER' | 'ADMIN' | 'USER';
  isProfileCompleted?: boolean;
  avatarUrl?: string | null;
  backgroundUrl?: string | null;

  // Working fields
  college?: CollegeInfo;
  faculty?: FacultyInfo;

  // Student specific fields
  major?: MajorInfo;
  batch?: BatchInfo;
  gender?: GenderInfo;

  // Faculty specific fields
  staffCode?: string;
  position?: PositionInfo;
  academic?: AcademicInfo;
  degree?: DegreeInfo;
}

// Updated Post interface to match backend PostResponse
export interface Post {
  id: string;
  title?: string;
  content: string;
  authorId: string;
  authorName?: string;
  authorAvatar?: string;
  images?: string[]; // Array of image URLs
  videos?: string[]; // Array of video URLs - ADDED for video support
  tags?: string[];
  category?: string;
  visibility?: string;
  stats: PostStats;
  createdAt: string;
  updatedAt: string;
}

export interface PostStats {
  views: number;
  likes: number;
  shares: number;
  comments: number;
  bookmarks: number;
  reactions?: { [key: string]: number };
}

// Post creation request to match backend PostRequest
export interface CreatePostRequest {
  title?: string;
  content: string;
  tags?: string[];
  category?: string;
  visibility?: 'PUBLIC' | 'FRIENDS' | 'PRIVATE';
}

// Post update request
export interface UpdatePostRequest {
  title?: string;
  content?: string;
  tags?: string[];
  category?: string;
  visibility?: 'PUBLIC' | 'FRIENDS' | 'PRIVATE';
}

export interface Comment {
  id: string;
  content: string;
  postId: string;
  author: Author;
  authorName?: string ;
  authorAvatar?: string;
  stats?: CommentStats;
  createdAt: string;
  updatedAt: string;
}

export interface Author {
  id: string;
  name: string;
  avatar: string | null;
}


export interface CommentStats {
  likes: number;
  replies: number;
}

export interface CreateCommentRequest {
  content: string;
  parentId?: string; // For reply comments
}

export interface Interaction {
  id: string;
  postId: string;
  authorId: string;
  type: InteractionType;
  reactionType?: ReactionType;
  createdAt: string;
}

export enum InteractionType {
  LIKE = 'LIKE',
  SHARE = 'SHARE',
  BOOKMARK = 'BOOKMARK',
  VIEW = 'VIEW'
}

export enum ReactionType {
  LIKE = 'LIKE',
  LOVE = 'LOVE',
  HAHA = 'HAHA',
  WOW = 'WOW',
  SAD = 'SAD',
  ANGRY = 'ANGRY',
  BOOKMARK = 'BOOKMARK'
}

export interface CreateInteractionRequest {
  type: InteractionType;
  reactionType?: ReactionType;
}

export interface LoginRequest {
  email?: string;
  username?: string;
  password: string;
  recaptchaToken?: string;
}

export interface RegisterRequest {
  email: string;
  username: string;
  password: string;
  recaptchaToken?: string;
}

export interface AuthResponse {
  token: string;
  user: User;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
  errorCode?: string;
}

export interface ApiError {
  success: false;
  message: string;
  errorCode?: string;
  errors?: { [key: string]: string };
}

export interface PaginatedResponse<T> {
  content: T[];
  totalElements: number;
  totalPages: number;
  size: number;
  number: number;
  first: boolean;
  last: boolean;
}

export interface MajorInfo {
  name: string; // Sử dụng name làm identifier chính
  code?: string; // Code tùy chọn
  faculty?: FacultyInfo;
}

export interface FacultyInfo {
  name: string; // Sử dụng name làm identifier chính
  code?: string; // Code tùy chọn
  college?: CollegeInfo;
}

export interface CollegeInfo {
  name: string; // Sử dụng name làm identifier chính
  code?: string; // Code tùy chọn
}

export interface BatchInfo {
  year: string;
}

export interface GenderInfo {
  code: string;
  name: string;
}

export interface PositionInfo {
    code: string;
    name: string;
}

export interface AcademicInfo {
  code: string;
  name: string;
}

export interface DegreeInfo {
  code: string;
  name: string;
}

export interface StudentProfileUpdateRequest {
  fullName: string;
  bio?: string;
  studentId: string;
  collegeCode: string; // Đổi từ collegeCode sang collegeName
  facultyCode: string; // Đổi từ workingFacultyCode sang facultyName
  majorCode: string; // Đổi từ majorCode sang majorName
  batchYear: string;
  genderCode: string;
  avatarUrl?: string;
  backgroundUrl?: string;
}

export interface LecturerProfileUpdateRequest {
  fullName: string;
  bio?: string;
  staffCode: string;
  positionCode: string;
  academicCode?: string;
  degreeCode?: string;
  facultyCode: string; // Đổi từ workingFacultyCode sang workingFacultyName
  genderCode: string;
  avatarUrl?: string;
  backgroundUrl?: string;
}

export interface ProfileCompletionStatus {
  isCompleted: boolean;
  missingFields?: string[];
}

export interface CollegeWithHierarchyInfo {
  name: string; // Sử dụng name làm identifier chính
  code?: string; // Code tùy chọn
  faculties: FacultyWithMajorsInfo[];
}

export interface FacultyWithMajorsInfo {
  name: string; // Sử dụng name làm identifier chính
  code?: string; // Code tùy chọn
  collegeName: string; // Tên college
  majors: MajorInfo[];
}

// Updated interface for hierarchical categories
export interface HierarchicalCategories {
  degrees?: DegreeInfo[];
  academics?: AcademicInfo[];
  positions?: PositionInfo[];
  colleges: CollegeWithHierarchyInfo[];
  batches?: BatchInfo[];
  genders: GenderInfo[];
}
