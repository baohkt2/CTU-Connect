// Types for API responses and entities

export interface User {
  id: string;
  email: string;
  username: string;
  fullName: string;
  avatar?: string;
  bio?: string;
  studentId?: string;
  yearOfStudy?: number;
  isVerified: boolean;
  isOnline: boolean;
  createdAt: string;
  updatedAt: string;
  // Enhanced profile fields
  role: 'STUDENT' | 'FACULTY' | 'ADMIN' | 'USER';
  isProfileCompleted: boolean;
  avatarUrl?: string;
  backgroundUrl?: string;

  // Student specific fields
  major?: MajorInfo;
  batch?: BatchInfo;
  gender?: GenderInfo;

  // Faculty specific fields
  staffCode?: string;
  position?: PositionInfo;
  academic?: AcademicInfo;
  degree?: DegreeInfo;
  faculty?: FacultyInfo;
}

export interface Post {
  id: string;
  content: string;
  images?: string[];
  authorId: string;
  author: User;
  likes: number;
  comments: number;
  isLiked: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface Comment {
  id: string;
  content: string;
  postId: string;
  authorId: string;
  author: User;
  likes: number;
  isLiked: boolean;
  createdAt: string;
  updatedAt: string;
}

export interface ChatMessage {
  id: string;
  content: string;
  senderId: string;
  receiverId: string;
  sender: User;
  receiver: User;
  isRead: boolean;
  createdAt: string;
}

export interface ChatRoom {
  id: string;
  participants: User[];
  lastMessage?: ChatMessage;
  unreadCount: number;
  createdAt: string;
  updatedAt: string;
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
  year: number;
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
  majorName: string; // Đổi từ majorCode sang majorName
  batchYear: number;
  genderCode: string;
  avatarUrl?: string;
  backgroundUrl?: string;
}

export interface FacultyProfileUpdateRequest {
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
  degrees: DegreeInfo[];
  academics: AcademicInfo[];
  positions: PositionInfo[];
  colleges: CollegeWithHierarchyInfo[];
  batches: BatchInfo[];
  genders: GenderInfo[];

}
