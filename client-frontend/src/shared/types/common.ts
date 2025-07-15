// ============================
// Base Entity
// ============================
export interface BaseEntity {
  id: string;
  createdAt: string; // ISO string
  updatedAt: string;
}

// ============================
// Generic API Response
// ============================
export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
  errors?: Record<string, string[]>; // key = field, value = list of error messages
}

// ============================
// Paginated Response
// ============================
export interface PaginatedResponse<T> {
  content: T[];
  totalElements: number;
  totalPages: number;
  size: number;
  number: number;
  first: boolean;
  last: boolean;
  empty: boolean;
}

// ============================
// API Error Object
// ============================
export interface ApiError {
  status: number;
  message: string;
  code?: string;
  details?: unknown;
}

// ============================
// Form Field Error (optional use in UI)
// ============================
export interface FormFieldError {
  field: string;
  message: string;
}

// ============================
// File Upload Response
// ============================
export interface FileUploadResponse {
  url: string;
  filename: string;
  size: number; // bytes
  type: string; // MIME type
}
