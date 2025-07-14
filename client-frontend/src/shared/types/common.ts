// Base types
export interface BaseEntity {
  id: string;
  createdAt: string;
  updatedAt: string;
}

// API Response types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
  errors?: Record<string, string[]>;
}

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

// Error types
export interface ApiError {
  status: number;
  message: string;
  code?: string;
  details?: any;
}

// Form types
export interface FormFieldError {
  field: string;
  message: string;
}

// Upload types
export interface FileUploadResponse {
  url: string;
  filename: string;
  size: number;
  type: string;
}
