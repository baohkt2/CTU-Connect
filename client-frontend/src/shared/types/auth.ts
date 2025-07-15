import { User } from './user';
import { ApiResponse } from './common';

// ==========================
// Authentication Request DTOs
// ==========================
export interface LoginRequest {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface RegisterRequest {
  email: string;
  username: string;
  fullName: string;
  password: string;
  confirmPassword: string;
}

export interface ForgotPasswordRequest {
  email: string;
}

export interface ResetPasswordRequest {
  token: string;
  newPassword: string;
  confirmPassword: string;
}

export interface VerifyEmailRequest {
  token: string;
}

// ==========================
// Authentication Response DTOs
// ==========================
export interface AuthResponse {
  token: string;
  refreshToken: string;
  user: User;
  expiresIn: number;
}

export interface TokenResponse {
  token: string;
  refreshToken: string;
  expiresIn: number;
}

// JWT payload (optional)
export interface TokenPayload {
  sub: string;
  email: string;
  exp: number;
  iat: number;
  roles?: string[];
}

// ==========================
// Re-exports
// ==========================
export type { User, ApiResponse };
