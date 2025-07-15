import { VALIDATION_RULES } from '@/shared/constants';

/**
 * Validate email format
 */
export const validateEmail = (email: string): boolean => {
  return VALIDATION_RULES.EMAIL_REGEX.test(email);
};

/**
 * Validate password strength
 */
export const validatePassword = (password: string): {
  isValid: boolean;
  errors: string[];
} => {
  const errors: string[] = [];

  if (password.length < VALIDATION_RULES.PASSWORD_MIN_LENGTH) {
    errors.push(`Mật khẩu phải có ít nhất ${VALIDATION_RULES.PASSWORD_MIN_LENGTH} ký tự`);
  }

  if (!/[A-Z]/.test(password)) {
    errors.push('Mật khẩu phải có ít nhất 1 chữ hoa');
  }

  if (!/[a-z]/.test(password)) {
    errors.push('Mật khẩu phải có ít nhất 1 chữ thường');
  }

  if (!/\d/.test(password)) {
    errors.push('Mật khẩu phải có ít nhất 1 chữ số');
  }

  return {
    isValid: errors.length === 0,
    errors,
  };
};

/**
 * Validate username format
 */
export const validateUsername = (username: string): {
  isValid: boolean;
  errors: string[];
} => {
  const errors: string[] = [];

  if (username.length < VALIDATION_RULES.USERNAME_MIN_LENGTH) {
    errors.push(`Tên đăng nhập phải có ít nhất ${VALIDATION_RULES.USERNAME_MIN_LENGTH} ký tự`);
  }

  if (!/^[a-zA-Z0-9_]+$/.test(username)) {
    errors.push('Tên đăng nhập chỉ được chứa chữ cái, số và dấu gạch dưới');
  }

  return {
    isValid: errors.length === 0,
    errors,
  };
};

/**
 * Validate file size
 */
export const validateFileSize = (file: File, maxSize: number): boolean => {
  return file.size <= maxSize;
};

/**
 * Validate file type
 */
export const validateFileType = (file: File, allowedTypes: string[]): boolean => {
  return allowedTypes.includes(file.type);
};

/**
 * Sanitize HTML content
 */
export const sanitizeHtml = (html: string): string => {
  const div = document.createElement('div');
  div.textContent = html;
  return div.innerHTML;
};

/**
 * Validate post content
 */
export const validatePostContent = (content: string): {
  isValid: boolean;
  errors: string[];
} => {
  const errors: string[] = [];

  if (!content.trim()) {
    errors.push('Nội dung bài đăng không được để trống');
  }

  if (content.length > VALIDATION_RULES.POST_MAX_LENGTH) {
    errors.push(`Nội dung bài đăng không được vượt quá ${VALIDATION_RULES.POST_MAX_LENGTH} ký tự`);
  }

  return {
    isValid: errors.length === 0,
    errors,
  };
};
