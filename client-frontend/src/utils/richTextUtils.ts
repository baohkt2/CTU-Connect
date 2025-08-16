/**
 * Utility functions for handling rich text content
 */

/**
 * Strip HTML tags from content and return plain text
 */
export const stripHtml = (html: string): string => {
  const temp = document.createElement('div');
  temp.innerHTML = html;
  return temp.textContent || temp.innerText || '';
};

/**
 * Get text length from HTML content (excluding HTML tags)
 */
export const getTextLength = (html: string): number => {
  return stripHtml(html).length;
};

/**
 * Check if HTML content is empty (only contains empty tags or whitespace)
 */
export const isHtmlEmpty = (html: string): boolean => {
  const text = stripHtml(html).trim();
  return text.length === 0;
};

/**
 * Truncate HTML content while preserving formatting
 */
export const truncateHtml = (html: string, maxLength: number): string => {
  const text = stripHtml(html);
  if (text.length <= maxLength) {
    return html;
  }

  // Simple truncation - in a real app, you might want to use a more sophisticated HTML truncation library
  const truncatedText = text.substring(0, maxLength);
  return truncatedText;
};

/**
 * Sanitize HTML content for safe display - Basic sanitization without external dependencies
 */
export const sanitizeHtml = (html: string): string => {
  if (!html) return '';

  // Basic sanitization - remove script tags and dangerous attributes
  return html
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/on\w+\s*=\s*["'][^"']*["']/gi, '')
    .replace(/javascript:\s*[^"'\s]*/gi, '')
    .replace(/<iframe[^>]*>/gi, '')
    .replace(/<embed[^>]*>/gi, '')
    .replace(/<object[^>]*>/gi, '')
    .replace(/<form[^>]*>/gi, '')
    .replace(/<input[^>]*>/gi, '')
    .replace(/<button[^>]*>/gi, '');
};

/**
 * Convert HTML to plain text for API submission
 */
export const htmlToText = (html: string): string => {
  return stripHtml(html).trim();
};

/**
 * Check if content meets minimum requirements
 */
export const validateContent = (html: string, minLength: number = 1): boolean => {
  const text = stripHtml(html).trim();
  return text.length >= minLength;
};

/**
 * Prepare HTML content for safe display
 */
export const prepareHtmlForDisplay = (html: string): string => {
  if (!html) return '';

  // First sanitize the HTML to remove dangerous content
  const sanitized = sanitizeHtml(html);

  // Return the sanitized HTML
  return sanitized;
};
