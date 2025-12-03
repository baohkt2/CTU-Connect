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
 * Sanitize HTML content (basic sanitization)
 */
export const sanitizeHtml = (html: string): string => {
  // Basic sanitization - remove script tags and dangerous attributes
  return html
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/on\w+="[^"]*"/gi, '')
    .replace(/javascript:/gi, '');
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
