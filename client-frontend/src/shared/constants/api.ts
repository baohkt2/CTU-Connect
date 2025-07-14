// API Configuration
export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8090',
  TIMEOUT: 10000,
  RETRY_ATTEMPTS: 3,
} as const;

// Endpoints
export const API_ENDPOINTS = {
  AUTH: {
    LOGIN: '/auth/login',
    REGISTER: '/auth/register',
    LOGOUT: '/auth/logout',
    REFRESH: '/auth/refresh',
    ME: '/auth/me',
    VERIFY_EMAIL: '/auth/verify-email',
    FORGOT_PASSWORD: '/auth/forgot-password',
    RESET_PASSWORD: '/auth/reset-password',
  },
  USERS: {
    PROFILE: '/users',
    SEARCH: '/users/search',
    FOLLOW: '/users/:id/follow',
    FOLLOWERS: '/users/:id/followers',
    FOLLOWING: '/users/:id/following',
    FRIENDS: '/users/:id/friends',
    AVATAR: '/users/avatar',
  },
  POSTS: {
    BASE: '/posts',
    BY_USER: '/posts/user/:id',
    LIKE: '/posts/:id/like',
    COMMENTS: '/posts/:id/comments',
    COMMENT_LIKE: '/comments/:id/like',
  },
  CHAT: {
    ROOMS: '/chat/rooms',
    MESSAGES: '/chat/rooms/:roomId/messages',
    UNREAD_COUNT: '/chat/unread-count',
    MARK_READ: '/chat/messages/:id/read',
    ROOM_READ: '/chat/rooms/:id/read',
  },
} as const;

// Socket Configuration
export const SOCKET_CONFIG = {
  URL: process.env.NEXT_PUBLIC_SOCKET_URL || 'ws://localhost:8090/ws',
  RECONNECTION_ATTEMPTS: 5,
  RECONNECTION_DELAY: 1000,
} as const;
