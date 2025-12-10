// API Endpoints Constants for Client-Frontend
export const API_ENDPOINTS = {
  // Base URLs for services
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080',
  
  // Authentication endpoints
  AUTH: {
    BASE: '/api',
    LOGIN: '/api/login',
    REGISTER: '/api/register',
    REFRESH: '/api/refresh',
    LOGOUT: '/api/logout',
    VERIFY_EMAIL: '/api/verify-email',
    FORGOT_PASSWORD: '/api/forgot-password',
    RESET_PASSWORD: '/api/reset-password',
    PROFILE: '/api/profile'
  },

  // User management endpoints
  USERS: {
    BASE: '/api/users',
    PROFILE: '/api/users/me/profile',
    BY_ID: '/api/users/:id',
    UPDATE_PROFILE: '/api/users/me/profile',
    SEARCH: '/api/users/search',
    FRIENDS: '/api/users/:id/friends',
    FRIEND_SUGGESTIONS: '/api/users/friend-suggestions',
    SEND_FRIEND_REQUEST: '/api/users/:id/friend-request',
    ACCEPT_FRIEND_REQUEST: '/api/users/:id/accept-friend',
    MUTUAL_FRIENDS: '/api/users/:id/mutual-friends-count',
    TIMELINE: '/api/users/:id/timeline',
    ACTIVITIES: '/api/users/:id/activities'
  },

  // Post management endpoints
  POSTS: {
    BASE: '/api/posts',
    BY_ID: '/api/posts/:id',
    BY_USER: '/api/posts/user/:userId',
    FEED: '/api/posts/feed',
    TRENDING: '/api/posts/trending',
    TIMELINE: '/api/posts/timeline/:userId',
    SEARCH: '/api/posts/search',
    LIKE: '/api/posts/:id/interact',
    COMMENTS: '/api/posts/:id/comments',
    COMMENT_LIKE: '/api/posts/comments/:id/like',
    SCHEDULE: '/api/posts/schedule',
    ANALYTICS: '/api/posts/:id/analytics'
  },

  // Recommendation Service endpoints (AI-powered)
  RECOMMENDATIONS: {
    BASE: '/api/recommend',
    POSTS: '/api/recommend/posts',
    FEED: '/api/recommendation/feed',
    FEEDBACK: '/api/recommend/feedback',
    INTERACTION: '/api/recommendation/interaction',
    CACHE_INVALIDATE: '/api/recommend/cache/:userId',
    HEALTH: '/api/recommend/health'
  },

  // Chat endpoints
  CHAT: {
    BASE: '/api/chats',
    ROOMS: '/api/chats/conversations',
    MESSAGES: '/api/chats/messages',
    SEND_MESSAGE: '/api/chats/messages',
    MARK_READ: '/api/chats/messages/:id/read',
    ROOM_READ: '/api/chats/conversations/:id/read',
    UNREAD_COUNT: '/api/chats/messages/unread-count',
    ONLINE_USERS: '/api/chats/users/online',
    TYPING_START: '/api/chats/conversations/:roomId/typing/start',
    TYPING_STOP: '/api/chats/conversations/:roomId/typing/stop',
    USER_PRESENCE: '/api/chats/users/presence'
  },

  // Media endpoints
  MEDIA: {
    BASE: '/api/media',
    UPLOAD: '/api/media/upload',
    BY_ID: '/api/media/:id'
  },

  // Notification endpoints
  NOTIFICATIONS: {
    BASE: '/api/notifications',
    UNREAD_COUNT: '/api/notifications/unread-count',
    MARK_READ: '/api/notifications/:id/read',
    MARK_ALL_READ: '/api/notifications/mark-all-read'
  },

  // Analytics endpoints
  ANALYTICS: {
    BASE: '/api/analytics',
    USER_STATS: '/api/analytics/users/:id/stats',
    POST_STATS: '/api/analytics/posts/:id/stats'
  }
};

// API Response types
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  errors?: string[];
}

export interface PaginatedResponse<T = any> {
  content: T[];
  totalElements: number;
  totalPages: number;
  size: number;
  number: number;
  first: boolean;
  last: boolean;
}
