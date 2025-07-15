# Services Migration Guide

## 📋 Tổng quan

Dự án đã được tái cấu trúc để tách biệt services theo domain và chỉ giữ lại cross-domain services trong shared/services.

## 🔄 Cấu trúc mới

### Feature-based Services
```
features/
├── auth/services/           # Authentication domain
│   └── auth.service.ts
├── users/services/          # User management domain  
│   └── user.service.ts
├── posts/services/          # Posts & comments domain
│   └── post.service.ts
└── chat/services/           # Chat & messaging domain
    └── chat.service.ts
```

### Cross-domain Services
```
shared/services/
├── upload.service.ts        # File uploads
├── notification.service.ts  # Push notifications
├── analytics.service.ts     # User tracking
└── settings.service.ts      # App configuration
```

## 📝 Migration Instructions

### 1. Update Imports

#### Before (Old Structure):
```typescript
import { authService } from '@/shared/services';
import { userService } from '@/shared/services';
import { postService } from '@/shared/services';
import { chatService } from '@/shared/services';
```

#### After (New Structure):
```typescript
import { authService } from '@/features/auth';
import { userService } from '@/features/users';
import { postService } from '@/features/posts';
import { chatService } from '@/features/chat';

// Cross-domain services remain in shared
import { uploadService, notificationService } from '@/shared/services';
```

### 2. Feature-specific Usage

#### Auth Service:
```typescript
// Import from auth feature
import { authService } from '@/features/auth';

// Usage remains the same
const login = async () => {
  const result = await authService.login(credentials);
};
```

#### User Service:
```typescript
// Import from users feature
import { userService } from '@/features/users';

// Usage remains the same
const profile = await userService.getUserProfile(userId);
```

#### Post Service:
```typescript
// Import from posts feature
import { postService } from '@/features/posts';

// Usage remains the same
const posts = await postService.getPosts();
```

#### Chat Service:
```typescript
// Import from chat feature
import { chatService } from '@/features/chat';

// Usage remains the same
const rooms = await chatService.getChatRooms();
```

### 3. Cross-domain Services

#### Upload Service:
```typescript
import { uploadService } from '@/shared/services';

const uploadAvatar = async (file: File) => {
  return uploadService.uploadFile(file, '/users/avatar');
};
```

#### Notification Service:
```typescript
import { notificationService } from '@/shared/services';

const notifications = await notificationService.getNotifications();
```

#### Analytics Service:
```typescript
import { analyticsService } from '@/shared/services';

// Track events
analyticsService.trackEvent('post_created', { postId: '123' });
analyticsService.trackPageView('home');
```

#### Settings Service:
```typescript
import { settingsService } from '@/shared/services';

// App settings
const settings = await settingsService.getUserSettings();

// Local preferences
settingsService.setTheme('dark');
settingsService.setLanguage('vi');
```

## 🏗️ Architecture Benefits

### 1. Domain Separation
- Each feature manages its own services
- Clear boundaries between domains
- Easier to maintain and extend

### 2. Scalability
- New features can add their own services
- Cross-domain functionality centralized
- Independent development possible

### 3. Code Organization
- Services closer to their usage
- Reduced coupling between features
- Better discoverability

## 🔧 Implementation Steps

### Step 1: Update Existing Components
```bash
# Find all service imports
grep -r "from '@/shared/services'" src/

# Replace with feature-specific imports
# Auth: @/features/auth
# Users: @/features/users  
# Posts: @/features/posts
# Chat: @/features/chat
```

### Step 2: Update Hooks
```typescript
// Before
import { authService } from '@/shared/services';

// After
import { authService } from '@/features/auth';
```

### Step 3: Update Tests
Update all test files to use new import paths.

### Step 4: Verify Build
```bash
npm run build
npm run type-check
```

## 📋 Checklist

- [ ] Update all component imports
- [ ] Update all hook imports
- [ ] Update all service imports in pages
- [ ] Update test files
- [ ] Verify TypeScript compilation
- [ ] Test application functionality
- [ ] Update documentation

## ⚠️ Breaking Changes

### Old shared/services exports removed:
- `authService` → Use `@/features/auth`
- `userService` → Use `@/features/users`
- `postService` → Use `@/features/posts`
- `chatService` → Use `@/features/chat`

### New shared/services exports:
- `uploadService` - File upload functionality
- `notificationService` - Push notifications
- `analyticsService` - User tracking
- `settingsService` - App configuration

## 🚀 Next Steps

1. **Complete Migration**: Update all existing imports
2. **Add Feature Services**: Create additional services within features as needed
3. **Implement Cross-domain**: Use new shared services for cross-feature functionality
4. **Testing**: Ensure all functionality works with new structure

## 📞 Support

If you encounter issues during migration:
1. Check import paths are correct
2. Verify feature exports in index.ts files
3. Ensure TypeScript compilation passes
4. Test functionality manually
