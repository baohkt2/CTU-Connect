# CTU Connect - Client Frontend

## 📋 Tổng quan

CTU Connect là một social network dành cho sinh viên CTU (Can Tho University), được xây dựng với Next.js 15, TypeScript, và Tailwind CSS theo chuẩn enterprise architecture.

## 🏗️ Cấu trúc dự án

```
src/
├── app/                    # Next.js App Router
│   ├── (auth)/
│   │   ├── login/
│   │   └── register/
│   ├── profile/[userId]/
│   ├── messages/
│   └── search/
├── features/               # Feature-based modules
│   ├── auth/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── types/
│   │   └── utils/
│   ├── posts/
│   ├── users/
│   ├── chat/
│   └── search/
├── shared/                 # Shared resources
│   ├── components/         # Reusable UI components
│   ├── config/            # Configuration files
│   ├── constants/         # Application constants
│   ├── hooks/             # Shared custom hooks
│   ├── services/          # API services
│   ├── types/             # TypeScript types
│   └── utils/             # Utility functions
├── components/            # Legacy components (to be migrated)
├── contexts/              # React contexts
├── hooks/                 # Legacy hooks (to be migrated)
├── lib/                   # Third-party library configs
├── services/              # Legacy services (to be migrated)
├── types/                 # Legacy types (to be migrated)
└── utils/                 # Legacy utils (to be migrated)
```

## 🚀 Cài đặt và Chạy

### Prerequisites
- Node.js 18.17+ hoặc 20.9+
- npm hoặc yarn
- Backend services chạy trên port 8090

### Cài đặt
```bash
# Clone repository
git clone <repository-url>
cd client-frontend

# Cài đặt dependencies
npm install

# Tạo file environment
cp .env.example .env.local

# Chạy development server
npm run dev
```

### Environment Variables
```env
NEXT_PUBLIC_API_URL=http://localhost:8090
NEXT_PUBLIC_SOCKET_URL=ws://localhost:8090/ws
NEXT_PUBLIC_MAX_FILE_SIZE=10485760
NEXT_PUBLIC_ALLOWED_FILE_TYPES=image/jpeg,image/png,image/gif,image/webp
```

## 📦 Tech Stack

- **Framework**: Next.js 15 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **State Management**: React Query (TanStack Query)
- **Authentication**: JWT với refresh token
- **Form Handling**: React Hook Form + Yup
- **HTTP Client**: Axios
- **Icons**: Heroicons
- **UI Components**: Headless UI

## 🏛️ Architecture

### Feature-Based Architecture
Dự án sử dụng feature-based architecture để tổ chức code:

```
features/
├── auth/
│   ├── components/        # Auth-specific components
│   ├── hooks/            # Auth-specific hooks
│   ├── services/         # Auth API calls
│   ├── types/            # Auth TypeScript types
│   ├── utils/            # Auth utility functions
│   └── index.ts          # Feature exports
```

### Shared Resources
Tài nguyên dùng chung được tổ chức trong thư mục `shared/`:

- **Components**: UI components tái sử dụng
- **Services**: API clients và service classes
- **Types**: TypeScript interfaces và types
- **Utils**: Utility functions
- **Constants**: Application constants
- **Config**: Configuration files

## 🔧 Development Guidelines

### Code Style
- Sử dụng TypeScript strict mode
- Follow ESLint và Prettier configuration
- Naming conventions:
  - Components: PascalCase
  - Files: camelCase
  - Constants: UPPER_SNAKE_CASE
  - Types/Interfaces: PascalCase

### Component Structure
```typescript
// Component template
import React from 'react';
import { ComponentProps } from './types';

interface Props extends ComponentProps {
  // Component-specific props
}

const ComponentName: React.FC<Props> = ({ ...props }) => {
  // Component logic
  return (
    // JSX
  );
};

export default ComponentName;
```

### API Integration
```typescript
// Service example
import { apiClient } from '@/shared/config/api-client';
import { API_ENDPOINTS } from '@/shared/constants';

export class ExampleService {
  async getData(): Promise<Data> {
    return apiClient.get<Data>(API_ENDPOINTS.EXAMPLE);
  }
}
```

### State Management
```typescript
// Hook example
import { useQuery } from '@tanstack/react-query';
import { exampleService } from '@/shared/services';

export const useExampleData = () => {
  return useQuery({
    queryKey: ['example'],
    queryFn: () => exampleService.getData(),
  });
};
```

## 🧪 Testing

```bash
# Run tests
npm run test

# Run tests with coverage
npm run test:coverage

# Run e2e tests
npm run test:e2e
```

## 📱 Features

### Authentication
- [x] Login/Register
- [x] JWT Authentication
- [x] Password reset
- [x] Email verification
- [x] Auto refresh tokens

### Posts
- [x] Create posts with images
- [x] Like/Unlike posts
- [x] Comment system
- [x] Share posts
- [x] Bookmark posts

### Users
- [x] User profiles
- [x] Follow/Unfollow users
- [x] User search
- [x] Avatar upload
- [x] Profile editing

### Chat
- [x] Real-time messaging
- [x] Private chat rooms
- [x] Online status
- [x] Typing indicators
- [x] Message history

### Search
- [x] User search
- [x] Search suggestions
- [x] Search history

## 🔄 API Integration

### Backend Services
- **API Gateway**: Port 8090
- **Auth Service**: Port 8080
- **User Service**: Port 8081
- **Post Service**: Port 8082
- **Chat Service**: Port 8083

### Error Handling
- Automatic retry for failed requests
- Token refresh on 401 errors
- User-friendly error messages
- Network error handling

## 📈 Performance

### Optimization
- Code splitting by routes
- Lazy loading components
- Image optimization
- Bundle size monitoring
- React Query caching

### Monitoring
- Error tracking
- Performance metrics
- User analytics

## 🚀 Deployment

### Build
```bash
npm run build
npm run start
```

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.
