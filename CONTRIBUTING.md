# Hướng dẫn đóng góp cho CTU-Connect

Cảm ơn bạn đã quan tâm đến việc đóng góp cho CTU-Connect! 🎉

## 📋 Mục lục

- [Quy tắc ứng xử](#quy-tắc-ứng-xử)
- [Cách đóng góp](#cách-đóng-góp)
- [Quy trình phát triển](#quy-trình-phát-triển)
- [Coding Standards](#coding-standards)
- [Commit Messages](#commit-messages)
- [Pull Request](#pull-request)

---

## 📜 Quy tắc ứng xử

- Tôn trọng tất cả thành viên trong cộng đồng
- Sử dụng ngôn ngữ lịch sự và chuyên nghiệp
- Chấp nhận phản hồi mang tính xây dựng
- Tập trung vào những gì tốt nhất cho cộng đồng

---

## 🚀 Cách đóng góp

### 1. Fork Repository

```bash
# Fork repository trên GitHub, sau đó clone
git clone https://github.com/your-username/CTU-Connect.git
cd CTU-Connect
```

### 2. Tạo Branch mới

```bash
# Tạo branch cho feature/fix mới
git checkout -b feature/ten-tinh-nang

# Hoặc cho bug fix
git checkout -b fix/ten-bug
```

### 3. Cài đặt môi trường phát triển

```bash
# Copy file environment
cp .env.example .env

# Cấu hình các biến môi trường cần thiết
# Xem README.md để biết chi tiết

# Khởi động infrastructure với Docker
docker-compose up -d postgres neo4j mongodb redis kafka
```

### 4. Thực hiện thay đổi

- Viết code theo [Coding Standards](#coding-standards)
- Thêm tests cho code mới
- Cập nhật documentation nếu cần

### 5. Commit và Push

```bash
git add .
git commit -m "feat: mô tả ngắn gọn thay đổi"
git push origin feature/ten-tinh-nang
```

### 6. Tạo Pull Request

- Tạo PR từ branch của bạn vào `main`
- Điền đầy đủ thông tin trong PR template
- Chờ review và feedback

---

## 🔄 Quy trình phát triển

### Branch Naming Convention

| Prefix | Mô tả | Ví dụ |
|--------|-------|-------|
| `feature/` | Tính năng mới | `feature/add-notification` |
| `fix/` | Sửa bug | `fix/login-error` |
| `docs/` | Cập nhật documentation | `docs/update-readme` |
| `refactor/` | Refactoring code | `refactor/user-service` |
| `test/` | Thêm/sửa tests | `test/add-user-tests` |

### Workflow

```
main
  │
  ├── feature/new-feature
  │     └── commit → commit → PR → merge
  │
  └── fix/bug-fix
        └── commit → PR → merge
```

---

## 📝 Coding Standards

### Java (Backend Services)

```java
// Package naming
package vn.ctu.edu.servicename;

// Class naming - PascalCase
public class UserService { }

// Method naming - camelCase
public void getUserById(String id) { }

// Constants - UPPER_SNAKE_CASE
public static final String API_VERSION = "v1";

// Use Lombok annotations
@Data
@AllArgsConstructor
@NoArgsConstructor
public class User { }
```

### TypeScript/JavaScript (Frontend)

```typescript
// Component naming - PascalCase
export function UserProfile() { }

// Function naming - camelCase
const getUserData = async () => { }

// Constants - UPPER_SNAKE_CASE
const API_BASE_URL = 'http://localhost:8090';

// Type/Interface naming - PascalCase
interface UserData {
  id: string;
  name: string;
}
```

### Python (AI Service)

```python
# Module naming - snake_case
from recommendation import user_embedding

# Function naming - snake_case
def get_user_embedding(user_id: str) -> List[float]:
    pass

# Class naming - PascalCase
class RecommendationEngine:
    pass

# Constants - UPPER_SNAKE_CASE
MODEL_PATH = "/app/model"
```

### Code Quality

- ✅ Không có code duplication
- ✅ Hàm ngắn gọn, làm một việc duy nhất
- ✅ Đặt tên biến/hàm có ý nghĩa
- ✅ Comment cho logic phức tạp
- ✅ Xử lý errors/exceptions đầy đủ

---

## 💬 Commit Messages

Sử dụng [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Mô tả |
|------|-------|
| `feat` | Tính năng mới |
| `fix` | Sửa bug |
| `docs` | Thay đổi documentation |
| `style` | Format code (không thay đổi logic) |
| `refactor` | Refactoring code |
| `test` | Thêm/sửa tests |
| `chore` | Maintenance tasks |

### Examples

```bash
# Tính năng mới
git commit -m "feat(user-service): add friend suggestion endpoint"

# Sửa bug
git commit -m "fix(auth): resolve token expiration issue"

# Documentation
git commit -m "docs: update API documentation"

# Refactoring
git commit -m "refactor(post-service): optimize database queries"
```

---

## 🔍 Pull Request

### PR Template

```markdown
## Mô tả
<!-- Mô tả ngắn gọn thay đổi -->

## Loại thay đổi
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] Code follows coding standards
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No breaking changes
- [ ] Self-reviewed code

## Screenshots (nếu có)
<!-- Thêm screenshots nếu có UI changes -->

## Related Issues
<!-- Link đến issues liên quan -->
Fixes #123
```

### Review Process

1. **Automated Checks**: CI/CD pipeline sẽ chạy tests
2. **Code Review**: Ít nhất 1 reviewer approve
3. **Merge**: Sau khi approved, merge vào `main`

---

## 🧪 Testing

### Backend (Java)

```bash
# Chạy tất cả tests
cd auth-service
mvn test

# Chạy specific test
mvn test -Dtest=UserServiceTest
```

### Frontend (TypeScript)

```bash
cd client-frontend
npm test

# Với coverage
npm test -- --coverage
```

### Python (AI Service)

```bash
cd recommend-service/python-model
pytest tests/

# Với coverage
pytest --cov=. tests/
```

---

## 📞 Liên hệ

- **GitHub Issues**: Tạo issue nếu có bug hoặc feature request
- **Discussions**: Thảo luận về ý tưởng mới

---

Cảm ơn bạn đã đóng góp! 🙏
