# Profile Update và Avatar/Cover Image Upload Fix

## Vấn đề đã được sửa:

### 1. Avatar và Cover Image Upload Logic
- **Backend**: Thêm hỗ trợ `avatarUrl` và `backgroundUrl` trong UserService.updateUserProfile()
- **Backend**: Cập nhật UserMapper để map `avatarUrl` và `backgroundUrl` từ UserEntity
- **Backend**: Lưu ngay sau khi update image URLs để đảm bảo persist vào database
- **Frontend**: Luồng upload hoạt động đúng:
  1. User chọn file → UploadFile component
  2. File được upload lên media-service qua useUploadMedia hook
  3. Media service trả về CloudinaryUrl
  4. URL được set vào formData (avatarUrl/backgroundUrl)
  5. Khi submit form, URL được gửi về user-service
  6. User-service lưu URL vào database

### 2. Auto-select Dropdown Fields
- **Backend**: Thêm các field `collegeCode`, `facultyCode`, `majorCode`, `batchCode`, `genderCode` vào UserProfileDTO
- **Backend**: Cập nhật UserMapper để trả về cả name (hiển thị) và code (form selection)
- **Backend**: Hỗ trợ lookup gender bằng cả code và name
- **Frontend**: Cập nhật types và logic để sử dụng code fields từ backend response
- **Frontend**: Thêm debug logging để theo dõi auto-selection
- **Frontend**: Sử dụng `user.collegeCode`, `user.facultyCode`, `user.majorCode`, `user.genderCode` để auto-select

### 3. Gender Code Mapping
- **Backend**: Thêm method `findByCode()` trong GenderRepository
- **Backend**: Cập nhật UserService để accept cả gender code (M, F) và name (Nam, Nữ)
- **Frontend**: Map genderCode từ form vào genderName field khi gửi request

## Files đã sửa:

### Backend:
1. `user-service/src/main/java/com/ctuconnect/service/UserService.java`
   - Thêm xử lý `avatarUrl` và `backgroundUrl` trong updateUserProfile()
   - Lưu user entity ngay sau update để persist image URLs
   - Hỗ trợ lookup gender bằng code hoặc name

2. `user-service/src/main/java/com/ctuconnect/mapper/UserMapper.java`
   - Thêm mapping `avatarUrl` và `backgroundUrl`
   - Thêm mapping các code fields cho auto-selection

3. `user-service/src/main/java/com/ctuconnect/dto/UserProfileDTO.java`
   - Thêm fields: `collegeCode`, `facultyCode`, `majorCode`, `batchCode`, `genderCode`

4. `user-service/src/main/java/com/ctuconnect/repository/GenderRepository.java`
   - Thêm method `findByCode(String code)`

### Frontend:
1. `client-frontend/src/types/index.ts`
   - Cập nhật User interface với các fields code và name riêng biệt

2. `client-frontend/src/components/profile/StudentProfileForm.tsx`
   - Sử dụng code fields từ backend response để auto-select
   - Thêm debug logging

3. `client-frontend/src/services/userService.ts`
   - Map genderCode sang genderName khi gửi update request

## Vấn đề đã xác định và giải pháp:

### ✅ Đã sửa:
1. **Avatar không hiển thị**: Backend không lưu avatarUrl sau update → Đã sửa bằng cách save entity ngay sau update
2. **Giới tính không auto-select**: Frontend gửi code nhưng backend expect name → Đã sửa bằng cách hỗ trợ lookup bằng cả code và name
3. **Avatar URL không trả về trong response**: Mapper không include avatarUrl → Đã thêm vào mapper

### ⚠️ Cần kiểm tra:
1. **Upload ảnh nền lỗi 401**: Có thể do authentication token issue khi upload lần thứ 2 liên tiếp
   - Cần check: API gateway có refresh token đúng không?
   - Cần check: Cookie có được gửi đúng trong multipart/form-data request không?

## Testing:
- Backend compile thành công ✅
- Luồng upload avatar đầy đủ từ frontend → media-service → user-service ✅
- Auto-selection sử dụng code fields từ backend ✅
- Gender code mapping (M → Nam, F → Nữ) ✅
- Avatar URL được trả về trong profile response ✅

## Hướng dẫn test:
1. Login vào hệ thống
2. Vào trang /profile/update
3. Upload avatar → Kiểm tra URL được hiển thị
4. Upload cover image → Nếu lỗi 401, check browser console và network tab
5. Điền đầy đủ thông tin và submit
6. Kiểm tra profile response có đầy đủ avatarUrl, backgroundUrl, và các code fields
7. Reload trang /profile/update → Kiểm tra các dropdown đã auto-select đúng