package com.ctuconnect.service;

import com.ctuconnect.dto.UserDTO;
import com.ctuconnect.entity.UserEntity;
import com.ctuconnect.repository.UserRepository;
import com.ctuconnect.security.SecurityContextHolder;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;

@Service
public class UserSyncService {

    @Autowired
    private UserRepository userRepository;

    /**
     * Đồng bộ user từ auth-service khi tạo mới.
     */
    @Transactional
    public UserDTO syncUserFromAuth(String userId, String email, String role) {
        if (userRepository.existsByEmail(email)) {
            throw new IllegalStateException("User already exists: " + userId);
        }

        UserEntity user = new UserEntity();
        user.setId(userId);
        user.setEmail(email);
        user.setRole(role);
        user.setIsActive(true);
        user.setCreatedAt(LocalDateTime.now());
        user.setUpdatedAt(LocalDateTime.now());

        return mapToDTO(userRepository.save(user));
    }

    /**
     * Cập nhật thông tin user từ auth-db
     */
    @Transactional
    public UserDTO updateUserFromAuth(String userId, String email, String role) {
        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found: " + userId));

        user.setEmail(email);
        user.setRole(role);
        user.setUpdatedAt(LocalDateTime.now());

        return mapToDTO(userRepository.save(user));
    }

    /**
     * Tạo hoặc cập nhật user từ auth-service (email, username, role)
     */
    @Transactional
    public UserDTO createUserFromAuthService(String userId, String email, String username, String role) {
        if (userRepository.existsById(userId)) {
            return updateUserFromAuth(userId, email, role);
        }

        UserEntity user = UserEntity.fromAuthService(userId, email, username, role);
        return mapToDTO(userRepository.save(user));
    }

    /**
     * Xóa user và các mối quan hệ liên quan (friendship, friend requests).
     */
    @Transactional
    public void deleteUserFromAuth(String userId) {
        if (!userRepository.existsById(userId)) {
            throw new RuntimeException("User not found: " + userId);
        }

        // Xóa mối quan hệ bạn bè
        var friends = userRepository.findFriends(userId, null).getContent(); // Pageable=null nghĩa là tất cả
        for (var friendProj : friends) {
            String friendId = friendProj.getUser().getId();
            userRepository.removeFriend(userId, friendId);
        }

        // Xóa lời mời đã gửi
        var sentRequests = userRepository.findSentFriendRequests(userId);
        for (var req : sentRequests) {
            userRepository.rejectFriendRequest(userId, req.getUser().getId());
        }

        // Xóa lời mời đã nhận
        var receivedRequests = userRepository.findReceivedFriendRequests(userId);
        for (var req : receivedRequests) {
            userRepository.rejectFriendRequest(req.getUser().getId(), userId);
        }

        userRepository.deleteById(userId);
    }

    /**
     * Kiểm tra dữ liệu đồng bộ từ auth-db
     */
    public boolean isUserSynced(String userId, String email, String role) {
        return userRepository.findById(userId)
                .map(user -> email.equals(user.getEmail()) && role.equals(user.getRole()))
                .orElse(false);
    }

    /**
     * Kiểm tra quyền truy cập user
     */
    public void validateUserAccess(String userId) {
        String currentUserId = SecurityContextHolder.getCurrentUserId();
        if (!userId.equals(currentUserId)) {
            throw new SecurityException("Access denied: User can only access their own data");
        }
    }

    /**
     * Mapping UserEntity sang UserDTO
     */
    private UserDTO mapToDTO(UserEntity userEntity) {
        UserDTO dto = new UserDTO();
        dto.setId(userEntity.getId());
        dto.setEmail(userEntity.getEmail());
        dto.setUsername(userEntity.getUsername());
        dto.setFullName(userEntity.getFullName());
        dto.setStudentId(userEntity.getStudentId());
        dto.setRole(userEntity.getRole());
        dto.setBio(userEntity.getBio());
        dto.setIsActive(userEntity.isActive());
        dto.setCreatedAt(userEntity.getCreatedAt());
        dto.setUpdatedAt(userEntity.getUpdatedAt());

        var profile = userRepository.findUserProfileById(userEntity.getId()).orElse(null);
        if (profile != null) {
            dto.setCollege(profile.getCollege());
            dto.setFaculty(profile.getFaculty());
            dto.setMajor(profile.getMajor());
            dto.setBatch(profile.getBatch());
            dto.setGender(profile.getGender());
        }

        return dto;
    }
}
