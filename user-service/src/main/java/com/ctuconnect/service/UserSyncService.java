package com.ctuconnect.service;

import com.ctuconnect.dto.UserDTO;
import com.ctuconnect.entity.UserEntity;
import com.ctuconnect.enums.Role;
import com.ctuconnect.repository.UserRepository;
import com.ctuconnect.security.SecurityContextHolder;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.stream.Collectors;

/**
 * Service để đồng bộ dữ liệu user giữa auth-db và user-db
 * Đảm bảo user ID ở cả 2 database luôn nhất quán
 */
@Service
public class UserSyncService {

    @Autowired
    private UserRepository userRepository;

    /**
     * Tạo user profile trong user-db khi user được tạo ở auth-db
     * Được gọi từ auth-service thông qua message queue hoặc API call
     */
    @Transactional
    public UserDTO syncUserFromAuth(String userId, String email, String role) {
        // Kiểm tra xem user đã tồn tại chưa
        if (userRepository.existsByEmail(email)) {
            throw new IllegalStateException("User already exists in user database: " + userId);
        }

        // Tạo user entity mới với thông tin cơ bản từ auth-db
        UserEntity userEntity = new UserEntity();
        userEntity.setId(userId); // ID đồng bộ với auth-db
        userEntity.setEmail(email);
        userEntity.setIsProfileCompleted(false);
        userEntity.setRole(Role.valueOf(role));
        userEntity.setCreatedAt(LocalDateTime.now());
        userEntity.setUpdatedAt(LocalDateTime.now());

        UserEntity savedUser = userRepository.save(userEntity);
        return mapToDTO(savedUser);
    }

    /**
     * Cập nhật thông tin user khi có thay đổi từ auth-db
     */
    @Transactional
    public UserDTO updateUserFromAuth(String userId, String email, String role) {
        UserEntity userEntity = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found in user database: " + userId));

        // Chỉ cập nhật những field được đồng bộ từ auth-db
        userEntity.setEmail(email);
        userEntity.setRole(Role.valueOf(role));
        userEntity.setUpdatedAt(LocalDateTime.now());

        UserEntity updatedUser = userRepository.save(userEntity);
        return mapToDTO(updatedUser);
    }

    /**
     * Xóa user khỏi user-db khi user bị xóa ở auth-db
     */
    @Transactional
    public void deleteUserFromAuth(String userId) {
        if (!userRepository.existsById(userId)) {
            throw new RuntimeException("User not found in user database: " + userId);
        }

        // Xóa tất cả relationships trước khi xóa user
        UserEntity user = userRepository.findById(userId).get();
        user.getFriends().clear();
        userRepository.save(user);

        // Xóa user khỏi friend lists của những user khác
        userRepository.findAll().forEach(otherUser -> {
            otherUser.getFriends().removeIf(friend -> friend.getId().equals(userId));
            userRepository.save(otherUser);
        });

        // Xóa user
        userRepository.deleteById(userId);
    }

    /**
     * Kiểm tra tính nhất quán dữ liệu giữa auth-db và user-db
     */
    public boolean isUserSynced(String userId, String email, String role) {
        UserEntity userEntity = userRepository.findById(userId).orElse(null);

        if (userEntity == null) {
            return false;
        }

        return email.equals(userEntity.getEmail()) && role.equals(userEntity.getRole());
    }

    /**
     * Đảm bảo user hiện tại có quyền truy cập vào dữ liệu
     */
    public void validateUserAccess(String targetUserId) {
        String currentUserId = SecurityContextHolder.getCurrentUserId();
        boolean isAdmin = SecurityContextHolder.isCurrentUserAdmin();

        if (currentUserId == null) {
            throw new SecurityException("Authentication required");
        }

        if (!isAdmin && !currentUserId.equals(targetUserId)) {
            throw new SecurityException("Access denied: Can only access own data");
        }
    }

    /**
     * Map UserEntity to UserDTO
     */
    private UserDTO mapToDTO(UserEntity entity) {
        UserDTO dto = new UserDTO();
        dto.setId(entity.getId());
        dto.setEmail(entity.getEmail());
        dto.setUsername(entity.getUsername());
        dto.setFullName(entity.getFullName());
        dto.setRole(entity.getRole() != null ? entity.getRole().toString() : null);
        dto.setBio(entity.getBio());
        dto.setIsActive(entity.getIsActive());
        dto.setIsProfileCompleted(entity.getIsProfileCompleted());
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setUpdatedAt(entity.getUpdatedAt());

        // Student fields
        dto.setStudentId(entity.getStudentId());
        dto.setMajor(entity.getMajor());
        dto.setBatch(entity.getBatch());
        // Lecturer fields
        dto.setStaffCode(entity.getStaffCode());
        dto.setAcademic(entity.getAcademic());
        dto.setDegree(entity.getDegree());
        dto.setPosition(entity.getPosition());

        // Common fields
        dto.setFaculty(entity.getFaculty());
        dto.setCollege(entity.getCollege());
        dto.setGender(entity.getGender());

        // Media fields
        dto.setAvatarUrl(entity.getAvatarUrl());
        dto.setBackgroundUrl(entity.getBackgroundUrl());

        // Friends mapping
        if (entity.getFriends() != null) {
            dto.setFriendIds(
                    entity.getFriends().stream()
                            .map(UserEntity::getId)
                            .collect(Collectors.toSet())
            );
        }

        return dto;
    }

}
