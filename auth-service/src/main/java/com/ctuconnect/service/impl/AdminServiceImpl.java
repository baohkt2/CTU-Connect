package com.ctuconnect.service.impl;

import com.ctuconnect.dto.AdminCreateUserRequest;
import com.ctuconnect.dto.AdminDashboardDTO;
import com.ctuconnect.dto.AdminUpdateUserRequest;
import com.ctuconnect.dto.UserManagementDTO;
import com.ctuconnect.entity.EmailVerificationEntity;
import com.ctuconnect.entity.RefreshTokenEntity;
import com.ctuconnect.entity.UserEntity;
import com.ctuconnect.repository.EmailVerificationRepository;
import com.ctuconnect.repository.RefreshTokenRepository;
import com.ctuconnect.repository.UserRepository;
import com.ctuconnect.service.AdminService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class AdminServiceImpl implements AdminService {

    private final UserRepository userRepository;
    private final EmailVerificationRepository emailVerificationRepository;
    private final RefreshTokenRepository refreshTokenRepository;
    private final PasswordEncoder passwordEncoder;
    private final KafkaTemplate<String, Object> kafkaTemplate;

    @Override
    public AdminDashboardDTO getDashboardStats() {
        log.info("Fetching admin dashboard statistics");

        AdminDashboardDTO dashboard = new AdminDashboardDTO();

        // Basic user counts
        dashboard.setTotalUsers(userRepository.count());
        dashboard.setActiveUsers(userRepository.countByIsActive(true));
        dashboard.setInactiveUsers(userRepository.countByIsActive(false));
        dashboard.setUnverifiedUsers(userRepository.countUnverifiedUsers());

        // Time-based statistics
        LocalDateTime now = LocalDateTime.now();
        dashboard.setNewUsersToday(userRepository.countByCreatedAtAfter(now.truncatedTo(ChronoUnit.DAYS)));
        dashboard.setNewUsersThisWeek(userRepository.countByCreatedAtAfter(now.minusWeeks(1)));
        dashboard.setNewUsersThisMonth(userRepository.countByCreatedAtAfter(now.minusMonths(1)));

        // Users by role
        Map<String, Long> usersByRole = new HashMap<>();
        List<Object[]> roleStats = userRepository.countUsersByRole();
        for (Object[] stat : roleStats) {
            usersByRole.put((String) stat[0], (Long) stat[1]);
        }
        dashboard.setUsersByRole(usersByRole);

        // Recent users
        List<UserEntity> recentUsers = userRepository.findTop10ByOrderByCreatedAtDesc();
        dashboard.setRecentUsers(recentUsers.stream()
                .map(this::convertToUserManagementDTO)
                .collect(Collectors.toList()));

        // Login statistics (placeholder - would need login tracking)
        Map<String, Long> loginStats = new HashMap<>();
        loginStats.put("today", 0L);
        loginStats.put("week", 0L);
        loginStats.put("month", 0L);
        dashboard.setLoginStatistics(loginStats);

        return dashboard;
    }

    @Override
    public Page<UserManagementDTO> getAllUsers(Pageable pageable, String search, String role, Boolean isActive) {
        log.info("Fetching all users with filters - search: {}, role: {}, isActive: {}", search, role, isActive);

        Page<UserEntity> users;

        if (search != null && !search.trim().isEmpty()) {
            users = userRepository.findByEmailContainingIgnoreCaseOrUsernameContainingIgnoreCase(
                    search, search, pageable);
        } else if (role != null && !role.trim().isEmpty()) {
            users = userRepository.findByRole(role, pageable);
        } else if (isActive != null) {
            users = userRepository.findByIsActive(isActive, pageable);
        } else {
            users = userRepository.findAll(pageable);
        }

        return users.map(this::convertToUserManagementDTO);
    }

    @Override
    public UserManagementDTO getUserById(Long id) {
        log.info("Fetching user by ID: {}", id);

        UserEntity user = userRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + id));

        return convertToUserManagementDTO(user);
    }

    @Override
    @Transactional
    public UserManagementDTO createUser(AdminCreateUserRequest request) {
        log.info("Admin creating user with email: {}", request.getEmail());

        // Check if user already exists
        if (userRepository.existsByEmail(request.getEmail())) {
            throw new RuntimeException("User with this email already exists");
        }

        // Create user entity
        UserEntity user = UserEntity.builder()
                .email(request.getEmail())
                .username(request.getUsername())
                .password(passwordEncoder.encode(request.getPassword()))
                .role(request.getRole())
                .isActive(request.isActive())
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build();

        UserEntity savedUser = userRepository.save(user);

        // Create email verification if needed
        if (request.isEmailVerified()) {
            EmailVerificationEntity verification = EmailVerificationEntity.builder()
                    .user(savedUser)
                    .token("admin-verified")
                    .isVerified(true)
                    .createdAt(LocalDateTime.now())
                    .expiryDate(System.currentTimeMillis() + 86400000) // 24 hours
                    .build();
            emailVerificationRepository.save(verification);
        }

        // Publish user created event
        publishUserCreatedEvent(savedUser);

        return convertToUserManagementDTO(savedUser);
    }

    @Override
    @Transactional
    public UserManagementDTO updateUser(Long id, AdminUpdateUserRequest request) {
        log.info("Admin updating user with ID: {}", id);

        UserEntity user = userRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + id));

        // Update fields if provided
        if (request.getEmail() != null && !request.getEmail().equals(user.getEmail())) {
            if (userRepository.existsByEmail(request.getEmail())) {
                throw new RuntimeException("Email already exists");
            }
            user.setEmail(request.getEmail());
        }

        if (request.getUsername() != null) {
            user.setUsername(request.getUsername());
        }

        if (request.getRole() != null) {
            user.setRole(request.getRole());
        }

        if (request.getIsActive() != null) {
            user.setActive(request.getIsActive());
        }

        if (request.getPassword() != null) {
            user.setPassword(passwordEncoder.encode(request.getPassword()));
        }

        user.setUpdatedAt(LocalDateTime.now());
        UserEntity updatedUser = userRepository.save(user);

        // Update email verification status if needed
        if (request.getIsEmailVerified() != null) {
            EmailVerificationEntity verification = emailVerificationRepository.findByUser(user)
                    .orElse(EmailVerificationEntity.builder()
                            .user(user)
                            .token("admin-verified")
                            .createdAt(LocalDateTime.now())
                            .expiryDate(System.currentTimeMillis() + 86400000)
                            .build());
            verification.setVerified(request.getIsEmailVerified());
            emailVerificationRepository.save(verification);
        }

        // Publish user updated event
        publishUserUpdatedEvent(updatedUser);

        return convertToUserManagementDTO(updatedUser);
    }

    @Override
    @Transactional
    public void deleteUser(Long id) {
        log.info("Admin deleting user with ID: {}", id);

        UserEntity user = userRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + id));

        // Delete related entities
        emailVerificationRepository.deleteByUser(user);
        refreshTokenRepository.deleteByUser(user);

        // Publish user deleted event
        publishUserDeletedEvent(user);

        // Delete user
        userRepository.delete(user);
    }

    @Override
    @Transactional
    public void toggleUserStatus(Long id) {
        log.info("Admin toggling user status for ID: {}", id);

        UserEntity user = userRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + id));

        user.setActive(!user.isActive());
        user.setUpdatedAt(LocalDateTime.now());
        userRepository.save(user);

        // Publish user updated event
        publishUserUpdatedEvent(user);
    }

    @Override
    @Transactional
    public void forceVerifyUser(Long id) {
        log.info("Admin force verifying user with ID: {}", id);

        UserEntity user = userRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + id));

        EmailVerificationEntity verification = emailVerificationRepository.findByUser(user)
                .orElse(EmailVerificationEntity.builder()
                        .user(user)
                        .token("admin-verified")
                        .createdAt(LocalDateTime.now())
                        .expiryDate(System.currentTimeMillis() + 86400000)
                        .build());

        verification.setVerified(true);
        emailVerificationRepository.save(verification);

        // Publish user verification event
        publishUserVerificationEvent(user, true);
    }

    @Override
    @Transactional
    public void resetUserPassword(Long id, String newPassword) {
        log.info("Admin resetting password for user ID: {}", id);

        UserEntity user = userRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + id));

        user.setPassword(passwordEncoder.encode(newPassword));
        user.setUpdatedAt(LocalDateTime.now());
        userRepository.save(user);

        // Invalidate all refresh tokens
        refreshTokenRepository.deleteByUser(user);

        // Publish user updated event
        publishUserUpdatedEvent(user);
    }

    @Override
    public List<UserManagementDTO> getUsersByRole(String role) {
        log.info("Fetching users by role: {}", role);

        List<UserEntity> users = userRepository.findByRole(role);
        return users.stream()
                .map(this::convertToUserManagementDTO)
                .collect(Collectors.toList());
    }

    @Override
    public List<UserManagementDTO> getInactiveUsers() {
        log.info("Fetching inactive users");

        List<UserEntity> users = userRepository.findByIsActive(false);
        return users.stream()
                .map(this::convertToUserManagementDTO)
                .collect(Collectors.toList());
    }

    @Override
    public List<UserManagementDTO> getUnverifiedUsers() {
        log.info("Fetching unverified users");

        List<UserEntity> users = userRepository.findUnverifiedUsers();
        return users.stream()
                .map(this::convertToUserManagementDTO)
                .collect(Collectors.toList());
    }

    @Override
    @Transactional
    public void forceLogoutUser(Long id) {
        log.info("Admin forcing logout for user ID: {}", id);

        UserEntity user = userRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + id));

        // Delete all refresh tokens for this user
        refreshTokenRepository.deleteByUser(user);
    }

    @Override
    public List<Object> getUserLoginHistory(Long id) {
        log.info("Fetching login history for user ID: {}", id);

        // This would require a login history table - placeholder for now
        return List.of();
    }

    @Override
    @Transactional
    public void updateUserRole(Long id, String newRole) {
        log.info("Admin updating role for user ID: {} to role: {}", id, newRole);

        UserEntity user = userRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + id));

        user.setRole(newRole);
        user.setUpdatedAt(LocalDateTime.now());
        userRepository.save(user);

        // Publish user updated event
        publishUserUpdatedEvent(user);
    }

    // Helper methods
    private UserManagementDTO convertToUserManagementDTO(UserEntity user) {
        boolean isEmailVerified = emailVerificationRepository.findByUser(user)
                .map(EmailVerificationEntity::isVerified)
                .orElse(false);

        return new UserManagementDTO(
                user.getId(),
                user.getEmail(),
                user.getUsername(),
                user.getRole(),
                user.isActive(),
                isEmailVerified,
                user.getCreatedAt(),
                user.getUpdatedAt(),
                null // lastLoginAt - would need login tracking
        );
    }

    private void publishUserCreatedEvent(UserEntity user) {
        Map<String, Object> event = new HashMap<>();
        event.put("userId", user.getId());
        event.put("email", user.getEmail());
        event.put("username", user.getUsername());
        event.put("role", user.getRole());
        kafkaTemplate.send("user-registration", user.getId().toString(), event);
    }

    private void publishUserUpdatedEvent(UserEntity user) {
        Map<String, Object> event = new HashMap<>();
        event.put("userId", user.getId());
        event.put("email", user.getEmail());
        event.put("username", user.getUsername());
        event.put("role", user.getRole());
        event.put("isActive", user.isActive());
        kafkaTemplate.send("user-updated", user.getId().toString(), event);
    }

    private void publishUserDeletedEvent(UserEntity user) {
        Map<String, Object> event = new HashMap<>();
        event.put("userId", user.getId());
        event.put("email", user.getEmail());
        kafkaTemplate.send("user-deleted", user.getId().toString(), event);
    }

    private void publishUserVerificationEvent(UserEntity user, boolean isVerified) {
        Map<String, Object> event = new HashMap<>();
        event.put("userId", user.getId());
        event.put("email", user.getEmail());
        event.put("verificationToken", "admin-verified");
        event.put("isVerified", isVerified);
        kafkaTemplate.send("user-verification", user.getId().toString(), event);
    }
}
