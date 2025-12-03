package com.ctuconnect.service;

import com.ctuconnect.dto.AdminCreateUserRequest;
import com.ctuconnect.dto.AdminDashboardDTO;
import com.ctuconnect.dto.AdminUpdateUserRequest;
import com.ctuconnect.dto.UserManagementDTO;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

public interface AdminService {

    /**
     * Get admin dashboard statistics
     */
    AdminDashboardDTO getDashboardStats();

    /**
     * Get all users with pagination and filtering
     */
    Page<UserManagementDTO> getAllUsers(Pageable pageable, String search, String role, Boolean isActive);

    /**
     * Get user by ID
     */
    UserManagementDTO getUserById(Long id);

    UserManagementDTO getUserById(String id);

    /**
     * Create a new user (Admin only)
     */
    UserManagementDTO createUser(AdminCreateUserRequest request);

    /**
     * Update user information (Admin only)
     */
    UserManagementDTO updateUser(Long id, AdminUpdateUserRequest request);

    /**
     * Delete user (Admin only)
     */
    void deleteUser(Long id);

    /**
     * Activate/Deactivate user
     */
    void toggleUserStatus(Long id);

    /**
     * Force verify user email
     */
    void forceVerifyUser(Long id);

    /**
     * Reset user password (Admin only)
     */
    void resetUserPassword(Long id, String newPassword);

    @Transactional
    UserManagementDTO updateUser(String id, AdminUpdateUserRequest request);

    @Transactional
    void deleteUser(String id);

    @Transactional
    void toggleUserStatus(String id);

    @Transactional
    void forceVerifyUser(String id);

    @Transactional
    void resetUserPassword(String id, String newPassword);

    /**
     * Get users by role
     */
    List<UserManagementDTO> getUsersByRole(String role);

    /**
     * Get inactive users
     */
    List<UserManagementDTO> getInactiveUsers();

    /**
     * Get unverified users
     */
    List<UserManagementDTO> getUnverifiedUsers();

    /**
     * Force logout user (invalidate all refresh tokens)
     */
    void forceLogoutUser(Long id);

    /**
     * Get user login history
     */
    List<Object> getUserLoginHistory(Long id);

    /**
     * Update user role
     */
    void updateUserRole(Long id, String newRole);

    @Transactional
    void forceLogoutUser(String id);

    List<Object> getUserLoginHistory(String id);

    @Transactional
    void updateUserRole(String id, String newRole);
}
