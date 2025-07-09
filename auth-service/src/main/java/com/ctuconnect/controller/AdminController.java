package com.ctuconnect.controller;

import com.ctuconnect.dto.AdminCreateUserRequest;
import com.ctuconnect.dto.AdminDashboardDTO;
import com.ctuconnect.dto.AdminUpdateUserRequest;
import com.ctuconnect.dto.UserManagementDTO;
import com.ctuconnect.service.AdminService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import jakarta.validation.Valid;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/admin")
@RequiredArgsConstructor
@Slf4j
@Tag(name = "Admin Management", description = "Admin operations for user management")
@SecurityRequirement(name = "bearerAuth")
public class AdminController {

    private final AdminService adminService;

    @GetMapping("/dashboard")
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "Get admin dashboard statistics")
    public ResponseEntity<AdminDashboardDTO> getDashboardStats() {
        log.info("Admin requesting dashboard statistics");
        AdminDashboardDTO dashboard = adminService.getDashboardStats();
        return ResponseEntity.ok(dashboard);
    }

    @GetMapping("/users")
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "Get all users with pagination and filtering")
    public ResponseEntity<Page<UserManagementDTO>> getAllUsers(
            @PageableDefault(size = 20) Pageable pageable,
            @RequestParam(required = false) String search,
            @RequestParam(required = false) String role,
            @RequestParam(required = false) Boolean isActive) {

        log.info("Admin requesting users list with filters - search: {}, role: {}, isActive: {}",
                search, role, isActive);
        Page<UserManagementDTO> users = adminService.getAllUsers(pageable, search, role, isActive);
        return ResponseEntity.ok(users);
    }

    @GetMapping("/users/{id}")
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "Get user by ID")
    public ResponseEntity<UserManagementDTO> getUserById(@PathVariable Long id) {
        log.info("Admin requesting user with ID: {}", id);
        UserManagementDTO user = adminService.getUserById(id);
        return ResponseEntity.ok(user);
    }

    @PostMapping("/users")
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "Create a new user")
    public ResponseEntity<UserManagementDTO> createUser(@Valid @RequestBody AdminCreateUserRequest request) {
        log.info("Admin creating user with email: {}", request.getEmail());
        UserManagementDTO user = adminService.createUser(request);
        return ResponseEntity.ok(user);
    }

    @PutMapping("/users/{id}")
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "Update user information")
    public ResponseEntity<UserManagementDTO> updateUser(
            @PathVariable Long id,
            @Valid @RequestBody AdminUpdateUserRequest request) {
        log.info("Admin updating user with ID: {}", id);
        UserManagementDTO user = adminService.updateUser(id, request);
        return ResponseEntity.ok(user);
    }

    @DeleteMapping("/users/{id}")
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "Delete user")
    public ResponseEntity<Void> deleteUser(@PathVariable Long id) {
        log.info("Admin deleting user with ID: {}", id);
        adminService.deleteUser(id);
        return ResponseEntity.noContent().build();
    }

    @PatchMapping("/users/{id}/toggle-status")
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "Toggle user active status")
    public ResponseEntity<Void> toggleUserStatus(@PathVariable Long id) {
        log.info("Admin toggling status for user with ID: {}", id);
        adminService.toggleUserStatus(id);
        return ResponseEntity.ok().build();
    }

    @PatchMapping("/users/{id}/verify")
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "Force verify user email")
    public ResponseEntity<Void> forceVerifyUser(@PathVariable Long id) {
        log.info("Admin force verifying user with ID: {}", id);
        adminService.forceVerifyUser(id);
        return ResponseEntity.ok().build();
    }

    @PatchMapping("/users/{id}/reset-password")
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "Reset user password")
    public ResponseEntity<Void> resetUserPassword(
            @PathVariable Long id,
            @RequestBody Map<String, String> request) {
        log.info("Admin resetting password for user with ID: {}", id);
        String newPassword = request.get("newPassword");
        if (newPassword == null || newPassword.trim().isEmpty()) {
            return ResponseEntity.badRequest().build();
        }
        adminService.resetUserPassword(id, newPassword);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/users/role/{role}")
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "Get users by role")
    public ResponseEntity<List<UserManagementDTO>> getUsersByRole(@PathVariable String role) {
        log.info("Admin requesting users with role: {}", role);
        List<UserManagementDTO> users = adminService.getUsersByRole(role);
        return ResponseEntity.ok(users);
    }

    @GetMapping("/users/inactive")
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "Get inactive users")
    public ResponseEntity<List<UserManagementDTO>> getInactiveUsers() {
        log.info("Admin requesting inactive users");
        List<UserManagementDTO> users = adminService.getInactiveUsers();
        return ResponseEntity.ok(users);
    }

    @GetMapping("/users/unverified")
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "Get unverified users")
    public ResponseEntity<List<UserManagementDTO>> getUnverifiedUsers() {
        log.info("Admin requesting unverified users");
        List<UserManagementDTO> users = adminService.getUnverifiedUsers();
        return ResponseEntity.ok(users);
    }

    @PatchMapping("/users/{id}/force-logout")
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "Force logout user")
    public ResponseEntity<Void> forceLogoutUser(@PathVariable Long id) {
        log.info("Admin forcing logout for user with ID: {}", id);
        adminService.forceLogoutUser(id);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/users/{id}/login-history")
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "Get user login history")
    public ResponseEntity<List<Object>> getUserLoginHistory(@PathVariable Long id) {
        log.info("Admin requesting login history for user with ID: {}", id);
        List<Object> history = adminService.getUserLoginHistory(id);
        return ResponseEntity.ok(history);
    }

    @PatchMapping("/users/{id}/role")
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "Update user role")
    public ResponseEntity<Void> updateUserRole(
            @PathVariable Long id,
            @RequestBody Map<String, String> request) {
        log.info("Admin updating role for user with ID: {}", id);
        String newRole = request.get("role");
        if (newRole == null || newRole.trim().isEmpty()) {
            return ResponseEntity.badRequest().build();
        }
        adminService.updateUserRole(id, newRole);
        return ResponseEntity.ok().build();
    }
}
