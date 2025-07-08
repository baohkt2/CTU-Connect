package com.ctuconnect.controller;

import com.ctuconnect.dto.UserDTO;
import com.ctuconnect.security.annotation.RequireAuth;
import com.ctuconnect.service.UserSyncService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

/**
 * Controller để xử lý đồng bộ dữ liệu giữa auth-db và user-db
 * Các endpoint này được gọi từ auth-service hoặc gateway
 */
@RestController
@RequestMapping("/api/users/sync")
public class UserSyncController {

    @Autowired
    private UserSyncService userSyncService;

    /**
     * Tạo user profile trong user-db khi user được tạo ở auth-db
     * Endpoint này được gọi từ auth-service
     */
    @PostMapping("/create")
    @RequireAuth(roles = {"SYSTEM", "ADMIN"}) // Chỉ system hoặc admin mới có thể gọi
    public ResponseEntity<UserDTO> syncUserFromAuth(
            @RequestParam String userId,
            @RequestParam String email,
            @RequestParam String role) {
        UserDTO userDTO = userSyncService.syncUserFromAuth(userId, email, role);
        return ResponseEntity.ok(userDTO);
    }

    /**
     * Cập nhật thông tin user khi có thay đổi từ auth-db
     */
    @PutMapping("/update")
    @RequireAuth(roles = {"SYSTEM", "ADMIN"})
    public ResponseEntity<UserDTO> updateUserFromAuth(
            @RequestParam String userId,
            @RequestParam String email,
            @RequestParam String role) {
        UserDTO userDTO = userSyncService.updateUserFromAuth(userId, email, role);
        return ResponseEntity.ok(userDTO);
    }

    /**
     * Xóa user khỏi user-db khi user bị xóa ở auth-db
     */
    @DeleteMapping("/delete")
    @RequireAuth(roles = {"SYSTEM", "ADMIN"})
    public ResponseEntity<String> deleteUserFromAuth(@RequestParam String userId) {
        userSyncService.deleteUserFromAuth(userId);
        return ResponseEntity.ok("User deleted from user database successfully");
    }

    /**
     * Kiểm tra tính nhất quán dữ liệu giữa auth-db và user-db
     */
    @GetMapping("/check")
    @RequireAuth(roles = {"SYSTEM", "ADMIN"})
    public ResponseEntity<Boolean> checkUserSync(
            @RequestParam String userId,
            @RequestParam String email,
            @RequestParam String role) {
        boolean isSynced = userSyncService.isUserSynced(userId, email, role);
        return ResponseEntity.ok(isSynced);
    }
}
