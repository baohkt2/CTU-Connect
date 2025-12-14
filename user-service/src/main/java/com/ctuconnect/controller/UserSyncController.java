package com.ctuconnect.controller;

import com.ctuconnect.dto.AuthorDTO;
import com.ctuconnect.dto.UserDTO;
import com.ctuconnect.security.annotation.RequireAuth;
import com.ctuconnect.service.UserSyncService;
import com.ctuconnect.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Set;

/**
 * Controller để xử lý đồng bộ dữ liệu giữa auth-db và user-db
 * Các endpoint này được gọi từ auth-service hoặc các microservices khác
 */
@RestController
@RequestMapping("/api/users/sync")
public class UserSyncController {

    @Autowired
    private UserSyncService userSyncService;

    @Autowired
    private UserService userService;

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

    /**
     * Lấy thông tin tác giả cho post-service
     * Endpoint này được gọi từ post-service để lấy author info
     * Không yêu cầu authentication vì đây là internal service call
     * Trả về 404 nếu không tìm thấy user (thay vì 500 error)
     */
    @GetMapping("/authors/{id}")
    public ResponseEntity<AuthorDTO> getAuthorInfo(@PathVariable("id") String authorId) {
        AuthorDTO authorInfo = userSyncService.getAuthorInfo(authorId);
        if (authorInfo == null) {
            return ResponseEntity.notFound().build();
        }
        return ResponseEntity.ok(authorInfo);
    }

    /**
     * Lấy danh sách ID bạn bè của user
     * Endpoint này được gọi từ post-service cho news feed algorithm
     * Không yêu cầu authentication vì đây là internal service call
     */
    @GetMapping("/{userId}/friends/ids")
    public ResponseEntity<Set<String>> getFriendIds(@PathVariable("userId") String userId) {
        Set<String> friendIds = userService.getFriendIds(userId);
        return ResponseEntity.ok(friendIds);
    }

    /**
     * Lấy danh sách user có tương tác gần với user
     * Endpoint này được gọi từ post-service cho news feed algorithm
     */
    @GetMapping("/{userId}/close-interactions")
    public ResponseEntity<Set<String>> getCloseInteractionIds(@PathVariable("userId") String userId) {
        Set<String> closeInteractionIds = userService.getCloseInteractionIds(userId);
        return ResponseEntity.ok(closeInteractionIds);
    }

    /**
     * Lấy danh sách user cùng khoa
     * Endpoint này được gọi từ post-service cho news feed algorithm
     */
    @GetMapping("/{userId}/same-faculty")
    public ResponseEntity<Set<String>> getSameFacultyUserIds(@PathVariable("userId") String userId) {
        Set<String> sameFacultyIds = userService.getSameFacultyUserIds(userId);
        return ResponseEntity.ok(sameFacultyIds);
    }

    /**
     * Lấy danh sách user cùng ngành
     * Endpoint này được gọi từ post-service cho news feed algorithm
     */
    @GetMapping("/{userId}/same-major")
    public ResponseEntity<Set<String>> getSameMajorUserIds(@PathVariable("userId") String userId) {
        Set<String> sameMajorIds = userService.getSameMajorUserIds(userId);
        return ResponseEntity.ok(sameMajorIds);
    }

    /**
     * Lấy danh sách interest tags của user
     * Endpoint này được gọi từ post-service cho content recommendation
     */
    @GetMapping("/{userId}/interest-tags")
    public ResponseEntity<Set<String>> getUserInterestTags(@PathVariable("userId") String userId) {
        Set<String> interestTags = userService.getUserInterestTags(userId);
        return ResponseEntity.ok(interestTags);
    }

    /**
     * Lấy danh sách preferred categories của user
     * Endpoint này được gọi từ post-service cho content filtering
     */
    @GetMapping("/{userId}/preferred-categories")
    public ResponseEntity<Set<String>> getUserPreferredCategories(@PathVariable("userId") String userId) {
        Set<String> preferredCategories = userService.getUserPreferredCategories(userId);
        return ResponseEntity.ok(preferredCategories);
    }

    /**
     * Lấy faculty ID của user
     * Endpoint này được gọi từ post-service cho group filtering
     */
    @GetMapping("/{userId}/faculty-id")
    public ResponseEntity<String> getUserFacultyId(@PathVariable("userId") String userId) {
        String facultyId = userService.getUserFacultyId(userId);
        return ResponseEntity.ok(facultyId);
    }

    /**
     * Lấy major ID của user
     * Endpoint này được gọi từ post-service cho group filtering
     */
    @GetMapping("/{userId}/major-id")
    public ResponseEntity<String> getUserMajorId(@PathVariable("userId") String userId) {
        String majorId = userService.getUserMajorId(userId);
        return ResponseEntity.ok(majorId);
    }
}
