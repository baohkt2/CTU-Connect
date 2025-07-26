package com.ctuconnect.controller;

import com.ctuconnect.dto.FriendsDTO;
import com.ctuconnect.dto.RelationshipFilterDTO;
import com.ctuconnect.dto.UserDTO;
import com.ctuconnect.security.SecurityContextHolder;
import com.ctuconnect.security.annotation.RequireAuth;
import com.ctuconnect.service.UserService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Slf4j
@RestController
@RequestMapping("/api/users")
@RequireAuth // Yêu cầu xác thực cho tất cả endpoints
public class UserController {

    @Autowired
    private UserService userService;

    // ===================== USER PROFILE MANAGEMENT =====================

    @PostMapping("/register")
    @RequireAuth(roles = {"ADMIN"}) // Chỉ admin mới có thể tạo user mới
    public ResponseEntity<UserDTO> createUser(@RequestBody UserDTO userDTO) {
        return ResponseEntity.ok(userService.createUser(userDTO));
    }

    @GetMapping("/checkMyInfo")
    @RequireAuth(selfOnly = true) // Kiểm tra đã nhập thông tin cá nhân hay chưa
    public Boolean checkMyInfo() {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        return (userService.checkProfile(currentUserId));
    }

    @GetMapping("/{userId}/profile")
    @RequireAuth // Có thể xem profile của người khác
    public ResponseEntity<UserDTO> getUserProfile(@PathVariable String userId) {
        return ResponseEntity.ok(userService.getUserProfile(userId));
    }

    @GetMapping("/me/profile")
    @RequireAuth(selfOnly = true) // Xem profile của chính mình
    public ResponseEntity<UserDTO> getMyProfile() {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        return ResponseEntity.ok(userService.getUserProfile(currentUserId));
    }

    @PutMapping("/me/profile")
    @RequireAuth(selfOnly = true) // Cập nhật profile của chính mình
    public ResponseEntity<UserDTO> updateMyProfile(@RequestBody Object profileRequest) {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();

        // Xác định loại request dựa trên user role và fields có trong request
        UserDTO currentUser = userService.getUserProfile(currentUserId);

        if (currentUser.getRole().equals("STUDENT")) {
            // Convert to StudentProfileUpdateRequest và xử lý
            return ResponseEntity.ok(userService.updateStudentProfile(currentUserId, profileRequest));
        } else if (currentUser.getRole().equals("FACULTY")) {
            // Convert to FacultyProfileUpdateRequest và xử lý
            return ResponseEntity.ok(userService.updateFacultyProfile(currentUserId, profileRequest));
        } else {
            // Fallback to original UserDTO update
            return ResponseEntity.ok(userService.updateUserProfile(currentUserId, (UserDTO) profileRequest));
        }
    }

    // ===================== FRIEND REQUEST MANAGEMENT =====================

    @PostMapping("/{userId}/invite/{friendId}")
    @RequireAuth // Gửi lời mời kết bạn
    public ResponseEntity<String> sendFriendRequest(@PathVariable String userId, @PathVariable String friendId) {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();

        // Chỉ cho phép gửi lời mời từ chính tài khoản của mình
        if (!userId.equals(currentUserId)) {
            return ResponseEntity.status(403).body("Can only send friend requests from your own account");
        }

        userService.addFriend(userId, friendId);
        return ResponseEntity.ok("Friend request sent successfully");
    }

    @PostMapping("/me/invite/{friendId}")
    @RequireAuth // Gửi lời mời kết bạn từ tài khoản của mình
    public ResponseEntity<String> sendMyFriendRequest(@PathVariable String friendId) {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        userService.addFriend(currentUserId, friendId);
        return ResponseEntity.ok("Friend request sent successfully");
    }


    @PostMapping("/me/accept-invite/{friendId}")
    @RequireAuth // Chấp nhận lời mời kết bạn
    public ResponseEntity<String> acceptMyFriendRequest(@PathVariable String friendId) {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        userService.acceptFriendInvite(currentUserId, friendId);
        return ResponseEntity.ok("Friend request accepted successfully");
    }

    @PostMapping("/{userId}/reject-invite/{friendId}")
    @RequireAuth // Từ chối lời mời kết bạn
    public ResponseEntity<String> rejectFriendRequest(@PathVariable String userId, @PathVariable String friendId) {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();

        // Chỉ cho phép từ chối lời mời gửi đến chính mình
        if (!userId.equals(currentUserId)) {
            return ResponseEntity.status(403).body("Can only reject friend requests sent to your own account");
        }

        userService.rejectFriendInvite(userId, friendId);
        return ResponseEntity.ok("Friend request rejected successfully");
    }

    @PostMapping("/me/reject-invite/{friendId}")
    @RequireAuth // Từ chối lời mời kết bạn
    public ResponseEntity<String> rejectMyFriendRequest(@PathVariable String friendId) {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        userService.rejectFriendInvite(currentUserId, friendId);
        return ResponseEntity.ok("Friend request rejected successfully");
    }

    @DeleteMapping("/me/friends/{friendId}")
    @RequireAuth // Hủy kết bạn
    public ResponseEntity<String> removeMyFriend(@PathVariable String friendId) {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        userService.removeFriend(currentUserId, friendId);
        return ResponseEntity.ok("Friend removed successfully");
    }

    // ===================== FRIEND REQUESTS VIEWING =====================
    @GetMapping("/me/friend-requests")
    @RequireAuth(selfOnly = true) // Xem lời mời kết bạn nhận được
    public ResponseEntity<List<UserDTO>> getMyFriendRequests() {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        return ResponseEntity.ok(userService.getFriendRequests(currentUserId));
    }

    @GetMapping("/me/friend-requested")
    @RequireAuth(selfOnly = true) // Xem lời mời kết bạn đã gửi
    public ResponseEntity<List<UserDTO>> getMyFriendRequested() {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        return ResponseEntity.ok(userService.getFriendRequested(currentUserId));
    }

    // ===================== FRIENDS LISTING =====================

    @GetMapping("/me/friends")
    @RequireAuth(selfOnly = true) // Xem danh sách bạn bè của mình
    public ResponseEntity<FriendsDTO> getMyFriends() {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        return ResponseEntity.ok(userService.getFriends(currentUserId));
    }

    @GetMapping("/me/mutual-friends/{otherUserId}")
    @RequireAuth(selfOnly = true) // Xem bạn chung với người khác
    public ResponseEntity<FriendsDTO> getMyMutualFriends(@PathVariable String otherUserId) {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        return ResponseEntity.ok(userService.getMutualFriends(currentUserId, otherUserId));
    }

    // ===================== FRIEND SUGGESTIONS =====================
    @GetMapping("/me/friend-suggestions")
    @RequireAuth(selfOnly = true) // Xem gợi ý kết bạn
    public ResponseEntity<FriendsDTO> getMyFriendSuggestions() {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        return ResponseEntity.ok(userService.getFriendSuggestions(currentUserId));
    }

    // ===================== USER FILTERING =====================
    @PostMapping("/me/filter-relationships")
    @RequireAuth(selfOnly = true) // Lọc người dùng theo tiêu chí
    public ResponseEntity<List<UserDTO>> filterMyRelationships(@RequestBody RelationshipFilterDTO filters) {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        return ResponseEntity.ok(userService.getUsersByRelationshipFilters(currentUserId, filters));
    }

    // ===================== ADMIN ENDPOINTS =====================

    @GetMapping("/admin/all")
    @RequireAuth(roles = {"ADMIN"}) // Chỉ admin mới có thể xem tất cả users
    public ResponseEntity<List<UserDTO>> getAllUsers() {
        return ResponseEntity.ok(userService.getAllUsers());
    }

    @DeleteMapping("/admin/{userId}")
    @RequireAuth(roles = {"ADMIN"}) // Chỉ admin mới có thể xóa user
    public ResponseEntity<String> deleteUser(@PathVariable String userId) {
        userService.deleteUser(userId);
        return ResponseEntity.ok("User deleted successfully");
    }

    @PutMapping("/{userId}/profile")
    @RequireAuth(roles = {"ADMIN"}) // Chỉ admin mới có thể cập nhật profile của user khác
    public ResponseEntity<UserDTO> updateUserProfile(@PathVariable String userId, @RequestBody UserDTO userDTO) {
        return ResponseEntity.ok(userService.updateUserProfile(userId, userDTO));
    }

    @PostMapping("/{userId}/filter-relationships")
    @RequireAuth(roles = {"ADMIN"}) // Chỉ admin mới có thể lọc người dùng theo tiêu chí
    public ResponseEntity<List<UserDTO>> filterRelationships(
            @PathVariable String userId,
            @RequestBody RelationshipFilterDTO filters) {
        return ResponseEntity.ok(userService.getUsersByRelationshipFilters(userId, filters));
    }

    @GetMapping("/{userId}/friend-suggestions")
    @RequireAuth(roles = {"ADMIN"}) // Chỉ admin mới có thể xem gợi ý kết bạn của người khác
    public ResponseEntity<FriendsDTO> getFriendSuggestions(@PathVariable String userId) {
        return ResponseEntity.ok(userService.getFriendSuggestions(userId));
    }

    @GetMapping("/{userId}/mutual-friends/{otherUserId}")
    @RequireAuth(roles = {"ADMIN"}) // Chỉ admin mới có thể xem bạn chung của người khác
    public ResponseEntity<FriendsDTO> getMutualFriends(
            @PathVariable String userId,
            @PathVariable String otherUserId) {
        return ResponseEntity.ok(userService.getMutualFriends(userId, otherUserId));
    }

    @GetMapping("/{userId}/friends")
    @RequireAuth(roles = {"ADMIN"}) // Chỉ admin mới có thể xem danh sách bạn bè của người khác
    public ResponseEntity<FriendsDTO> getFriends(@PathVariable String userId) {
        return ResponseEntity.ok(userService.getFriends(userId));
    }

    @GetMapping("/{userId}/friend-requested")
    @RequireAuth(roles = {"ADMIN"}) // Chỉ admin mới có thể xem lời mời kết bạn đã gửi của người khác
    public ResponseEntity<List<UserDTO>> getFriendRequested(@PathVariable String userId) {
        return ResponseEntity.ok(userService.getFriendRequested(userId));
    }

    @GetMapping("/{userId}/friend-requests")
    @RequireAuth(roles = {"ADMIN"}) // Chỉ admin mới có thể xem lời mời kết bạn nhận được của người khác
    public ResponseEntity<List<UserDTO>> getFriendRequests(@PathVariable String userId) {
        return ResponseEntity.ok(userService.getFriendRequests(userId));
    }

    @DeleteMapping("/{userId}/friends/{friendId}")
    @RequireAuth(roles = {"ADMIN"}) // Chỉ admin mới có thể hủy kết bạn của người khác
    public ResponseEntity<String> removeFriend(@PathVariable String userId, @PathVariable String friendId) {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();

        // Chỉ cho phép hủy kết bạn từ chính tài khoản của mình
        if (!userId.equals(currentUserId)) {
            return ResponseEntity.status(403).body("Can only remove friends from your own account");
        }

        userService.removeFriend(userId, friendId);
        return ResponseEntity.ok("Friend removed successfully");
    }

    @PostMapping("/{userId}/accept-invite/{friendId}")
    @RequireAuth(roles = {"ADMIN"})  // Chỉ admin mới có thể chấp nhận lời mời kết bạn của người khác
    public ResponseEntity<String> acceptFriendRequest(@PathVariable String userId, @PathVariable String friendId) {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();

        // Chỉ cho phép chấp nhận lời mời gửi đến chính mình
        if (!userId.equals(currentUserId)) {
            return ResponseEntity.status(403).body("Can only accept friend requests sent to your own account");
        }

        userService.acceptFriendInvite(userId, friendId);
        return ResponseEntity.ok("Friend request accepted successfully");
    }
}
