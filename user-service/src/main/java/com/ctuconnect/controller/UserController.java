package com.ctuconnect.controller;

import com.ctuconnect.dto.FriendsDTO;
import com.ctuconnect.dto.RelationshipFilterDTO;
import com.ctuconnect.dto.UserDTO;
import com.ctuconnect.security.SecurityContextHolder;
import com.ctuconnect.security.annotation.RequireAuth;
import com.ctuconnect.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Objects;

@RestController
@RequestMapping("/api/users")
@RequireAuth // Yêu cầu xác thực cho tất cả endpoints
public class UserController {

    @Autowired
    private UserService userService;

    @PostMapping("/register")
    @RequireAuth(roles = {"ADMIN"}) // Chỉ admin mới có thể tạo user mới
    public ResponseEntity<UserDTO> createUser(@RequestBody UserDTO userDTO) {
        return ResponseEntity.ok(userService.createUser(userDTO));
    }

    @GetMapping("/{userId}/profile")
    @RequireAuth// User chỉ có thể xem profile của chính mình hoặc admin có thể xem tất cả
    public ResponseEntity<UserDTO> getUserProfile(@PathVariable String userId) {
        return ResponseEntity.ok(userService.getUserProfile(userId));
    }

    @GetMapping("/me/profile")
    @RequireAuth(selfOnly = true) // Endpoint mới để user xem profile của chính mình
    public ResponseEntity<UserDTO> getMyProfile() {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        return ResponseEntity.ok(userService.getUserProfile(currentUserId));
    }

    @PutMapping("/{userId}/profile")
    @RequireAuth // User chỉ có thể cập nhật profile của chính mình
    public ResponseEntity<UserDTO> updateUserProfile(@PathVariable String userId, @RequestBody UserDTO userDTO) {
        return ResponseEntity.ok(userService.updateUserProfile(userId, userDTO));
    }

    @PutMapping("/me/profile")
    @RequireAuth(selfOnly = true) // Endpoint mới để user cập nhật profile của chính mình
    public ResponseEntity<UserDTO> updateMyProfile(@RequestBody UserDTO userDTO) {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        return ResponseEntity.ok(userService.updateUserProfile(currentUserId, userDTO));
    }

    // Friend request management
    @PostMapping("/{userId}/invite/{friendId}")
    @RequireAuth(selfOnly = true) // User chỉ có thể gửi lời mời từ chính tài khoản của mình
    public ResponseEntity<String> addFriend(@PathVariable String userId, @PathVariable String friendId) {
        userService.addFriend(userId, friendId);
        return ResponseEntity.ok("Friend request sent successfully");
    }

    @PostMapping("/me/invite/{friendId}")
    @RequireAuth(selfOnly = true) // Endpoint mới để user gửi lời mời kết bạn
    public ResponseEntity<String> sendFriendRequest(@PathVariable String friendId) {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        userService.addFriend(currentUserId, friendId);
        return ResponseEntity.ok("Friend request sent successfully");
    }

    @PostMapping("/{userId}/accept-invite/{friendId}")
    @RequireAuth(selfOnly = true) // User chỉ có thể chấp nhận lời mời gửi đến mình
    public ResponseEntity<String> acceptFriendInvite(@PathVariable String userId, @PathVariable String friendId) {
        userService.acceptFriendInvite(userId, friendId);
        return ResponseEntity.ok("Friend request accepted successfully");
    }

    @PostMapping("/me/accept-invite/{friendId}")
    @RequireAuth(selfOnly = true) // Endpoint mới để user chấp nhận lời mời kết bạn
    public ResponseEntity<String> acceptMyFriendInvite(@PathVariable String friendId) {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        userService.acceptFriendInvite(currentUserId, friendId);
        return ResponseEntity.ok("Friend request accepted successfully");
    }

    @PostMapping("/{userId}/reject-invite/{friendId}")
    @RequireAuth(selfOnly = true) // User chỉ có thể từ chối lời mời gửi đến mình
    public ResponseEntity<String> rejectFriendInvite(@PathVariable String userId, @PathVariable String friendId) {
        userService.rejectFriendInvite(userId, friendId);
        return ResponseEntity.ok("Friend request rejected successfully");
    }

    @PostMapping("/me/reject-invite/{friendId}")
    @RequireAuth(selfOnly = true) // Endpoint mới để user từ chối lời mời kết bạn
    public ResponseEntity<String> rejectMyFriendInvite(@PathVariable String friendId) {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        userService.rejectFriendInvite(currentUserId, friendId);
        return ResponseEntity.ok("Friend request rejected successfully");
    }

    // Friend listing and suggestions
    @GetMapping("/{userId}/friends")
    @RequireAuth // User chỉ có thể xem danh sách bạn bè của mình
    public ResponseEntity<FriendsDTO> getFriends(@PathVariable String userId) {
        return ResponseEntity.ok(userService.getFriends(userId));
    }

    @GetMapping("/me/friends")
    @RequireAuth(selfOnly = true) // Endpoint mới để user xem danh sách bạn bè của mình
    public ResponseEntity<FriendsDTO> getMyFriends() {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        return ResponseEntity.ok(userService.getFriends(currentUserId));
    }

    @GetMapping("/{userId}/mutual-friends/{otherUserId}")
    @RequireAuth// User chỉ có thể xem bạn chung với người khác từ tài khoản của mình
    public ResponseEntity<FriendsDTO> getMutualFriends(
            @PathVariable String userId,
            @PathVariable String otherUserId) {
        return ResponseEntity.ok(userService.getMutualFriends(userId, otherUserId));
    }

    @GetMapping("/me/mutual-friends/{otherUserId}")
    @RequireAuth(selfOnly = true) // Endpoint mới để user xem bạn chung với người khác
    public ResponseEntity<FriendsDTO> getMyMutualFriends(@PathVariable String otherUserId) {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        return ResponseEntity.ok(userService.getMutualFriends(currentUserId, otherUserId));
    }

    @GetMapping("/{userId}/friend-suggestions")
    @RequireAuth // User chỉ có thể xem gợi ý kết bạn cho mình
    public ResponseEntity<FriendsDTO> getFriendSuggestions(@PathVariable String userId) {
        return ResponseEntity.ok(userService.getFriendSuggestions(userId));
    }

    @GetMapping("/me/friend-suggestions")
    @RequireAuth(selfOnly = true) // Endpoint mới để user xem gợi ý kết bạn
    public ResponseEntity<FriendsDTO> getMyFriendSuggestions() {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        return ResponseEntity.ok(userService.getFriendSuggestions(currentUserId));
    }

    /**
     * Filter users by relationship criteria
     */
    @PostMapping("/{userId}/filter-relationships")
    @RequireAuth // User chỉ có thể filter từ tài khoản của mình
    public ResponseEntity<List<UserDTO>> filterRelationships(
            @PathVariable String userId,
            @RequestBody RelationshipFilterDTO filters) {
        return ResponseEntity.ok(userService.getUsersByRelationshipFilters(userId, filters));
    }

    @PostMapping("/me/filter-relationships")
    @RequireAuth(selfOnly = true) // Endpoint mới để user filter relationships
    public ResponseEntity<List<UserDTO>> filterMyRelationships(@RequestBody RelationshipFilterDTO filters) {
        String currentUserId = SecurityContextHolder.getCurrentUserIdOrThrow();
        return ResponseEntity.ok(userService.getUsersByRelationshipFilters(currentUserId, filters));
    }

    // Admin endpoints
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
}