package com.ctuconnect.controller;

import com.ctuconnect.dto.FriendsDTO;
import com.ctuconnect.dto.RelationshipFilterDTO;
import com.ctuconnect.dto.UserDTO;
import com.ctuconnect.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/users")
public class UserController {

    @Autowired
    private UserService userService;

    @PostMapping("/register")
    public ResponseEntity<UserDTO> createUser(@RequestBody UserDTO userDTO) {
        return ResponseEntity.ok(userService.createUser(userDTO));
    }

    @GetMapping("/{userId}/profile")
    public ResponseEntity<UserDTO> getUserProfile(@PathVariable String userId) {
        return ResponseEntity.ok(userService.getUserProfile(userId));
    }

    @PutMapping("/{userId}/profile")
    public ResponseEntity<UserDTO> updateUserProfile(@PathVariable String userId, @RequestBody UserDTO userDTO) {
        return ResponseEntity.ok(userService.updateUserProfile(userId, userDTO));
    }

    // Friend request management
    @PostMapping("/{userId}/invite/{friendId}")
    public ResponseEntity<String> addFriend(@PathVariable String userId, @PathVariable String friendId) {
        userService.addFriend(userId, friendId);
        return ResponseEntity.ok("Friend request sent successfully");
    }

    @PostMapping("/{userId}/accept-invite/{friendId}")
    public ResponseEntity<String> acceptFriendInvite(@PathVariable String userId, @PathVariable String friendId) {
        userService.acceptFriendInvite(userId, friendId);
        return ResponseEntity.ok("Friend request accepted successfully");
    }

    @PostMapping("/{userId}/reject-invite/{friendId}")
    public ResponseEntity<String> rejectFriendInvite(@PathVariable String userId, @PathVariable String friendId) {
        userService.rejectFriendInvite(userId, friendId);
        return ResponseEntity.ok("Friend request rejected successfully");
    }

    // Friend listing and suggestions
    @GetMapping("/{userId}/friends")
    public ResponseEntity<FriendsDTO> getFriends(@PathVariable String userId) {
        return ResponseEntity.ok(userService.getFriends(userId));
    }

    @GetMapping("/{userId}/mutual-friends/{otherUserId}")
    public ResponseEntity<FriendsDTO> getMutualFriends(
            @PathVariable String userId,
            @PathVariable String otherUserId) {
        return ResponseEntity.ok(userService.getMutualFriends(userId, otherUserId));
    }

    @GetMapping("/{userId}/friend-suggestions")
    public ResponseEntity<FriendsDTO> getFriendSuggestions(@PathVariable String userId) {
        return ResponseEntity.ok(userService.getFriendSuggestions(userId));
    }

    /**
     * Filter users by relationship criteria
     */
    @PostMapping("/{userId}/filter-relationships")
    public ResponseEntity<List<UserDTO>> filterRelationships(
            @PathVariable String userId,
            @RequestBody RelationshipFilterDTO filters) {
        return ResponseEntity.ok(userService.getUsersByRelationshipFilters(userId, filters));
    }
}