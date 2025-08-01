package com.ctuconnect.controller;

import com.ctuconnect.dto.ActivityDTO;
import com.ctuconnect.dto.FriendSuggestionDTO;
import com.ctuconnect.dto.UserDTO;
import com.ctuconnect.service.SocialGraphService;
import com.ctuconnect.service.UserService;
import com.ctuconnect.security.annotation.RequireAuth;
import com.ctuconnect.security.AuthenticatedUser;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Set;

@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
public class EnhancedUserController {

    private final UserService userService;
    private final SocialGraphService socialGraphService;

    /**
     * Get Facebook-like friend suggestions
     */
    @GetMapping("/friend-suggestions")
    @RequireAuth
    public ResponseEntity<List<FriendSuggestionDTO>> getFriendSuggestions(
            @RequestParam(defaultValue = "20") int limit,
            AuthenticatedUser user) {

        List<FriendSuggestionDTO> suggestions = socialGraphService.getFriendSuggestions(
            user.getId(), limit);

        return ResponseEntity.ok(suggestions);
    }

    /**
     * Get mutual friends count between current user and target user
     */
    @GetMapping("/{targetUserId}/mutual-friends-count")
    @RequireAuth
    public ResponseEntity<Integer> getMutualFriendsCount(
            @PathVariable String targetUserId,
            AuthenticatedUser user) {

        int count = socialGraphService.getMutualFriendsCount(user.getId(), targetUserId);
        return ResponseEntity.ok(count);
    }

    /**
     * Get user's friend IDs (for internal service communication)
     */
    @GetMapping("/{userId}/friends/ids")
    public ResponseEntity<Set<String>> getFriendIds(@PathVariable String userId) {
        Set<String> friendIds = userService.getFriendIds(userId);
        return ResponseEntity.ok(friendIds);
    }

    /**
     * Get users with close interactions (for feed ranking)
     */
    @GetMapping("/{userId}/close-interactions")
    public ResponseEntity<Set<String>> getCloseInteractionIds(@PathVariable String userId) {
        Set<String> closeInteractionIds = userService.getCloseInteractionIds(userId);
        return ResponseEntity.ok(closeInteractionIds);
    }

    /**
     * Get users from same faculty
     */
    @GetMapping("/{userId}/same-faculty")
    public ResponseEntity<Set<String>> getSameFacultyUserIds(@PathVariable String userId) {
        Set<String> sameFacultyIds = userService.getSameFacultyUserIds(userId);
        return ResponseEntity.ok(sameFacultyIds);
    }

    /**
     * Get users from same major
     */
    @GetMapping("/{userId}/same-major")
    public ResponseEntity<Set<String>> getSameMajorUserIds(@PathVariable String userId) {
        Set<String> sameMajorIds = userService.getSameMajorUserIds(userId);
        return ResponseEntity.ok(sameMajorIds);
    }

    /**
     * Get user's interest tags
     */
    @GetMapping("/{userId}/interest-tags")
    public ResponseEntity<Set<String>> getUserInterestTags(@PathVariable String userId) {
        Set<String> interestTags = userService.getUserInterestTags(userId);
        return ResponseEntity.ok(interestTags);
    }

    /**
     * Get user's preferred categories
     */
    @GetMapping("/{userId}/preferred-categories")
    public ResponseEntity<Set<String>> getUserPreferredCategories(@PathVariable String userId) {
        Set<String> preferredCategories = userService.getUserPreferredCategories(userId);
        return ResponseEntity.ok(preferredCategories);
    }

    /**
     * Get user's faculty ID
     */
    @GetMapping("/{userId}/faculty-id")
    public ResponseEntity<String> getUserFacultyId(@PathVariable String userId) {
        String facultyId = userService.getUserFacultyId(userId);
        return ResponseEntity.ok(facultyId);
    }

    /**
     * Get user's major ID
     */
    @GetMapping("/{userId}/major-id")
    public ResponseEntity<String> getUserMajorId(@PathVariable String userId) {
        String majorId = userService.getUserMajorId(userId);
        return ResponseEntity.ok(majorId);
    }

    /**
     * Enhanced user search with academic context
     */
    @GetMapping("/search")
    @RequireAuth
    public ResponseEntity<List<UserDTO>> searchUsers(
            @RequestParam String query,
            @RequestParam(required = false) String faculty,
            @RequestParam(required = false) String major,
            @RequestParam(required = false) String batch,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size,
            AuthenticatedUser user) {

        List<UserDTO> users = userService.searchUsersWithContext(
            query, faculty, major, batch, user.getId(), page, size);

        return ResponseEntity.ok(users);
    }

    /**
     * Send friend request
     */
    @PostMapping("/{targetUserId}/friend-request")
    @RequireAuth
    public ResponseEntity<Void> sendFriendRequest(
            @PathVariable String targetUserId,
            AuthenticatedUser user) {

        userService.addFriend(user.getId(), targetUserId);

        // Invalidate friend suggestions cache for both users
        socialGraphService.invalidateFriendSuggestionsCache(user.getId());
        socialGraphService.invalidateFriendSuggestionsCache(targetUserId);

        return ResponseEntity.ok().build();
    }

    /**
     * Accept friend request
     */
    @PostMapping("/{requesterId}/accept-friend")
    @RequireAuth
    public ResponseEntity<Void> acceptFriendRequest(
            @PathVariable String requesterId,
            AuthenticatedUser user) {

        userService.acceptFriendInvite(requesterId, user.getId());

        // Invalidate caches for both users
        socialGraphService.invalidateFriendSuggestionsCache(user.getId());
        socialGraphService.invalidateFriendSuggestionsCache(requesterId);

        return ResponseEntity.ok().build();
    }

    /**
     * Get user activity feed (for profile timeline)
     */
    @GetMapping("/{userId}/activity")
    @RequireAuth
    public ResponseEntity<List<ActivityDTO>> getUserActivity(
            @PathVariable String userId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size,
            AuthenticatedUser viewer) {

        List<ActivityDTO> activities = userService.getUserActivity(
            userId, viewer.getId(), page, size);

        return ResponseEntity.ok(activities);
    }
}
