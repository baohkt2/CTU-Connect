package com.ctuconnect.controller;

import com.ctuconnect.dto.*;
import com.ctuconnect.service.SocialGraphService;
import com.ctuconnect.service.UserService;
import com.ctuconnect.security.annotation.RequireAuth;
import com.ctuconnect.security.AuthenticatedUser;
import com.ctuconnect.security.SecurityContextHolder;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import jakarta.validation.Valid;

import java.util.List;
import java.util.Map;
import java.util.Set;

@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
@Slf4j
public class EnhancedUserController {

    private final UserService userService;
    private final SocialGraphService socialGraphService;

    // ==================== PROFILE ENDPOINTS ====================
    
    /**
     * Get current user's profile (alias for /me/profile)
     * Frontend expects: GET /api/users/profile
     */
    @GetMapping("/profile")
    @RequireAuth
    public ResponseEntity<UserProfileDTO> getCurrentUserProfile() {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }
        
        log.info("GET /profile - Getting profile for user: {}", currentUser.getEmail());
        UserProfileDTO profile = userService.getUserProfile(currentUser.getId());
        log.info("GET /profile - Returning profile with avatarUrl: {}, backgroundUrl: {}", 
                profile.getAvatarUrl(), profile.getBackgroundUrl());
        return ResponseEntity.ok(profile);
    }
    
    /**
     * Get current user's profile (alternative path)
     */
    @GetMapping("/me/profile")
    @RequireAuth
    public ResponseEntity<UserProfileDTO> getMyProfile() {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }
        
        log.info("GET /me/profile - Getting profile for user: {}", currentUser.getEmail());
        UserProfileDTO profile = userService.getUserProfile(currentUser.getId());
        log.info("GET /me/profile - Returning profile with avatarUrl: {}, backgroundUrl: {}", 
                profile.getAvatarUrl(), profile.getBackgroundUrl());
        return ResponseEntity.ok(profile);
    }
    
    /**
     * Update current user's profile
     * Frontend expects: PUT /api/users/profile
     */
    @PutMapping("/me/profile")
    @RequireAuth
    public ResponseEntity<UserProfileDTO> updateCurrentUserProfile(
            @RequestBody @Valid UserUpdateDTO updateDTO) {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }
        
        log.info("PUT /profile - Updating profile for user: {}", currentUser.getEmail());
        UserProfileDTO updatedProfile = userService.updateUserProfile(currentUser.getId(), updateDTO);
        return ResponseEntity.ok(updatedProfile);
    }

    // ==================== PROFILE COMPLETION CHECK ====================
    
    /**
     * Check if current user's profile is complete
     * Frontend expects: GET /api/users/checkMyInfo
     * Returns true if profile has all required fields filled
     */
    @GetMapping("/checkMyInfo")
    @RequireAuth
    public ResponseEntity<Boolean> checkProfileCompletion() {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        log.info("GET /checkMyInfo - Checking profile completion for user: {}", currentUser.getEmail());
        
        try {
            UserProfileDTO profile = userService.getUserProfile(currentUser.getId());
            
            // Admin users don't need to complete profile
            if ("ADMIN".equals(profile.getRole())) {
                log.info("User {} is ADMIN, profile completion not required", currentUser.getEmail());
                return ResponseEntity.ok(true);
            }
            
            // For students: check required fields
            boolean isComplete = profile.getFullName() != null && !profile.getFullName().trim().isEmpty()
                    && profile.getStudentId() != null && !profile.getStudentId().trim().isEmpty()
                    && profile.getMajor() != null && !profile.getMajor().trim().isEmpty()
                    && profile.getBatch() != null && !profile.getBatch().trim().isEmpty()
                    && profile.getGender() != null && !profile.getGender().trim().isEmpty();
            
            log.info("Profile completion check for user {}: {}", currentUser.getEmail(), isComplete);
            log.info("Profile details - fullName: {}, studentId: {}, major: {}, batch: {}, gender: {}", 
                    profile.getFullName() != null, 
                    profile.getStudentId() != null, 
                    profile.getMajor() != null,
                    profile.getBatch() != null,
                    profile.getGender() != null);
            
            return ResponseEntity.ok(isComplete);
        } catch (Exception e) {
            log.error("Error checking profile completion for user {}: {}", currentUser.getEmail(), e.getMessage());
            // If there's an error, assume profile is not complete to be safe
            return ResponseEntity.ok(false);
        }
    }

    // ==================== FRIEND SUGGESTIONS ====================
    
    /**
     * Get friend suggestions
     * Frontend expects: GET /api/users/friend-suggestions
     */
    @GetMapping("/friend-suggestions")
    @RequireAuth
    public ResponseEntity<List<FriendSuggestionDTO>> getFriendSuggestions(
            @RequestParam(defaultValue = "20") int limit) {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        List<FriendSuggestionDTO> suggestions = socialGraphService.getFriendSuggestions(
            currentUser.getId(), limit);

        return ResponseEntity.ok(suggestions);
    }

    // ==================== FRIEND MANAGEMENT ====================
    
    /**
     * Send friend request
     * Frontend expects: POST /api/users/:id/friend-request
     */
    @PostMapping("/{targetUserId}/friend-request")
    @RequireAuth
    public ResponseEntity<Map<String, String>> sendFriendRequest(
            @PathVariable String targetUserId) {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        log.info("POST /{}/friend-request - Sending friend request", targetUserId);
        userService.sendFriendRequest(currentUser.getId(), targetUserId);

        // Invalidate friend suggestions cache for both users
        socialGraphService.invalidateFriendSuggestionsCache(currentUser.getId());
        socialGraphService.invalidateFriendSuggestionsCache(targetUserId);

        return ResponseEntity.ok(Map.of("message", "Friend request sent successfully"));
    }

    /**
     * Accept friend request
     * Frontend expects: POST /api/users/:id/accept-friend
     */
    @PostMapping("/{requesterId}/accept-friend")
    @RequireAuth
    public ResponseEntity<Map<String, String>> acceptFriendRequest(
            @PathVariable String requesterId) {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        log.info("POST /{}/accept-friend - Accepting friend request", requesterId);
        userService.acceptFriendRequest(requesterId, currentUser.getId());

        // Invalidate caches for both users
        socialGraphService.invalidateFriendSuggestionsCache(currentUser.getId());
        socialGraphService.invalidateFriendSuggestionsCache(requesterId);

        return ResponseEntity.ok(Map.of("message", "Friend request accepted successfully"));
    }
    
    /**
     * Reject friend request
     */
    @PostMapping("/{requesterId}/reject-friend")
    @RequireAuth
    public ResponseEntity<Map<String, String>> rejectFriendRequest(
            @PathVariable String requesterId) {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        log.info("POST /{}/reject-friend - Rejecting friend request", requesterId);
        userService.rejectFriendRequest(requesterId, currentUser.getId());

        return ResponseEntity.ok(Map.of("message", "Friend request rejected successfully"));
    }
    
    /**
     * Unfriend/Remove friend
     */
    @DeleteMapping("/{friendId}/friend")
    @RequireAuth
    public ResponseEntity<Map<String, String>> removeFriend(
            @PathVariable String friendId) {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        log.info("DELETE /{}/friend - Removing friend", friendId);
        userService.removeFriend(currentUser.getId(), friendId);

        return ResponseEntity.ok(Map.of("message", "Friend removed successfully"));
    }
    
    /**
     * Cancel sent friend request
     */
    @DeleteMapping("/{targetUserId}/friend-request")
    @RequireAuth
    public ResponseEntity<Map<String, String>> cancelFriendRequest(
            @PathVariable String targetUserId) {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        log.info("DELETE /{}/friend-request - Canceling friend request", targetUserId);
        userService.rejectFriendRequest(currentUser.getId(), targetUserId);

        return ResponseEntity.ok(Map.of("message", "Friend request cancelled successfully"));
    }

    /**
     * Get sent friend requests
     */
    @GetMapping("/sent-requests")
    @RequireAuth
    public ResponseEntity<List<FriendRequestDTO>> getSentFriendRequests() {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        List<FriendRequestDTO> sentRequests = userService.getSentFriendRequests(currentUser.getId());
        return ResponseEntity.ok(sentRequests);
    }

    /**
     * Get received friend requests
     */
    @GetMapping("/received-requests")
    @RequireAuth
    public ResponseEntity<List<FriendRequestDTO>> getReceivedFriendRequests() {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        List<FriendRequestDTO> receivedRequests = userService.getReceivedFriendRequests(currentUser.getId());
        return ResponseEntity.ok(receivedRequests);
    }

    // ==================== MUTUAL FRIENDS ====================
    
    /**
     * Get mutual friends count
     * Frontend expects: GET /api/users/:id/mutual-friends-count
     */
    @GetMapping("/{targetUserId}/mutual-friends-count")
    @RequireAuth
    public ResponseEntity<Map<String, Integer>> getMutualFriendsCount(
            @PathVariable String targetUserId) {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        int count = socialGraphService.getMutualFriendsCount(currentUser.getId(), targetUserId);
        return ResponseEntity.ok(Map.of("count", count));
    }

    // ==================== USER SEARCH ====================
    
    /**
     * Enhanced user search
     * Frontend expects: GET /api/users/search
     */
    @GetMapping("/search")
    @RequireAuth
    public ResponseEntity<List<UserDTO>> searchUsers(
            @RequestParam String query,
            @RequestParam(required = false) String faculty,
            @RequestParam(required = false) String major,
            @RequestParam(required = false) String batch,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        List<UserDTO> users = userService.searchUsersWithContext(
            query, faculty, major, batch, currentUser.getId(), page, size);

        return ResponseEntity.ok(users);
    }

    // ==================== TIMELINE & ACTIVITIES ====================
    
    /**
     * Get user timeline (posts by user)
     * Frontend expects: GET /api/users/:id/timeline
     */
    @GetMapping("/{userId}/timeline")
    @RequireAuth
    public ResponseEntity<Map<String, String>> getUserTimeline(
            @PathVariable String userId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        log.info("GET /{}/timeline - This endpoint should query post-service", userId);
        // Note: This should actually call post-service to get user's posts
        // For now, return a message indicating to use post-service endpoint instead
        return ResponseEntity.ok(Map.of(
            "message", "Please use /api/posts/timeline/" + userId + " endpoint from post-service",
            "userId", userId
        ));
    }
    
    /**
     * Get user activities
     * Frontend expects: GET /api/users/:id/activities
     */
    @GetMapping("/{userId}/activities")
    @RequireAuth
    public ResponseEntity<List<ActivityDTO>> getUserActivities(
            @PathVariable String userId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        List<ActivityDTO> activities = userService.getUserActivity(
            userId, currentUser.getId(), page, size);

        return ResponseEntity.ok(activities);
    }

    // ==================== INTERNAL SERVICE ENDPOINTS ====================
    
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
}
