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

    /**
     * Search friend suggestions with filters (enhanced version)
     * Frontend expects: GET /api/users/friend-suggestions/search
     * Supports: query (fullname/email), college, faculty, batch filters
     */
    @GetMapping("/friend-suggestions/search")
    @RequireAuth
    public ResponseEntity<List<UserSearchDTO>> searchFriendSuggestions(
            @RequestParam(required = false) String query,
            @RequestParam(required = false) String college,
            @RequestParam(required = false) String faculty,
            @RequestParam(required = false) String batch,
            @RequestParam(defaultValue = "50") int limit) {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        log.info("GET /friend-suggestions/search - query={}, college={}, faculty={}, batch={}, limit={}", 
                 query, college, faculty, batch, limit);

        List<UserSearchDTO> suggestions = userService.searchFriendSuggestions(
            currentUser.getId(), query, college, faculty, batch, limit);

        return ResponseEntity.ok(suggestions);
    }

    // ==================== FRIEND MANAGEMENT ====================
    
    /**
     * Get current user's friends list
     * Frontend expects: GET /api/users/me/friends
     */
    @GetMapping("/me/friends")
    @RequireAuth
    public ResponseEntity<PageResponse<UserSearchDTO>> getMyFriends(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        log.info("GET /me/friends - Getting friends for user: {} (page={}, size={})", 
                 currentUser.getEmail(), page, size);
        
        try {
            Page<UserSearchDTO> friendsPage = userService.getFriends(
                currentUser.getId(), PageRequest.of(page, size));
            
            log.info("GET /me/friends - Successfully retrieved {} friends (total: {})", 
                     friendsPage.getNumberOfElements(), friendsPage.getTotalElements());
            
            // Convert Page to PageResponse to avoid serialization issues
            PageResponse<UserSearchDTO> response = PageResponse.of(friendsPage);
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            log.error("GET /me/friends - Error getting friends: {}", e.getMessage(), e);
            throw e;
        }
    }

    /**
     * Get RECEIVED friend requests ONLY
     * Frontend expects: GET /api/users/me/friend-requests
     * Returns only requests that OTHER users sent TO current user
     */
    @GetMapping("/me/friend-requests")
    @RequireAuth
    public ResponseEntity<List<FriendRequestDTO>> getMyFriendRequests() {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        log.info("GET /me/friend-requests - Getting RECEIVED requests for user: {}", currentUser.getEmail());
        List<FriendRequestDTO> receivedRequests = userService.getReceivedFriendRequests(currentUser.getId());
        return ResponseEntity.ok(receivedRequests);
    }

    /**
     * Get SENT friend requests ONLY
     * Frontend expects: GET /api/users/me/friend-requested  
     * Returns only requests that current user sent TO other users
     */
    @GetMapping("/me/friend-requested")
    @RequireAuth
    public ResponseEntity<List<FriendRequestDTO>> getMySentFriendRequests() {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        log.info("GET /me/friend-requested - Getting SENT requests for user: {}", currentUser.getEmail());
        List<FriendRequestDTO> sentRequests = userService.getSentFriendRequests(currentUser.getId());
        return ResponseEntity.ok(sentRequests);
    }

    /**
     * Get ALL friend requests (both sent and received) - Optional endpoint
     * Frontend can use: GET /api/users/me/friend-requests/all
     * Returns combined list with requestType to distinguish
     */
    @GetMapping("/me/friend-requests/all")
    @RequireAuth
    public ResponseEntity<List<FriendRequestDTO>> getAllMyFriendRequests() {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        log.info("GET /me/friend-requests/all - Getting ALL requests for user: {}", currentUser.getEmail());
        List<FriendRequestDTO> allRequests = userService.getAllFriendRequests(currentUser.getId());
        return ResponseEntity.ok(allRequests);
    }

    /**
     * Send friend request
     * Frontend expects: POST /api/users/me/invite/{friendId}
     */
    @PostMapping("/me/invite/{friendId}")
    @RequireAuth
    public ResponseEntity<Map<String, String>> sendMyFriendRequest(
            @PathVariable String friendId) {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        log.info("POST /me/invite/{} - Sending friend request", friendId);
        userService.sendFriendRequest(currentUser.getId(), friendId);

        // Invalidate friend suggestions cache for both users
        socialGraphService.invalidateFriendSuggestionsCache(currentUser.getId());
        socialGraphService.invalidateFriendSuggestionsCache(friendId);

        return ResponseEntity.ok(Map.of("message", "Friend request sent successfully"));
    }

    /**
     * Accept friend request
     * Frontend expects: POST /api/users/me/accept-invite/{friendId}
     */
    @PostMapping("/me/accept-invite/{friendId}")
    @RequireAuth
    public ResponseEntity<Map<String, String>> acceptMyFriendRequest(
            @PathVariable String friendId) {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        log.info("POST /me/accept-invite/{} - Accepting friend request", friendId);
        userService.acceptFriendRequest(friendId, currentUser.getId());

        // Invalidate caches for both users
        socialGraphService.invalidateFriendSuggestionsCache(currentUser.getId());
        socialGraphService.invalidateFriendSuggestionsCache(friendId);

        return ResponseEntity.ok(Map.of("message", "Friend request accepted successfully"));
    }

    /**
     * Reject friend request
     * Frontend expects: POST /api/users/me/reject-invite/{friendId}
     */
    @PostMapping("/me/reject-invite/{friendId}")
    @RequireAuth
    public ResponseEntity<Map<String, String>> rejectMyFriendRequest(
            @PathVariable String friendId) {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        log.info("POST /me/reject-invite/{} - Rejecting friend request", friendId);
        userService.rejectFriendRequest(friendId, currentUser.getId());

        return ResponseEntity.ok(Map.of("message", "Friend request rejected successfully"));
    }

    /**
     * Remove friend
     * Frontend expects: DELETE /api/users/me/friends/{friendId}
     */
    @DeleteMapping("/me/friends/{friendId}")
    @RequireAuth
    public ResponseEntity<Map<String, String>> removeMyFriend(
            @PathVariable String friendId) {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        log.info("DELETE /me/friends/{} - Removing friend", friendId);
        userService.removeFriend(currentUser.getId(), friendId);

        return ResponseEntity.ok(Map.of("message", "Friend removed successfully"));
    }
    
    /**
     * Send friend request (alternative path)
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
     * Accept friend request (alternative path)
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
     * Reject friend request (alternative path)
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
     * Unfriend/Remove friend (alternative path)
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
     * Cancel sent friend request (alternative path)
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
     * Get sent friend requests (alternative path)
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
     * Get received friend requests (alternative path)
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

    /**
     * Get mutual friends list (paginated)
     * Frontend expects: GET /api/users/:id/mutual-friends
     */
    @GetMapping("/{targetUserId}/mutual-friends")
    @RequireAuth
    public ResponseEntity<PageResponse<UserSearchDTO>> getMutualFriendsList(
            @PathVariable String targetUserId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        log.info("GET /{}/mutual-friends - Getting mutual friends", targetUserId);
        
        try {
            Page<UserSearchDTO> mutualFriendsPage = userService.getMutualFriendsList(
                currentUser.getId(), targetUserId, PageRequest.of(page, size));
            
            log.info("GET /{}/mutual-friends - Successfully retrieved {} mutual friends (total: {})",
                     targetUserId, mutualFriendsPage.getNumberOfElements(), mutualFriendsPage.getTotalElements());
            
            // Convert Page to PageResponse to avoid serialization issues
            PageResponse<UserSearchDTO> response = PageResponse.of(mutualFriendsPage);
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            log.error("GET /{}/mutual-friends - Error getting mutual friends: {}", targetUserId, e.getMessage(), e);
            throw e;
        }
    }

    /**
     * Get friendship status with target user
     * Frontend expects: GET /api/users/:id/friendship-status
     * Returns: {"status": "none" | "friends" | "sent" | "received" | "self"}
     */
    @GetMapping("/{targetUserId}/friendship-status")
    @RequireAuth
    public ResponseEntity<Map<String, String>> getFriendshipStatus(
            @PathVariable String targetUserId) {
        AuthenticatedUser currentUser = SecurityContextHolder.getAuthenticatedUser();
        if (currentUser == null) {
            throw new SecurityException("No authenticated user found");
        }

        log.info("GET /{}/friendship-status - Getting friendship status", targetUserId);
        String status = userService.getFriendshipStatus(currentUser.getId(), targetUserId);

        return ResponseEntity.ok(Map.of("status", status));
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
