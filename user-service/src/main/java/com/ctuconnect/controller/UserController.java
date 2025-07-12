package com.ctuconnect.controller;

import com.ctuconnect.dto.UserProfileDTO;
import com.ctuconnect.dto.UserUpdateDTO;
import com.ctuconnect.dto.UserSearchDTO;
import com.ctuconnect.dto.FriendRequestDTO;
import com.ctuconnect.service.UserService;
import com.ctuconnect.exception.ErrorResponse;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;

import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;


import java.util.List;
import java.util.Map;
import java.time.LocalDateTime;

@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
@Slf4j
@Validated
@Tag(name = "User Management", description = "APIs for managing user profiles and relationships")
public class UserController {

    private final UserService userService;

    // Health check endpoint for microservice
    @GetMapping("/health")
    public ResponseEntity<Map<String, String>> health() {
        return ResponseEntity.ok(Map.of(
            "status", "UP",
            "service", "user-service",
            "timestamp", LocalDateTime.now().toString()
        ));
    }

    // User Profile Management

    @Operation(summary = "Get user profile by ID",
               description = "Retrieve detailed user profile information including academic and social data")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "User profile retrieved successfully",
                    content = @Content(mediaType = "application/json",
                                     schema = @Schema(implementation = UserProfileDTO.class))),
        @ApiResponse(responseCode = "404", description = "User not found",
                    content = @Content(mediaType = "application/json",
                                     schema = @Schema(implementation = ErrorResponse.class)))
    })
    @GetMapping("/{userId}")
    public ResponseEntity<UserProfileDTO> getUserProfile(
            @Parameter(description = "User ID", required = true)
            @PathVariable @NotBlank String userId) {

        log.info("GET /api/users/{} - Getting user profile", userId);
        UserProfileDTO userProfile = userService.getUserProfile(userId);
        return ResponseEntity.ok(userProfile);
    }

    @Operation(summary = "Get user profile by email",
               description = "Retrieve user profile using email address")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "User profile retrieved successfully"),
        @ApiResponse(responseCode = "404", description = "User not found")
    })
    @GetMapping("/email/{email}")
    public ResponseEntity<UserProfileDTO> getUserProfileByEmail(
            @Parameter(description = "User email", required = true)
            @PathVariable @NotBlank String email) {

        log.info("GET /users/email/{} - Getting user profile by email", email);
        UserProfileDTO userProfile = userService.getUserProfileByEmail(email);
        return ResponseEntity.ok(userProfile);
    }

    @Operation(summary = "Create new user",
               description = "Create a new user from authentication service data")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "201", description = "User created successfully"),
        @ApiResponse(responseCode = "409", description = "User already exists")
    })
    @PostMapping
    public ResponseEntity<Map<String, String>> createUser(
            @RequestBody @Valid Map<String, String> userRequest) {

        String authUserId = userRequest.get("authUserId");
        String email = userRequest.get("email");
        String username = userRequest.get("username");
        String role = userRequest.get("role");

        log.info("POST /users - Creating user with email: {}", email);

        userService.createUser(authUserId, email, username, role);

        return ResponseEntity.status(HttpStatus.CREATED)
                .body(Map.of("message", "User created successfully", "userId", authUserId));
    }

    @Operation(summary = "Update user profile",
               description = "Update user profile information including academic details")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "User profile updated successfully"),
        @ApiResponse(responseCode = "404", description = "User not found"),
        @ApiResponse(responseCode = "400", description = "Invalid input data")
    })
    @PutMapping("/{userId}")
    public ResponseEntity<UserProfileDTO> updateUserProfile(
            @Parameter(description = "User ID", required = true)
            @PathVariable @NotBlank String userId,
            @RequestBody @Valid UserUpdateDTO updateDTO) {

        log.info("PUT /users/{} - Updating user profile", userId);
        UserProfileDTO updatedProfile = userService.updateUserProfile(userId, updateDTO);
        return ResponseEntity.ok(updatedProfile);
    }

    @Operation(summary = "Deactivate user",
               description = "Deactivate user account")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "User deactivated successfully"),
        @ApiResponse(responseCode = "404", description = "User not found")
    })
    @PatchMapping("/{userId}/deactivate")
    public ResponseEntity<Map<String, String>> deactivateUser(
            @Parameter(description = "User ID", required = true)
            @PathVariable @NotBlank String userId) {

        log.info("PATCH /users/{}/deactivate - Deactivating user", userId);
        userService.deactivateUser(userId);
        return ResponseEntity.ok(Map.of("message", "User deactivated successfully"));
    }

    @Operation(summary = "Activate user",
               description = "Activate user account")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "User activated successfully"),
        @ApiResponse(responseCode = "404", description = "User not found")
    })
    @PatchMapping("/{userId}/activate")
    public ResponseEntity<Map<String, String>> activateUser(
            @Parameter(description = "User ID", required = true)
            @PathVariable @NotBlank String userId) {

        log.info("PATCH /users/{}/activate - Activating user", userId);
        userService.activateUser(userId);
        return ResponseEntity.ok(Map.of("message", "User activated successfully"));
    }

    // User Search and Discovery

    @Operation(summary = "Search users",
               description = "Search for users by name, email, username, or student ID")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "Search completed successfully")
    })
    @GetMapping("/search")
    public ResponseEntity<Page<UserSearchDTO>> searchUsers(
            @Parameter(description = "Search term")
            @RequestParam @NotBlank String q,
            @Parameter(description = "Current user ID for relationship context")
            @RequestParam(required = false) String currentUserId,
            @Parameter(description = "Page number (0-based)")
            @RequestParam(defaultValue = "0") @Min(0) int page,
            @Parameter(description = "Page size")
            @RequestParam(defaultValue = "20") @Min(1) @Max(100) int size) {

        log.info("GET /users/search - Searching users with term: {}", q);
        Pageable pageable = PageRequest.of(page, size);
        Page<UserSearchDTO> searchResults = userService.searchUsers(q, currentUserId, pageable);
        return ResponseEntity.ok(searchResults);
    }

    @Operation(summary = "Find users by college",
               description = "Find all users belonging to a specific college")
    @GetMapping("/college/{collegeName}")
    public ResponseEntity<Page<UserSearchDTO>> findUsersByCollege(
            @Parameter(description = "College name", required = true)
            @PathVariable @NotBlank String collegeName,
            @Parameter(description = "Current user ID for relationship context")
            @RequestParam(required = false) String currentUserId,
            @Parameter(description = "Page number (0-based)")
            @RequestParam(defaultValue = "0") @Min(0) int page,
            @Parameter(description = "Page size")
            @RequestParam(defaultValue = "20") @Min(1) @Max(100) int size) {

        log.info("GET /users/college/{} - Finding users by college", collegeName);
        Pageable pageable = PageRequest.of(page, size);
        Page<UserSearchDTO> users = userService.findUsersByCollege(collegeName, currentUserId, pageable);
        return ResponseEntity.ok(users);
    }

    @Operation(summary = "Find users by faculty",
               description = "Find all users belonging to a specific faculty")
    @GetMapping("/faculty/{facultyName}")
    public ResponseEntity<Page<UserSearchDTO>> findUsersByFaculty(
            @Parameter(description = "Faculty name", required = true)
            @PathVariable @NotBlank String facultyName,
            @Parameter(description = "Current user ID for relationship context")
            @RequestParam(required = false) String currentUserId,
            @Parameter(description = "Page number (0-based)")
            @RequestParam(defaultValue = "0") @Min(0) int page,
            @Parameter(description = "Page size")
            @RequestParam(defaultValue = "20") @Min(1) @Max(100) int size) {

        log.info("GET /users/faculty/{} - Finding users by faculty", facultyName);
        Pageable pageable = PageRequest.of(page, size);
        Page<UserSearchDTO> users = userService.findUsersByFaculty(facultyName, currentUserId, pageable);
        return ResponseEntity.ok(users);
    }

    @Operation(summary = "Find users by major",
               description = "Find all users belonging to a specific major")
    @GetMapping("/major/{majorName}")
    public ResponseEntity<Page<UserSearchDTO>> findUsersByMajor(
            @Parameter(description = "Major name", required = true)
            @PathVariable @NotBlank String majorName,
            @Parameter(description = "Current user ID for relationship context")
            @RequestParam(required = false) String currentUserId,
            @Parameter(description = "Page number (0-based)")
            @RequestParam(defaultValue = "0") @Min(0) int page,
            @Parameter(description = "Page size")
            @RequestParam(defaultValue = "20") @Min(1) @Max(100) int size) {

        log.info("GET /users/major/{} - Finding users by major", majorName);
        Pageable pageable = PageRequest.of(page, size);
        Page<UserSearchDTO> users = userService.findUsersByMajor(majorName, currentUserId, pageable);
        return ResponseEntity.ok(users);
    }

    @Operation(summary = "Find users by batch",
               description = "Find all users belonging to a specific batch year")
    @GetMapping("/batch/{batchYear}")
    public ResponseEntity<Page<UserSearchDTO>> findUsersByBatch(
            @Parameter(description = "Batch year", required = true)
            @PathVariable @NotNull Integer batchYear,
            @Parameter(description = "Current user ID for relationship context")
            @RequestParam(required = false) String currentUserId,
            @Parameter(description = "Page number (0-based)")
            @RequestParam(defaultValue = "0") @Min(0) int page,
            @Parameter(description = "Page size")
            @RequestParam(defaultValue = "20") @Min(1) @Max(100) int size) {

        log.info("GET /users/batch/{} - Finding users by batch", batchYear);
        Pageable pageable = PageRequest.of(page, size);
        Page<UserSearchDTO> users = userService.findUsersByBatch(batchYear, currentUserId, pageable);
        return ResponseEntity.ok(users);
    }

    // Friend Management

    @Operation(summary = "Get user's friends",
               description = "Get paginated list of user's friends")
    @GetMapping("/{userId}/friends")
    public ResponseEntity<Page<UserSearchDTO>> getFriends(
            @Parameter(description = "User ID", required = true)
            @PathVariable @NotBlank String userId,
            @Parameter(description = "Page number (0-based)")
            @RequestParam(defaultValue = "0") @Min(0) int page,
            @Parameter(description = "Page size")
            @RequestParam(defaultValue = "20") @Min(1) @Max(100) int size) {

        log.info("GET /users/{}/friends - Getting friends", userId);
        Pageable pageable = PageRequest.of(page, size);
        Page<UserSearchDTO> friends = userService.getFriends(userId, pageable);
        return ResponseEntity.ok(friends);
    }

    @Operation(summary = "Get sent friend requests",
               description = "Get list of friend requests sent by user")
    @GetMapping("/{userId}/friend-requests/sent")
    public ResponseEntity<List<FriendRequestDTO>> getSentFriendRequests(
            @Parameter(description = "User ID", required = true)
            @PathVariable @NotBlank String userId) {

        log.info("GET /users/{}/friend-requests/sent - Getting sent friend requests", userId);
        List<FriendRequestDTO> sentRequests = userService.getSentFriendRequests(userId);
        return ResponseEntity.ok(sentRequests);
    }

    @Operation(summary = "Get received friend requests",
               description = "Get list of friend requests received by user")
    @GetMapping("/{userId}/friend-requests/received")
    public ResponseEntity<List<FriendRequestDTO>> getReceivedFriendRequests(
            @Parameter(description = "User ID", required = true)
            @PathVariable @NotBlank String userId) {

        log.info("GET /users/{}/friend-requests/received - Getting received friend requests", userId);
        List<FriendRequestDTO> receivedRequests = userService.getReceivedFriendRequests(userId);
        return ResponseEntity.ok(receivedRequests);
    }

    @Operation(summary = "Send friend request",
               description = "Send a friend request to another user")
    @PostMapping("/{senderId}/friend-requests/{receiverId}")
    public ResponseEntity<Map<String, String>> sendFriendRequest(
            @Parameter(description = "Sender user ID", required = true)
            @PathVariable @NotBlank String senderId,
            @Parameter(description = "Receiver user ID", required = true)
            @PathVariable @NotBlank String receiverId) {

        log.info("POST /users/{}/friend-requests/{} - Sending friend request", senderId, receiverId);
        userService.sendFriendRequest(senderId, receiverId);
        return ResponseEntity.ok(Map.of("message", "Friend request sent successfully"));
    }

    @Operation(summary = "Accept friend request",
               description = "Accept a friend request from another user")
    @PatchMapping("/{accepterId}/friend-requests/{requesterId}/accept")
    public ResponseEntity<Map<String, String>> acceptFriendRequest(
            @Parameter(description = "Accepter user ID", required = true)
            @PathVariable @NotBlank String accepterId,
            @Parameter(description = "Requester user ID", required = true)
            @PathVariable @NotBlank String requesterId) {

        log.info("PATCH /users/{}/friend-requests/{}/accept - Accepting friend request", accepterId, requesterId);
        userService.acceptFriendRequest(requesterId, accepterId);
        return ResponseEntity.ok(Map.of("message", "Friend request accepted successfully"));
    }

    @Operation(summary = "Reject friend request",
               description = "Reject a friend request from another user")
    @PatchMapping("/{rejecterId}/friend-requests/{requesterId}/reject")
    public ResponseEntity<Map<String, String>> rejectFriendRequest(
            @Parameter(description = "Rejecter user ID", required = true)
            @PathVariable @NotBlank String rejecterId,
            @Parameter(description = "Requester user ID", required = true)
            @PathVariable @NotBlank String requesterId) {

        log.info("PATCH /users/{}/friend-requests/{}/reject - Rejecting friend request", rejecterId, requesterId);
        userService.rejectFriendRequest(requesterId, rejecterId);
        return ResponseEntity.ok(Map.of("message", "Friend request rejected successfully"));
    }

    @Operation(summary = "Remove friend",
               description = "Remove a friend from user's friend list")
    @DeleteMapping("/{userId}/friends/{friendId}")
    public ResponseEntity<Map<String, String>> removeFriend(
            @Parameter(description = "User ID", required = true)
            @PathVariable @NotBlank String userId,
            @Parameter(description = "Friend ID", required = true)
            @PathVariable @NotBlank String friendId) {

        log.info("DELETE /users/{}/friends/{} - Removing friend", userId, friendId);
        userService.removeFriend(userId, friendId);
        return ResponseEntity.ok(Map.of("message", "Friend removed successfully"));
    }

    // Utility Endpoints

    @Operation(summary = "Check if user exists",
               description = "Check if a user exists by ID")
    @GetMapping("/{userId}/exists")
    public ResponseEntity<Map<String, Boolean>> userExists(
            @Parameter(description = "User ID", required = true)
            @PathVariable @NotBlank String userId) {

        boolean exists = userService.userExists(userId);
        return ResponseEntity.ok(Map.of("exists", exists));
    }

    @Operation(summary = "Check if email exists",
               description = "Check if an email is already registered")
    @GetMapping("/email/{email}/exists")
    public ResponseEntity<Map<String, Boolean>> emailExists(
            @Parameter(description = "Email address", required = true)
            @PathVariable @NotBlank String email) {

        boolean exists = userService.emailExists(email);
        return ResponseEntity.ok(Map.of("exists", exists));
    }

    @Operation(summary = "Check if student ID exists",
               description = "Check if a student ID is already registered")
    @GetMapping("/student-id/{studentId}/exists")
    public ResponseEntity<Map<String, Boolean>> studentIdExists(
            @Parameter(description = "Student ID", required = true)
            @PathVariable @NotBlank String studentId) {

        boolean exists = userService.studentIdExists(studentId);
        return ResponseEntity.ok(Map.of("exists", exists));
    }
}
