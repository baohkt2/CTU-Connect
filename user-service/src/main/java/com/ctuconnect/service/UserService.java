package com.ctuconnect.service;

import com.ctuconnect.dto.UserProfileDTO;
import com.ctuconnect.dto.UserUpdateDTO;
import com.ctuconnect.dto.UserSearchDTO;
import com.ctuconnect.dto.FriendRequestDTO;
import com.ctuconnect.entity.UserEntity;
import com.ctuconnect.exception.UserNotFoundException;
import com.ctuconnect.exception.InvalidOperationException;
import com.ctuconnect.exception.DuplicateResourceException;
import com.ctuconnect.mapper.UserMapper;
import com.ctuconnect.repository.UserRepository;
import com.ctuconnect.repository.MajorRepository;
import com.ctuconnect.repository.BatchRepository;
import com.ctuconnect.repository.GenderRepository;
import com.ctuconnect.event.UserCreatedEvent;
import com.ctuconnect.event.UserUpdatedEvent;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.validation.annotation.Validated;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
@Validated
@Transactional
public class UserService {

    private final UserRepository userRepository;
    private final MajorRepository majorRepository;
    private final BatchRepository batchRepository;
    private final GenderRepository genderRepository;
    private final UserMapper userMapper;
    private final KafkaTemplate<String, Object> kafkaTemplate;

    private static final String USER_CREATED_TOPIC = "user-created";
    private static final String USER_UPDATED_TOPIC = "user-updated";

    // User Profile Management

    @Transactional(readOnly = true)
    public UserProfileDTO getUserProfile(@NotBlank String userId) {
        log.info("Fetching user profile for userId: {}", userId);

        var profileProjection = userRepository.findUserProfileById(userId)
            .orElseThrow(() -> new UserNotFoundException("User not found with ID: " + userId));

        return userMapper.toUserProfileDTO(profileProjection);
    }

    @Transactional(readOnly = true)
    public UserProfileDTO getUserProfileByEmail(@NotBlank String email) {
        log.info("Fetching user profile for email: {}", email);

        var user = userRepository.findByEmail(email)
            .orElseThrow(() -> new UserNotFoundException("User not found with email: " + email));

        return getUserProfile(user.getId());
    }

    public UserEntity createUser(@NotBlank String authUserId,
                                @NotBlank String email,
                                String username,
                                @NotBlank String role) {
        log.info("Creating user with authUserId: {}, email: {}", authUserId, email);

        // Check if user already exists
        if (userRepository.existsById(authUserId)) {
            throw new DuplicateResourceException("User already exists with ID: " + authUserId);
        }

        if (userRepository.existsByEmail(email)) {
            throw new DuplicateResourceException("User already exists with email: " + email);
        }

        // Create user entity
        var user = UserEntity.fromAuthService(authUserId, email, username, role);
        var savedUser = userRepository.save(user);

        // Publish user created event
        publishUserCreatedEvent(savedUser);

        log.info("User created successfully with ID: {}", savedUser.getId());
        return savedUser;
    }

    public UserProfileDTO updateUserProfile(@NotBlank String userId,
                                          @Valid UserUpdateDTO updateDTO) {
        log.info("Updating user profile for userId: {}", userId);

        var user = userRepository.findById(userId)
            .orElseThrow(() -> new UserNotFoundException("User not found with ID: " + userId));

        // Update basic profile information
        if (updateDTO.getFullName() != null) {
            user.setFullName(updateDTO.getFullName());
        }

        if (updateDTO.getBio() != null) {
            user.setBio(updateDTO.getBio());
        }

        if (updateDTO.getStudentId() != null) {
            // Check if student ID is already taken
            if (!updateDTO.getStudentId().equals(user.getStudentId()) &&
                userRepository.existsByStudentId(updateDTO.getStudentId())) {
                throw new DuplicateResourceException("Student ID already exists: " + updateDTO.getStudentId());
            }
            user.setStudentId(updateDTO.getStudentId());
        }

        // Update academic information
        if (updateDTO.getMajorName() != null) {
            var major = majorRepository.findByName(updateDTO.getMajorName())
                .orElseThrow(() -> new UserNotFoundException("Major not found: " + updateDTO.getMajorName()));
            user.setMajor(major);
        }

        if (updateDTO.getBatchYear() != null) {
            var batch = batchRepository.findByYear(updateDTO.getBatchYear())
                .orElseThrow(() -> new UserNotFoundException("Batch not found: " + updateDTO.getBatchYear()));
            user.setBatch(batch);
        }

        if (updateDTO.getGenderName() != null) {
            var gender = genderRepository.findByName(updateDTO.getGenderName())
                .orElseThrow(() -> new UserNotFoundException("Gender not found: " + updateDTO.getGenderName()));
            user.setGender(gender);
        }

        user.setUpdatedAt(LocalDateTime.now());
        var savedUser = userRepository.save(user);

        // Publish user updated event
        publishUserUpdatedEvent(savedUser);

        log.info("User profile updated successfully for userId: {}", userId);
        return getUserProfile(userId);
    }

    public void deactivateUser(@NotBlank String userId) {
        log.info("Deactivating user with userId: {}", userId);

        var user = userRepository.findById(userId)
            .orElseThrow(() -> new UserNotFoundException("User not found with ID: " + userId));

        user.deactivate();
        userRepository.save(user);

        publishUserUpdatedEvent(user);

        log.info("User deactivated successfully for userId: {}", userId);
    }

    public void activateUser(@NotBlank String userId) {
        log.info("Activating user with userId: {}", userId);

        var user = userRepository.findById(userId)
            .orElseThrow(() -> new UserNotFoundException("User not found with ID: " + userId));

        user.activate();
        userRepository.save(user);

        publishUserUpdatedEvent(user);

        log.info("User activated successfully for userId: {}", userId);
    }

    // User Search and Discovery

    @Transactional(readOnly = true)
    public Page<UserSearchDTO> searchUsers(@NotBlank String searchTerm,
                                         String currentUserId,
                                         @NotNull Pageable pageable) {
        log.info("Searching users with term: {}, currentUserId: {}", searchTerm, currentUserId);

        var searchResults = userRepository.searchUsers(searchTerm, currentUserId, pageable);
        return searchResults.map(userMapper::toUserSearchDTO);
    }

    @Transactional(readOnly = true)
    public Page<UserSearchDTO> findUsersByCollege(@NotBlank String collegeName,
                                                String currentUserId,
                                                @NotNull Pageable pageable) {
        log.info("Finding users by college: {}, currentUserId: {}", collegeName, currentUserId);

        var searchResults = userRepository.findUsersByCollege(collegeName, currentUserId, pageable);
        return searchResults.map(userMapper::toUserSearchDTO);
    }

    @Transactional(readOnly = true)
    public Page<UserSearchDTO> findUsersByFaculty(@NotBlank String facultyName,
                                                String currentUserId,
                                                @NotNull Pageable pageable) {
        log.info("Finding users by faculty: {}, currentUserId: {}", facultyName, currentUserId);

        var searchResults = userRepository.findUsersByFaculty(facultyName, currentUserId, pageable);
        return searchResults.map(userMapper::toUserSearchDTO);
    }

    @Transactional(readOnly = true)
    public Page<UserSearchDTO> findUsersByMajor(@NotBlank String majorName,
                                              String currentUserId,
                                              @NotNull Pageable pageable) {
        log.info("Finding users by major: {}, currentUserId: {}", majorName, currentUserId);

        var searchResults = userRepository.findUsersByMajor(majorName, currentUserId, pageable);
        return searchResults.map(userMapper::toUserSearchDTO);
    }

    @Transactional(readOnly = true)
    public Page<UserSearchDTO> findUsersByBatch(@NotNull Integer batchYear,
                                              String currentUserId,
                                              @NotNull Pageable pageable) {
        log.info("Finding users by batch: {}, currentUserId: {}", batchYear, currentUserId);

        var searchResults = userRepository.findUsersByBatch(batchYear, currentUserId, pageable);
        return searchResults.map(userMapper::toUserSearchDTO);
    }

    // Friend Management

    @Transactional(readOnly = true)
    public Page<UserSearchDTO> getFriends(@NotBlank String userId, @NotNull Pageable pageable) {
        log.info("Getting friends for userId: {}", userId);

        var friends = userRepository.findFriends(userId, pageable);
        return friends.map(userMapper::toUserSearchDTO);
    }

    @Transactional(readOnly = true)
    public List<FriendRequestDTO> getSentFriendRequests(@NotBlank String userId) {
        log.info("Getting sent friend requests for userId: {}", userId);

        var sentRequests = userRepository.findSentFriendRequests(userId);
        return sentRequests.stream()
            .map(userMapper::toFriendRequestDTO)
            .collect(Collectors.toList());
    }

    @Transactional(readOnly = true)
    public List<FriendRequestDTO> getReceivedFriendRequests(@NotBlank String userId) {
        log.info("Getting received friend requests for userId: {}", userId);

        var receivedRequests = userRepository.findReceivedFriendRequests(userId);
        return receivedRequests.stream()
            .map(userMapper::toFriendRequestDTO)
            .collect(Collectors.toList());
    }

    public void sendFriendRequest(@NotBlank String senderId, @NotBlank String receiverId) {
        log.info("Sending friend request from userId: {} to userId: {}", senderId, receiverId);

        if (senderId.equals(receiverId)) {
            throw new InvalidOperationException("Cannot send friend request to yourself");
        }

        // Verify both users exist and are active
        var sender = userRepository.findById(senderId)
            .orElseThrow(() -> new UserNotFoundException("Sender not found with ID: " + senderId));

        var receiver = userRepository.findById(receiverId)
            .orElseThrow(() -> new UserNotFoundException("Receiver not found with ID: " + receiverId));

        if (!sender.isActive() || !receiver.isActive()) {
            throw new InvalidOperationException("Both users must be active to send friend request");
        }

        boolean success = userRepository.sendFriendRequest(senderId, receiverId);

        if (!success) {
            throw new InvalidOperationException("Unable to send friend request. Users may already be friends or request already exists");
        }

        log.info("Friend request sent successfully from userId: {} to userId: {}", senderId, receiverId);
    }

    public void acceptFriendRequest(@NotBlank String requesterId, @NotBlank String accepterId) {
        log.info("Accepting friend request from userId: {} by userId: {}", requesterId, accepterId);

        boolean success = userRepository.acceptFriendRequest(requesterId, accepterId);

        if (!success) {
            throw new InvalidOperationException("Unable to accept friend request. Request may not exist or users may be inactive");
        }

        log.info("Friend request accepted successfully from userId: {} by userId: {}", requesterId, accepterId);
    }

    public void rejectFriendRequest(@NotBlank String requesterId, @NotBlank String rejecterId) {
        log.info("Rejecting friend request from userId: {} by userId: {}", requesterId, rejecterId);

        boolean success = userRepository.rejectFriendRequest(requesterId, rejecterId);

        if (!success) {
            throw new InvalidOperationException("Unable to reject friend request. Request may not exist");
        }

        log.info("Friend request rejected successfully from userId: {} by userId: {}", requesterId, rejecterId);
    }

    public void removeFriend(@NotBlank String userId1, @NotBlank String userId2) {
        log.info("Removing friendship between userId: {} and userId: {}", userId1, userId2);

        boolean success = userRepository.removeFriend(userId1, userId2);

        if (!success) {
            throw new InvalidOperationException("Unable to remove friendship. Users may not be friends");
        }

        log.info("Friendship removed successfully between userId: {} and userId: {}", userId1, userId2);
    }

    // Utility Methods

    @Transactional(readOnly = true)
    public boolean userExists(@NotBlank String userId) {
        return userRepository.existsById(userId);
    }

    @Transactional(readOnly = true)
    public boolean emailExists(@NotBlank String email) {
        return userRepository.existsByEmail(email);
    }

    @Transactional(readOnly = true)
    public boolean studentIdExists(@NotBlank String studentId) {
        return userRepository.existsByStudentId(studentId);
    }

    @Transactional(readOnly = true)
    public List<UserEntity> getAllActiveUsers() {
        return userRepository.findByIsActiveTrue();
    }

    // Event Publishing

    private void publishUserCreatedEvent(UserEntity user) {
        try {
            var event = UserCreatedEvent.builder()
                .userId(user.getId())
                .email(user.getEmail())
                .username(user.getUsername())
                .fullName(user.getFullName())
                .role(user.getRole())
                .createdAt(user.getCreatedAt())
                .build();

            kafkaTemplate.send(USER_CREATED_TOPIC, user.getId(), event);
            log.info("Published user created event for userId: {}", user.getId());
        } catch (Exception e) {
            log.error("Failed to publish user created event for userId: {}", user.getId(), e);
        }
    }

    private void publishUserUpdatedEvent(UserEntity user) {
        try {
            var event = UserUpdatedEvent.builder()
                .userId(user.getId())
                .email(user.getEmail())
                .username(user.getUsername())
                .fullName(user.getFullName())
                .bio(user.getBio())
                .studentId(user.getStudentId())
                .role(user.getRole())
                .isActive(user.getIsActive())
                .updatedAt(user.getUpdatedAt())
                .build();

            kafkaTemplate.send(USER_UPDATED_TOPIC, user.getId(), event);
            log.info("Published user updated event for userId: {}", user.getId());
        } catch (Exception e) {
            log.error("Failed to publish user updated event for userId: {}", user.getId(), e);
        }
    }
}
