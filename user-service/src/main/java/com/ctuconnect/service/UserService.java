package com.ctuconnect.service;

import com.ctuconnect.dto.FriendsDTO;
import com.ctuconnect.dto.RelationshipFilterDTO;
import com.ctuconnect.dto.UserDTO;
import com.ctuconnect.entity.UserEntity;
import com.ctuconnect.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class UserService {

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private UserEventPublisher userEventPublisher;

    /**
     * Create a new user
     */
    public UserDTO createUser(UserDTO userDTO) {
        UserEntity userEntity = mapToEntity(userDTO);
        userEntity.setCreatedAt(LocalDateTime.now());
        userEntity.setUpdatedAt(LocalDateTime.now());

        UserEntity savedUser = userRepository.save(userEntity);
        return mapToDTO(savedUser);
    }

    /**
     * Get user profile by ID or email (fallback for compatibility)
     */
    public UserDTO getUserProfile(String userIdOrEmail) {
        Optional<UserEntity> userEntity = userRepository.findById(userIdOrEmail);

        // If not found by ID, try to find by email (fallback for compatibility)
        if (userEntity.isEmpty()) {
            userEntity = userRepository.findByEmail(userIdOrEmail);
        }

        if (userEntity.isEmpty()) {
            throw new RuntimeException("User not found with id or email: " + userIdOrEmail);
        }

        return mapToDTO(userEntity.get());
    }

    /**
     * Update user profile by ID or email (fallback for compatibility)
     */
    public UserDTO updateUserProfile(String userIdOrEmail, UserDTO userDTO) {
        Optional<UserEntity> userEntityOpt = userRepository.findById(userIdOrEmail);

        // If not found by ID, try to find by email (fallback for compatibility)
        if (userEntityOpt.isEmpty()) {
            userEntityOpt = userRepository.findByEmail(userIdOrEmail);
        }

        if (userEntityOpt.isEmpty()) {
            throw new RuntimeException("User not found with id or email: " + userIdOrEmail);
        }

        UserEntity userEntity = userEntityOpt.get();

        // Update only profile fields (not system fields like id, createdAt)
        if (userDTO.getFullName() != null) userEntity.setFullName(userDTO.getFullName());
        if (userDTO.getEmail() != null) userEntity.setEmail(userDTO.getEmail());
        if (userDTO.getUsername() != null) userEntity.setUsername(userDTO.getUsername());
        if (userDTO.getStudentId() != null) userEntity.setStudentId(userDTO.getStudentId());
        if (userDTO.getBatch() != null) userEntity.setBatch(userDTO.getBatch());
        if (userDTO.getCollege() != null) userEntity.setCollege(userDTO.getCollege());
        if (userDTO.getFaculty() != null) userEntity.setFaculty(userDTO.getFaculty());
        if (userDTO.getMajor() != null) userEntity.setMajor(userDTO.getMajor());
        if (userDTO.getGender() != null) userEntity.setGender(userDTO.getGender());
        if (userDTO.getBio() != null) userEntity.setBio(userDTO.getBio());
        if (userDTO.getRole() != null) userEntity.setRole(userDTO.getRole());

        userEntity.setUpdatedAt(LocalDateTime.now());
        UserEntity updatedUser = userRepository.save(userEntity);

        // Publish user profile updated event
        userEventPublisher.publishUserProfileUpdatedEvent(
            userIdOrEmail,
            updatedUser.getEmail(),
            updatedUser.getFullName(),
            updatedUser.getFullName(), // firstName - using fullName as we don't have separate first/last names
            "", // lastName - empty as we're using fullName
            updatedUser.getBio(),
            "" // profilePicture - not implemented yet
        );

        return mapToDTO(updatedUser);
    }

    /**
     * Send a friend request - GỬI LỜI MỜI KẾT BẠN
     */
    @Transactional
    public void addFriend(String userId, String friendId) {
        if (userId.equals(friendId)) {
            throw new IllegalArgumentException("Cannot add yourself as a friend");
        }

        // Kiểm tra user tồn tại
        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        UserEntity friend = userRepository.findById(friendId)
                .orElseThrow(() -> new RuntimeException("Friend not found with id: " + friendId));

        // Kiểm tra đã là bạn bè chưa
        if (userRepository.areFriends(userId, friendId)) {
            throw new IllegalStateException("Users are already friends");
        }

        // Kiểm tra đã có friend request chưa
        if (userRepository.hasPendingFriendRequest(userId, friendId)) {
            throw new IllegalStateException("Friend request already sent");
        }

        // Kiểm tra xem có friend request ngược lại không (để auto-accept)
        if (userRepository.hasPendingFriendRequest(friendId, userId)) {
            // Tự động chấp nhận nếu đã có request ngược lại
            userRepository.acceptFriendRequest(friendId, userId);

            // Publish friend accepted event
            userEventPublisher.publishUserRelationshipChangedEvent(
                userId,
                friendId,
                "FRIEND_ACCEPTED",
                "UPDATED"
            );
        } else {
            // Gửi friend request mới
            userRepository.sendFriendRequest(userId, friendId);

            // Publish friend request event
            userEventPublisher.publishUserRelationshipChangedEvent(
                userId,
                friendId,
                "FRIEND_REQUEST",
                "CREATED"
            );
        }
    }

    /**
     * Get friend requests received by this user - LỜI MỜI KẾT BẠN NHẬN ĐƯỢC
     */
    public List<UserDTO> getFriendRequests(String userId) {
        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        List<UserEntity> incomingRequests = userRepository.findIncomingFriendRequests(userId);
        return incomingRequests.stream()
                .map(this::mapToDTO)
                .collect(Collectors.toList());
    }

    /**
     * Get friend requests sent by this user - LỜI MỜI KẾT BẠN ĐÃ GỬI
     */
    public List<UserDTO> getFriendRequested(String userId) {
        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        List<UserEntity> outgoingRequests = userRepository.findOutgoingFriendRequests(userId);
        return outgoingRequests.stream()
                .map(this::mapToDTO)
                .collect(Collectors.toList());
    }

    /**
     * Accept a friend request - CHẤP NHẬN LỜI MỜI KẾT BẠN
     */
    @Transactional
    public void acceptFriendInvite(String userId, String friendId) {
        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        UserEntity friend = userRepository.findById(friendId)
                .orElseThrow(() -> new RuntimeException("Friend not found with id: " + friendId));

        // Kiểm tra có friend request từ friendId đến userId không
        if (!userRepository.hasPendingFriendRequest(friendId, userId)) {
            throw new IllegalStateException("No pending friend request from " + friendId);
        }

        // Chấp nhận friend request
        userRepository.acceptFriendRequest(friendId, userId);

        // Publish friend accepted event
        userEventPublisher.publishUserRelationshipChangedEvent(
            userId,
            friendId,
            "FRIEND_ACCEPTED",
            "UPDATED"
        );
    }

    /**
     * Reject a friend request - TỪ CHỐI LỜI MỜI KẾT BẠN
     */
    @Transactional
    public void rejectFriendInvite(String userId, String friendId) {
        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        UserEntity friend = userRepository.findById(friendId)
                .orElseThrow(() -> new RuntimeException("Friend not found with id: " + friendId));

        // Kiểm tra có friend request từ friendId đến userId không
        if (!userRepository.hasPendingFriendRequest(friendId, userId)) {
            throw new IllegalStateException("No pending friend request from " + friendId);
        }

        // Từ chối friend request
        userRepository.rejectFriendRequest(friendId, userId);

        // Publish friend rejected event
        userEventPublisher.publishUserRelationshipChangedEvent(
            userId,
            friendId,
            "FRIEND_REQUEST",
            "REJECTED"
        );
    }

    /**
     * Remove a friend (unfriend) - HỦY KẾT BẠN
     */
    @Transactional
    public void removeFriend(String userId, String friendId) {
        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        UserEntity friend = userRepository.findById(friendId)
                .orElseThrow(() -> new RuntimeException("Friend not found with id: " + friendId));

        // Kiểm tra có phải bạn bè không
        if (!userRepository.areFriends(userId, friendId)) {
            throw new IllegalStateException("Users are not friends");
        }

        // Xóa friendship
        userRepository.deleteFriendship(userId, friendId);

        // Publish friend removed event
        userEventPublisher.publishUserRelationshipChangedEvent(
            userId,
            friendId,
            "FRIEND_REMOVED",
            "DELETED"
        );
    }

    /**
     * Get all friends of a user - LẤY DANH SÁCH BẠN BÈ
     */
    public FriendsDTO getFriends(String userId) {
        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        List<UserEntity> friends = userRepository.findFriends(userId);
        List<UserDTO> friendDTOs = friends.stream()
                .map(this::mapToDTO)
                .collect(Collectors.toList());

        return new FriendsDTO(friendDTOs);
    }

    /**
     * Get mutual friends between two users - LẤY BẠN CHUNG
     */
    public FriendsDTO getMutualFriends(String userId, String otherUserId) {
        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        UserEntity otherUser = userRepository.findById(otherUserId)
                .orElseThrow(() -> new RuntimeException("Other user not found with id: " + otherUserId));

        List<UserEntity> mutualFriends = userRepository.findMutualFriends(userId, otherUserId);
        List<UserDTO> mutualFriendDTOs = mutualFriends.stream()
                .map(this::mapToDTO)
                .collect(Collectors.toList());

        return FriendsDTO.ofMutualFriends(mutualFriendDTOs);
    }

    /**
     * Get friend suggestions - GỢI Ý KẾT BẠN
     */
    public FriendsDTO getFriendSuggestions(String userId) {
        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        List<UserEntity> suggestions = userRepository.findFriendSuggestions(userId);
        List<UserDTO> suggestionDTOs = suggestions.stream()
                .map(u -> {
                    UserDTO dto = mapToDTO(u);
                    // Calculate mutual friends count
                    List<UserEntity> mutualFriends = userRepository.findMutualFriends(userId, u.getId());
                    dto.setMutualFriendsCount(mutualFriends.size());

                    // Calculate similarity
                    calculateSimilarityScore(user, u, dto);
                    return dto;
                })
                .collect(Collectors.toList());

        return FriendsDTO.ofSuggestions(suggestionDTOs);
    }

    /**
     * Filter users by relationship criteria - LỌC NGƯỜI DÙNG THEO TIÊU CHÍ
     */
    public List<UserDTO> getUsersByRelationshipFilters(String userId, RelationshipFilterDTO filters) {
        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        List<UserEntity> filteredUsers;

        // Sử dụng Neo4j query để filter hiệu quả
        if (filters.getCollege() != null || filters.getFaculty() != null ||
            filters.getMajor() != null || filters.getBatch() != null) {

            // Convert filters to boolean flags for Neo4j query
            boolean isSameCollege = filters.getCollege() != null;
            boolean isSameFaculty = filters.getFaculty() != null;
            boolean isSameMajor = filters.getMajor() != null;
            boolean isSameBatch = filters.getBatch() != null;

            filteredUsers = userRepository.findUsersWithFilters(userId, isSameCollege, isSameFaculty, isSameMajor, isSameBatch);
        } else {
            // If no specific filters, get all users except self
            filteredUsers = userRepository.findAll().stream()
                    .filter(u -> !u.getId().equals(userId))
                    .collect(Collectors.toList());
        }

        return filteredUsers.stream()
                .filter(u -> matchesFilters(user, u, filters))
                .map(this::mapToDTO)
                .collect(Collectors.toList());
    }

    /**
     * Get all users (Admin only)
     */
    public List<UserDTO> getAllUsers() {
        return userRepository.findAll().stream()
                .map(this::mapToDTO)
                .collect(Collectors.toList());
    }

    /**
     * Delete user (Admin only)
     */
    @Transactional
    public void deleteUser(String userId) {
        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        // Xóa tất cả friend relationships và friend requests
        List<UserEntity> friends = userRepository.findFriends(userId);
        for (UserEntity friend : friends) {
            userRepository.deleteFriendship(userId, friend.getId());
        }

        // Xóa tất cả friend requests (incoming và outgoing)
        List<UserEntity> incomingRequests = userRepository.findIncomingFriendRequests(userId);
        for (UserEntity requester : incomingRequests) {
            userRepository.rejectFriendRequest(requester.getId(), userId);
        }

        List<UserEntity> outgoingRequests = userRepository.findOutgoingFriendRequests(userId);
        for (UserEntity receiver : outgoingRequests) {
            userRepository.rejectFriendRequest(userId, receiver.getId());
        }

        // Delete the user
        userRepository.deleteById(userId);
    }

    /**
     * Calculate similarity score for friend suggestions
     */
    private void calculateSimilarityScore(UserEntity user, UserEntity candidate, UserDTO candidateDTO) {
        // Check similarity attributes
        candidateDTO.setSameCollege(Objects.equals(user.getCollege(), candidate.getCollege()));
        candidateDTO.setSameFaculty(Objects.equals(user.getFaculty(), candidate.getFaculty()));
        candidateDTO.setSameMajor(Objects.equals(user.getMajor(), candidate.getMajor()));
        candidateDTO.setSameBatch(Objects.equals(user.getBatch(), candidate.getBatch()));
    }

    /**
     * Check if user matches relationship filters
     */
    private boolean matchesFilters(UserEntity user, UserEntity candidate, RelationshipFilterDTO filters) {
        if (filters.getCollege() != null && !filters.getCollege().equals(candidate.getCollege())) {
            return false;
        }
        if (filters.getFaculty() != null && !filters.getFaculty().equals(candidate.getFaculty())) {
            return false;
        }
        if (filters.getMajor() != null && !filters.getMajor().equals(candidate.getMajor())) {
            return false;
        }
        if (filters.getBatch() != null && !filters.getBatch().equals(candidate.getBatch())) {
            return false;
        }
        if (filters.getGender() != null && !filters.getGender().equals(candidate.getGender())) {
            return false;
        }
        return true;
    }

    /**
     * Map UserEntity to UserDTO
     */
    private UserDTO mapToDTO(UserEntity entity) {
        UserDTO dto = new UserDTO();
        dto.setId(entity.getId());
        dto.setEmail(entity.getEmail());
        dto.setUsername(entity.getUsername());
        dto.setStudentId(entity.getStudentId());
        dto.setBatch(entity.getBatch());
        dto.setFullName(entity.getFullName());
        dto.setRole(entity.getRole());
        dto.setCollege(entity.getCollege());
        dto.setFaculty(entity.getFaculty());
        dto.setMajor(entity.getMajor());
        dto.setGender(entity.getGender());
        dto.setBio(entity.getBio());
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setUpdatedAt(entity.getUpdatedAt());
        dto.setIsActive(entity.getIsActive());

        return dto;
    }

    /**
     * Map UserDTO to UserEntity
     */
    private UserEntity mapToEntity(UserDTO dto) {
        UserEntity entity = new UserEntity();
        entity.setId(dto.getId());
        entity.setEmail(dto.getEmail());
        entity.setUsername(dto.getUsername());
        entity.setStudentId(dto.getStudentId());
        entity.setBatch(dto.getBatch());
        entity.setFullName(dto.getFullName());
        entity.setRole(dto.getRole());
        entity.setCollege(dto.getCollege());
        entity.setFaculty(dto.getFaculty());
        entity.setMajor(dto.getMajor());
        entity.setGender(dto.getGender());
        entity.setBio(dto.getBio());
        entity.setIsActive(dto.getIsActive());
        return entity;
    }
}
