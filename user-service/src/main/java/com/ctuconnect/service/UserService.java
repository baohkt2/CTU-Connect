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
     * Get user profile by ID
     */
    public UserDTO getUserProfile(String userId) {
        UserEntity userEntity = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));
        return mapToDTO(userEntity);
    }

    /**
     * Update user profile
     */
    public UserDTO updateUserProfile(String userId, UserDTO userDTO) {
        UserEntity userEntity = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        // Update only profile fields (not system fields like id, createdAt)
        if (userDTO.getFullName() != null) userEntity.setFullName(userDTO.getFullName());
        if (userDTO.getEmail() != null) userEntity.setEmail(userDTO.getEmail());
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
            userId,
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
     * Send a friend request (unidirectional relationship)
     */
    @Transactional
    public void addFriend(String userId, String friendId) {
        if (userId.equals(friendId)) {
            throw new IllegalArgumentException("Cannot add yourself as a friend");
        }

        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        UserEntity friend = userRepository.findById(friendId)
                .orElseThrow(() -> new RuntimeException("Friend not found with id: " + friendId));

        // Neo4j will automatically create the FRIEND relationship
        user.getFriends().add(friend);
        userRepository.save(user);

        // Publish friend request event
        userEventPublisher.publishUserRelationshipChangedEvent(
            userId,
            friendId,
            "FRIEND_REQUEST",
            "CREATED"
        );
    }

    /**
     * Accept a friend request (make relationship bidirectional)
     */
    @Transactional
    public void acceptFriendInvite(String userId, String friendId) {
        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        UserEntity friend = userRepository.findById(friendId)
                .orElseThrow(() -> new RuntimeException("Friend not found with id: " + friendId));

        // Check if friend request exists
        boolean requestExists = friend.getFriends().stream()
                .anyMatch(f -> f.getId().equals(userId));

        if (!requestExists) {
            throw new IllegalStateException("No friend request found from " + friendId);
        }

        // Make relationship bidirectional
        user.getFriends().add(friend);
        userRepository.save(user);

        // Publish friend accepted event
        userEventPublisher.publishUserRelationshipChangedEvent(
            userId,
            friendId,
            "FRIEND_ACCEPTED",
            "UPDATED"
        );
    }

    /**
     * Reject a friend request
     */
    @Transactional
    public void rejectFriendInvite(String userId, String friendId) {
        UserEntity friend = userRepository.findById(friendId)
                .orElseThrow(() -> new RuntimeException("Friend not found with id: " + friendId));

        // Remove the friend request (unidirectional relationship)
        friend.getFriends().removeIf(f -> f.getId().equals(userId));
        userRepository.save(friend);

        // Publish friend rejected event
        userEventPublisher.publishUserRelationshipChangedEvent(
            userId,
            friendId,
            "FRIEND_REQUEST",
            "DELETED"
        );
    }

    /**
     * Remove a friend (unfriend)
     */
    @Transactional
    public void removeFriend(String userId, String friendId) {
        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        UserEntity friend = userRepository.findById(friendId)
                .orElseThrow(() -> new RuntimeException("Friend not found with id: " + friendId));

        // Remove bidirectional friendship
        user.getFriends().removeIf(f -> f.getId().equals(friendId));
        friend.getFriends().removeIf(f -> f.getId().equals(userId));

        userRepository.save(user);
        userRepository.save(friend);

        // Publish friend removed event
        userEventPublisher.publishUserRelationshipChangedEvent(
            userId,
            friendId,
            "FRIEND_REMOVED",
            "DELETED"
        );
    }

    /**
     * Get all friends of a user
     */
    public FriendsDTO getFriends(String userId) {
        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        List<UserDTO> friends = user.getFriends().stream()
                .map(this::mapToDTO)
                .collect(Collectors.toList());

        return new FriendsDTO(friends);
    }

    /**
     * Get mutual friends between two users
     */
    public FriendsDTO getMutualFriends(String userId, String otherUserId) {
        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        UserEntity otherUser = userRepository.findById(otherUserId)
                .orElseThrow(() -> new RuntimeException("Other user not found with id: " + otherUserId));

        Set<String> userFriendIds = user.getFriends().stream()
                .map(UserEntity::getId)
                .collect(Collectors.toSet());

        List<UserDTO> mutualFriends = otherUser.getFriends().stream()
                .filter(friend -> userFriendIds.contains(friend.getId()))
                .map(this::mapToDTO)
                .collect(Collectors.toList());

        return FriendsDTO.ofMutualFriends(mutualFriends);
    }

    /**
     * Get friend suggestions based on mutual connections and similar attributes
     */
    public FriendsDTO getFriendSuggestions(String userId) {
        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        // Get existing friend IDs to exclude them from suggestions
        Set<String> existingFriendIds = user.getFriends().stream()
                .map(UserEntity::getId)
                .collect(Collectors.toSet());
        existingFriendIds.add(userId); // Exclude self

        // Find users with similar attributes or mutual friends
        List<UserEntity> allUsers = userRepository.findAll();

        List<UserDTO> suggestions = allUsers.stream()
                .filter(u -> !existingFriendIds.contains(u.getId()))
                .map(u -> {
                    UserDTO dto = mapToDTO(u);
                    // Calculate similarity score
                    calculateSimilarityScore(user, u, dto);
                    return dto;
                })
                .sorted((a, b) -> Integer.compare(b.getMutualFriendsCount(), a.getMutualFriendsCount()))
                .limit(10)
                .collect(Collectors.toList());

        return FriendsDTO.ofSuggestions(suggestions);
    }

    /**
     * Filter users by relationship criteria
     */
    public List<UserDTO> getUsersByRelationshipFilters(String userId, RelationshipFilterDTO filters) {
        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        List<UserEntity> allUsers = userRepository.findAll();

        return allUsers.stream()
                .filter(u -> true) // Exclude self
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

        // Remove all friend relationships
        user.getFriends().clear();
        userRepository.save(user);

        // Remove this user from other users' friend lists
        List<UserEntity> allUsers = userRepository.findAll();
        for (UserEntity otherUser : allUsers) {
            otherUser.getFriends().removeIf(friend -> friend.getId().equals(userId));
            userRepository.save(otherUser);
        }

        // Delete the user
        userRepository.deleteById(userId);
    }

    /**
     * Calculate similarity score for friend suggestions
     */
    private void calculateSimilarityScore(UserEntity user, UserEntity candidate, UserDTO candidateDTO) {
        // Count mutual friends - Updated to use String UUID
        Set<String> userFriendIds = user.getFriends().stream()
                .map(UserEntity::getId)
                .collect(Collectors.toSet());

        int mutualFriendsCount = (int) candidate.getFriends().stream()
                .mapToLong(friend -> userFriendIds.contains(friend.getId()) ? 1 : 0)
                .sum();

        candidateDTO.setMutualFriendsCount(mutualFriendsCount);

        // Check similarity attributes
        candidateDTO.setSameCollege(Objects.equals(user.getCollege(), candidate.getCollege()));
        candidateDTO.setSameFaculty(Objects.equals(user.getFaculty(), candidate.getFaculty()));
        candidateDTO.setSameMajor(Objects.equals(user.getMajor(), candidate.getMajor()));
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

        // Set friend IDs - Updated to use String UUID
        Set<String> friendIds = entity.getFriends().stream()
                .map(UserEntity::getId)
                .collect(Collectors.toSet());
        dto.setFriendIds(friendIds);

        return dto;
    }

    /**
     * Map UserDTO to UserEntity
     */
    private UserEntity mapToEntity(UserDTO dto) {
        UserEntity entity = new UserEntity();
        entity.setId(dto.getId());
        entity.setEmail(dto.getEmail());
        entity.setStudentId(dto.getStudentId());
        entity.setBatch(dto.getBatch());
        entity.setFullName(dto.getFullName());
        entity.setRole(dto.getRole());
        entity.setCollege(dto.getCollege());
        entity.setFaculty(dto.getFaculty());
        entity.setMajor(dto.getMajor());
        entity.setGender(dto.getGender());
        entity.setBio(dto.getBio());
        return entity;
    }
}
