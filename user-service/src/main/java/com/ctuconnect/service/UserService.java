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
            throw new RuntimeException("No friend request found from user: " + friendId);
        }

        // Create the bidirectional relationship
        user.getFriends().add(friend);
        userRepository.save(user);
    }

    /**
     * Reject a friend request
     */
    public void rejectFriendInvite(String userId, String friendId) {
        // No action needed as we're not establishing the relationship
    }

    /**
     * Get all friends of a user
     */
    public FriendsDTO getFriends(String userId) {
        List<UserEntity> friends = userRepository.findFriends(userId);

        FriendsDTO friendsDTO = new FriendsDTO();
        friendsDTO.setUserId(userId);
        friendsDTO.setFriends(friends.stream().map(this::mapToDTO).collect(Collectors.toList()));
        friendsDTO.setFriendIds(friends.stream().map(UserEntity::getId).collect(Collectors.toList()));

        return friendsDTO;
    }

    /**
     * Get mutual friends between two users
     */
    public FriendsDTO getMutualFriends(String userId1, String userId2) {
        List<UserEntity> mutualFriends = userRepository.findMutualFriends(userId1, userId2);

        FriendsDTO friendsDTO = new FriendsDTO();
        friendsDTO.setUserId(userId1);
        friendsDTO.setMutualFriends(mutualFriends.stream().map(this::mapToDTO).collect(Collectors.toList()));
        friendsDTO.setMutualFriendsCount(mutualFriends.size());

        return friendsDTO;
    }

    /**
     * Get friend suggestions based on various criteria:
     * 1. Friends of friends
     * 2. Same college
     * 3. Same faculty
     * 4. Same major
     */
    public FriendsDTO getFriendSuggestions(String userId) {
        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        // Get all users for now - in a real implementation, you'd want to optimize this
        List<UserEntity> allUsers = userRepository.findAll();
        List<UserEntity> friends = userRepository.findFriends(userId);

        // Filter out existing friends and self
        Set<String> friendIds = friends.stream()
                .map(UserEntity::getId)
                .collect(Collectors.toSet());

        // Calculate suggestions with their scores
        List<UserDTO> suggestions = allUsers.stream()
                .filter(u -> !u.getId().equals(userId) && !friendIds.contains(u.getId()))
                .map(u -> {
                    UserDTO dto = mapToDTO(u);

                    // Calculate matching criteria
                    dto.setSameCollege(Objects.equals(u.getCollege(), user.getCollege()));
                    dto.setSameFaculty(Objects.equals(u.getFaculty(), user.getFaculty()));
                    dto.setSameMajor(Objects.equals(u.getMajor(), user.getMajor()));

                    // Calculate mutual friends
                    List<UserEntity> mutual = userRepository.findMutualFriends(userId, u.getId());
                    dto.setMutualFriendsCount(mutual.size());

                    return dto;
                })
                // Sort by "most matching" - mutual friends count first, then other criteria
                .sorted(Comparator
                        .comparingInt(UserDTO::getMutualFriendsCount).reversed()
                        .thenComparing((UserDTO u) -> u.isSameCollege() ? 1 : 0, Comparator.reverseOrder())
                        .thenComparing((UserDTO u) -> u.isSameFaculty() ? 1 : 0, Comparator.reverseOrder())
                        .thenComparing((UserDTO u) -> u.isSameMajor() ? 1 : 0, Comparator.reverseOrder())
                )
                .limit(10) // Limit to top 10 suggestions
                .collect(Collectors.toList());

        FriendsDTO friendsDTO = new FriendsDTO();
        friendsDTO.setUserId(userId);
        friendsDTO.setFriendSuggestions(suggestions);

        return friendsDTO;
    }

    /**
     * Get users based on relationship filters
     */
    public List<UserDTO> getUsersByRelationshipFilters(String userId, RelationshipFilterDTO filters) {
        UserEntity user = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        List<UserEntity> filteredUsers;

        if (filters.isFriend()) {
            // Get friends with filters
            filteredUsers = userRepository.findFriendsWithFilters(
                    userId,
                    filters.isSameCollege(),
                    filters.isSameFaculty(),
                    filters.isSameMajor(),
                    filters.isSameBatch()
            );
        } else {
            // Get all users with filters
            filteredUsers = userRepository.findUsersWithFilters(
                    userId,
                    filters.isSameCollege(),
                    filters.isSameFaculty(),
                    filters.isSameMajor(),
                    filters.isSameBatch()
            );
        }

        return filteredUsers.stream()
                .map(entity -> {
                    UserDTO dto = mapToDTO(entity);

                    // Add relationship context
                    dto.setSameCollege(Objects.equals(entity.getCollege(), user.getCollege()));
                    dto.setSameFaculty(Objects.equals(entity.getFaculty(), user.getFaculty()));
                    dto.setSameMajor(Objects.equals(entity.getMajor(), user.getMajor()));

                    // Calculate mutual friends if needed
                    List<UserEntity> mutual = userRepository.findMutualFriends(userId, entity.getId());
                    dto.setMutualFriendsCount(mutual.size());

                    return dto;
                })
                .collect(Collectors.toList());
    }

    /**
     * Map entity to DTO
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

        if (entity.getFriends() != null) {
            dto.setFriendIds(entity.getFriends().stream()
                    .map(UserEntity::getId)
                    .collect(Collectors.toSet()));
        }

        return dto;
    }

    /**
     * Map DTO to entity
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
        entity.setCreatedAt(dto.getCreatedAt());
        entity.setUpdatedAt(dto.getUpdatedAt());

        return entity;
    }
}
