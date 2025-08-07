package com.ctuconnect.service;

import com.ctuconnect.dto.*;
import com.ctuconnect.entity.*;
import com.ctuconnect.enums.Role;
import com.ctuconnect.repository.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.validation.constraints.NotBlank;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Map;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

@Slf4j
@Service
public class UserService {

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private UserEventPublisher userEventPublisher;

    @Autowired
    private FacultyRepository facultyRepository;

    @Autowired
    private BatchRepository batchRepository;

    @Autowired
    private GenderRepository genderRepository;

    @Autowired
    private DegreeRepository degreeRepository;

    @Autowired
    private AcademicRepository academicRepository;

    @Autowired
    private PositionRepository positionRepository;

    @Autowired
    private CollegeRepository collegeRepository;

    @Autowired
    private MajorRepository majorRepository;
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
        Optional<UserEntity> userEntity;

        userEntity = userRepository.findById(userIdOrEmail);

        if (userEntity.isEmpty()) {
            throw new RuntimeException("User not found with id or email: " + userIdOrEmail);
        }

        return mapToDTO(userEntity.get());
    }


    /**
     * Update user profile by ID or email (fallback for compatibility)
     */
    /*@Transactional
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
        String userId = userEntity.getId();

        // Update basic profile fields
        if (userDTO.getFullName() != null) userEntity.setFullName(userDTO.getFullName());
        if (userDTO.getEmail() != null) userEntity.setEmail(userDTO.getEmail());
        if (userDTO.getUsername() != null) userEntity.setUsername(userDTO.getUsername());
        if (userDTO.getBio() != null) userEntity.setBio(userDTO.getBio());

        // Update role safely
        if (userDTO.getRole() != null) {
            try {
                userEntity.setRole(Role.valueOf(userDTO.getRole()));
            } catch (IllegalArgumentException e) {
                // Keep existing role if invalid role provided
            }
        }

        // Update student-specific fields
        if (userDTO.getStudentId() != null) userEntity.setStudentId(userDTO.getStudentId());

        // Update faculty-specific fields
        if (userDTO.getStaffCode() != null) userEntity.setStaffCode(userDTO.getStaffCode());

        // Update media fields
        if (userDTO.getAvatarUrl() != null) userEntity.setAvatarUrl(userDTO.getAvatarUrl());
        if (userDTO.getBackgroundUrl() != null) userEntity.setBackgroundUrl(userDTO.getBackgroundUrl());

        // Update relationships - this fixes the duplicate relationship issue
        if (userDTO.getMajorId() != null) {
            updateUserMajor(userId, userDTO.getMajorId());
        }

        if (userDTO.getBatchId() != null) {
            updateUserBatch(userId, userDTO.getBatchId());
        }

        if (userDTO.getGenderId() != null) {
            updateUserGender(userId, userDTO.getGenderId());
        }

        if (userDTO.getFacultyId() != null) {
            if (userEntity.isStudent()) {
                updateUserFaculty(userId, userDTO.getFacultyId());
            } else {
                updateUserWorkingFaculty(userId, userDTO.getFacultyId());
            }
        }

        if (userDTO.getCollegeId() != null) {
            if (userEntity.isStudent()) {
                updateUserCollege(userId, userDTO.getCollegeId());
            } else {
                updateUserWorkingCollege(userId, userDTO.getCollegeId());
            }
        }

        if (userDTO.getDegreeId() != null) {
            updateUserDegree(userId, userDTO.getDegreeId());
        }

        if (userDTO.getPositionId() != null) {
            updateUserPosition(userId, userDTO.getPositionId());
        }

        if (userDTO.getAcademicId() != null) {
            updateUserAcademic(userId, userDTO.getAcademicId());
        }

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
                updatedUser.getAvatarUrl() != null ? updatedUser.getAvatarUrl() : ""
        );

        return mapToDTO(updatedUser);
    }
*/
    // ========================= RELATIONSHIP UPDATE METHODS =========================

    /**
     * Update user's major relationship (for students)
     */
    @Transactional
    public void updateUserMajor(String userId, String majorId) {
        // Verify user exists
        userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        // Verify major exists
        majorRepository.findById(majorId)
                .orElseThrow(() -> new RuntimeException("Major not found with id: " + majorId));

        // Update relationship - this will delete old relationship and create new one
        userRepository.updateUserMajor(userId, majorId);
    }

    /**
     * Update user's batch relationship
     */
    @Transactional
    public void updateUserBatch(String userId, String batchId) {
        userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        batchRepository.findById(batchId)
                .orElseThrow(() -> new RuntimeException("Batch not found with id: " + batchId));

        userRepository.updateUserBatch(userId, batchId);
    }

    /**
     * Update user's gender relationship
     */
    @Transactional
    public void updateUserGender(String userId, String genderId) {
        userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        genderRepository.findById(genderId)
                .orElseThrow(() -> new RuntimeException("Gender not found with id: " + genderId));

        userRepository.updateUserGender(userId, genderId);
    }

    /**
     * Update user's faculty relationship (for students)
     */
    @Transactional
    public void updateUserFaculty(String userId, String facultyId) {
        userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        facultyRepository.findById(facultyId)
                .orElseThrow(() -> new RuntimeException("Faculty not found with id: " + facultyId));

        userRepository.updateUserFaculty(userId, facultyId);
    }

    /**
     * Update user's college relationship (for students)
     */
    @Transactional
    public void updateUserCollege(String userId, String collegeId) {
        userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        collegeRepository.findById(collegeId)
                .orElseThrow(() -> new RuntimeException("College not found with id: " + collegeId));

        userRepository.updateUserCollege(userId, collegeId);
    }

    /**
     * Update user's working faculty relationship (for staff)
     */
    @Transactional
    public void updateUserWorkingFaculty(String userId, String facultyId) {
        userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        facultyRepository.findById(facultyId)
                .orElseThrow(() -> new RuntimeException("Faculty not found with id: " + facultyId));

        userRepository.updateUserWorkingFaculty(userId, facultyId);
    }

    /**
     * Update user's working college relationship (for staff)
     */
    @Transactional
    public void updateUserWorkingCollege(String userId, String collegeId) {
        userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        collegeRepository.findById(collegeId)
                .orElseThrow(() -> new RuntimeException("College not found with id: " + collegeId));

        userRepository.updateUserWorkingCollege(userId, collegeId);
    }

    /**
     * Update user's degree relationship
     */
    @Transactional
    public void updateUserDegree(String userId, String degreeId) {
        userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        degreeRepository.findById(degreeId)
                .orElseThrow(() -> new RuntimeException("Degree not found with id: " + degreeId));

        userRepository.updateUserDegree(userId, degreeId);
    }

    /**
     * Update user's position relationship
     */
    @Transactional
    public void updateUserPosition(String userId, String positionId) {
        userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        positionRepository.findById(positionId)
                .orElseThrow(() -> new RuntimeException("Position not found with id: " + positionId));

        userRepository.updateUserPosition(userId, positionId);
    }

    /**
     * Update user's academic relationship
     */
    @Transactional
    public void updateUserAcademic(String userId, String academicId) {
        userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        academicRepository.findById(academicId)
                .orElseThrow(() -> new RuntimeException("Academic not found with id: " + academicId));

        userRepository.updateUserAcademic(userId, academicId);
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
        // Kiểm tra xem có friend request ngược lại không (để auto-accept)

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
        // Check similarity attributes using proper getter methods
        candidateDTO.setSameCollege(Objects.equals(user.getCollegeName(), candidate.getCollegeName()));
        candidateDTO.setSameFaculty(Objects.equals(user.getFacultyName(), candidate.getFacultyName()));
        candidateDTO.setSameMajor(Objects.equals(user.getMajorName(), candidate.getMajorName()));
        candidateDTO.setSameBatch(Objects.equals(user.getBatchYear(), candidate.getBatchYear()));
    }

    /**
     * Check if user matches relationship filters
     */
    private boolean matchesFilters(UserEntity user, UserEntity candidate, RelationshipFilterDTO filters) {
        if (filters.getCollege() != null && !filters.getCollege().equals(candidate.getCollegeName())) {
            return false;
        }
        if (filters.getFaculty() != null && !filters.getFaculty().equals(candidate.getFacultyName())) {
            return false;
        }
        if (filters.getMajor() != null && !filters.getMajor().equals(candidate.getMajorName())) {
            return false;
        }
        if (filters.getBatch() != null && !filters.getBatch().equals(candidate.getBatchYear())) {
            return false;
        }
        return filters.getGender() == null || filters.getGender().equals(candidate.getGenderName());
    }

    /**
     * Map UserEntity to UserDTO
     */
    private UserDTO mapToDTO(UserEntity entity) {
        UserDTO dto = new UserDTO();
        dto.setId(entity.getId());
        dto.setEmail(entity.getEmail());
        dto.setUsername(entity.getUsername());
        dto.setFullName(entity.getFullName());
        dto.setRole(entity.getRole() != null ? entity.getRole().toString() : null);
        dto.setBio(entity.getBio());
        dto.setIsActive(entity.getIsActive());
        dto.setIsProfileCompleted(entity.getIsProfileCompleted());
        dto.setCreatedAt(entity.getCreatedAt());
        dto.setUpdatedAt(entity.getUpdatedAt());

        // Student fields
        dto.setStudentId(entity.getStudentId());
        dto.setMajor(entity.getMajor());
        dto.setBatch(entity.getBatch());
        // Lecturer fields
        dto.setStaffCode(entity.getStaffCode());
        dto.setAcademic(entity.getAcademic());
        dto.setDegree(entity.getDegree());
        dto.setPosition(entity.getPosition());

        // Common fields
        dto.setFaculty(entity.getFaculty());
        dto.setCollege(entity.getCollege());
        dto.setGender(entity.getGender());

        // Media fields
        dto.setAvatarUrl(entity.getAvatarUrl());
        dto.setBackgroundUrl(entity.getBackgroundUrl());

        // Friends mapping
        if (entity.getFriends() != null) {
            dto.setFriendIds(
                    entity.getFriends().stream()
                            .map(UserEntity::getId)
                            .collect(Collectors.toSet())
            );
        }

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
        entity.setFullName(dto.getFullName());

        // Handle role conversion safely
        if (dto.getRole() != null) {
            try {
                entity.setRole(Role.valueOf(dto.getRole()));
            } catch (IllegalArgumentException e) {
                entity.setRole(Role.USER); // Default fallback
            }
        }

        entity.setBio(dto.getBio());
        entity.setIsActive(dto.getIsActive());

        // Student fields
        entity.setStudentId(dto.getStudentId());

        // Faculty fields
        entity.setStaffCode(dto.getStaffCode());

        // Media fields
        entity.setAvatarUrl(dto.getAvatarUrl());
        entity.setBackgroundUrl(dto.getBackgroundUrl());

        // Note: Relationship mappings (major, batch, gender, etc.) should be handled
        // separately as they require database lookups to establish Neo4j relationships
        // This method only handles direct field mappings

        return entity;
    }

    public Boolean checkProfile(String currentUserId) {
        UserEntity userEntity = userRepository.findById(currentUserId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + currentUserId));

        // Map to DTO
        UserDTO userDTO = mapToDTO(userEntity);

        return userDTO.getIsProfileCompleted();
    }

    // ========================= PROFILE UPDATE METHODS =========================

    /**
     * Update student profile with event publishing for post-service synchronization
     */
    @Transactional
    public UserDTO updateStudentProfile(String userId, Object profileRequest) {
        UserEntity userEntity = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        // Convert Object to map for flexible handling
        @SuppressWarnings("unchecked")
        Map<String, Object> profileData = (Map<String, Object>) profileRequest;

        // Update basic profile fields
        if (profileData.containsKey("fullName")) {
            userEntity.setFullName((String) profileData.get("fullName"));
        }
        if (profileData.containsKey("bio")) {
            userEntity.setBio((String) profileData.get("bio"));
        }
        if (profileData.containsKey("studentId")) {
            userEntity.setStudentId((String) profileData.get("studentId"));
        }
        if (profileData.containsKey("avatarUrl")) {
            userEntity.setAvatarUrl((String) profileData.get("avatarUrl"));
        }
        if (profileData.containsKey("backgroundUrl")) {
            userEntity.setBackgroundUrl((String) profileData.get("backgroundUrl"));
        }

        // Update relationships if provided
        if (profileData.containsKey("majorCode")) {
            String majorCode = (String) profileData.get("majorCode");
            if (majorCode != null) {
                updateUserMajor(userId, majorCode);
            }
        }
        if (profileData.containsKey("facultyCode")) {
            String facultyCode = (String) profileData.get("facultyCode");
            if (facultyCode != null) {
                updateUserFaculty(userId, facultyCode);
            }
        }
        if (profileData.containsKey("collegeCode")) {
            String collegeCode = (String) profileData.get("collegeCode");
            if (collegeCode != null) {
                updateUserCollege(userId, collegeCode);
            }
        }
        if (profileData.containsKey("batchYear")) {
            String batchYear = (String) profileData.get("batchYear");
            if (batchYear != null) {
                updateUserBatch(userId, batchYear);
            }
        }
        if (profileData.containsKey("genderCode")) {
            String genderCode = (String) profileData.get("genderCode");
            if (genderCode != null) {
                updateUserGender(userId, genderCode);
            }
        }

        userEntity.setUpdatedAt(LocalDateTime.now());
        UserEntity updatedUser = userRepository.save(userEntity);

        // CRITICAL FIX: Publish profile update event for post-service synchronization
        publishProfileUpdateEvent(updatedUser);

        return mapToDTO(updatedUser);
    }

    /**
     * Update lecturer profile with event publishing for post-service synchronization
     */
    @Transactional
    public UserDTO updateLecturerProfile(String userId, Object profileRequest) {
        UserEntity userEntity = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        // Convert Object to map for flexible handling
        @SuppressWarnings("unchecked")
        Map<String, Object> profileData = (Map<String, Object>) profileRequest;

        // Update basic profile fields
        if (profileData.containsKey("fullName")) {
            userEntity.setFullName((String) profileData.get("fullName"));
        }
        if (profileData.containsKey("bio")) {
            userEntity.setBio((String) profileData.get("bio"));
        }
        if (profileData.containsKey("staffCode")) {
            userEntity.setStaffCode((String) profileData.get("staffCode"));
        }
        if (profileData.containsKey("avatarUrl")) {
            userEntity.setAvatarUrl((String) profileData.get("avatarUrl"));
        }
        if (profileData.containsKey("backgroundUrl")) {
            userEntity.setBackgroundUrl((String) profileData.get("backgroundUrl"));
        }

        // Update relationships if provided
        if (profileData.containsKey("facultyCode")) {
            String facultyCode = (String) profileData.get("facultyCode");
            if (facultyCode != null) {
                updateUserWorkingFaculty(userId, facultyCode);
            }
        }
        if (profileData.containsKey("positionCode")) {
            String positionCode = (String) profileData.get("positionCode");
            if (positionCode != null) {
                updateUserPosition(userId, positionCode);
            }
        }
        if (profileData.containsKey("academicCode")) {
            String academicCode = (String) profileData.get("academicCode");
            if (academicCode != null) {
                updateUserAcademic(userId, academicCode);
            }
        }
        if (profileData.containsKey("degreeCode")) {
            String degreeCode = (String) profileData.get("degreeCode");
            if (degreeCode != null) {
                updateUserDegree(userId, degreeCode);
            }
        }
        if (profileData.containsKey("genderCode")) {
            String genderCode = (String) profileData.get("genderCode");
            if (genderCode != null) {
                updateUserGender(userId, genderCode);
            }
        }

        userEntity.setUpdatedAt(LocalDateTime.now());
        UserEntity updatedUser = userRepository.save(userEntity);

        // CRITICAL FIX: Publish profile update event for post-service synchronization
        publishProfileUpdateEvent(updatedUser);

        return mapToDTO(updatedUser);
    }

    /**
     * Update general user profile with event publishing
     */
    @Transactional
    public UserDTO updateUserProfile(String userId, UserDTO userDTO) {
        UserEntity userEntity = userRepository.findById(userId)
                .orElseThrow(() -> new RuntimeException("User not found with id: " + userId));

        // Update basic profile fields
        if (userDTO.getFullName() != null) userEntity.setFullName(userDTO.getFullName());
        if (userDTO.getEmail() != null) userEntity.setEmail(userDTO.getEmail());
        if (userDTO.getUsername() != null) userEntity.setUsername(userDTO.getUsername());
        if (userDTO.getBio() != null) userEntity.setBio(userDTO.getBio());
        if (userDTO.getAvatarUrl() != null) userEntity.setAvatarUrl(userDTO.getAvatarUrl());
        if (userDTO.getBackgroundUrl() != null) userEntity.setBackgroundUrl(userDTO.getBackgroundUrl());

        userEntity.setUpdatedAt(LocalDateTime.now());
        UserEntity updatedUser = userRepository.save(userEntity);

        // CRITICAL FIX: Publish profile update event for post-service synchronization
        publishProfileUpdateEvent(updatedUser);

        return mapToDTO(updatedUser);
    }

    /**
     * Publish profile update event to notify post-service
     * This is the CRITICAL method that fixes AuthorInfo synchronization
     */
    private void publishProfileUpdateEvent(UserEntity user) {
        try {
            userEventPublisher.publishUserProfileUpdatedEventForPostService(
                user.getId(),
                user.getFullName() != null ? user.getFullName() : "",
                user.getEmail() != null ? user.getEmail() : "",
                user.getUsername() != null ? user.getUsername() : "",
                user.getAvatarUrl() != null ? user.getAvatarUrl() : "",
                user.getRole() != null ? user.getRole().name() : "USER"
            );
            log.info("Published profile update event for user: {} ({})", user.getId(), user.getFullName());
        } catch (Exception e) {
            log.error("Failed to publish profile update event for user {}: {}", user.getId(), e.getMessage(), e);
            // Don't throw exception to avoid breaking the profile update
        }
    }

    /**
     * Get friend IDs for a user (for internal service communication)
     */
    public Set<String> getFriendIds(String userId) {
        Optional<UserEntity> userOpt = userRepository.findById(userId);
        if (userOpt.isPresent()) {
            return userOpt.get().getFriendIds();
        }
        return new HashSet<>();
    }

    /**
     * Get users with close interactions (for feed ranking)
     */
    public Set<String> getCloseInteractionIds(String userId) {
        // This would typically analyze interaction patterns
        // For now, returning a subset of friends with high interaction
        Set<String> friendIds = getFriendIds(userId);
        return friendIds.stream()
                .limit(10) // Top 10 close interactions
                .collect(Collectors.toSet());
    }

    /**
     * Get users from same faculty (for post-service news feed algorithm)
     */
    public Set<String> getSameFacultyUserIds(String userId) {
        Optional<UserEntity> userOpt = userRepository.findById(userId);
        if (userOpt.isEmpty()) {
            return new HashSet<>();
        }

        UserEntity user = userOpt.get();
        if (user.getFaculty() == null) {
            return new HashSet<>();
        }

        // Find all users in the same faculty
        List<UserEntity> sameFacultyUsers = userRepository.findUsersByFaculty(user.getFaculty().getId());
        return sameFacultyUsers.stream()
                .map(UserEntity::getId)
                .filter(id -> !id.equals(userId)) // Exclude self
                .collect(Collectors.toSet());
    }

    /**
     * Get users from same major (for post-service news feed algorithm)
     */
    public Set<String> getSameMajorUserIds(String userId) {
        Optional<UserEntity> userOpt = userRepository.findById(userId);
        if (userOpt.isEmpty()) {
            return new HashSet<>();
        }

        UserEntity user = userOpt.get();
        if (user.getMajor() == null) {
            return new HashSet<>();
        }

        // Find all users in the same major
        List<UserEntity> sameMajorUsers = userRepository.findUsersByMajor(user.getMajor().getId());
        return sameMajorUsers.stream()
                .map(UserEntity::getId)
                .filter(id -> !id.equals(userId)) // Exclude self
                .collect(Collectors.toSet());
    }

    /**
     * Get user interest tags (for post-service content recommendation)
     */
    public Set<String> getUserInterestTags(String userId) {
        Optional<UserEntity> userOpt = userRepository.findById(userId);
        if (userOpt.isEmpty()) {
            return new HashSet<>();
        }

        UserEntity user = userOpt.get();
        Set<String> interestTags = new HashSet<>();

        // Add tags based on user's major and faculty
        if (user.getMajor() != null) {
            interestTags.add(user.getMajor().getName().toLowerCase());
            interestTags.add(user.getMajor().getId().toLowerCase());
        }

        if (user.getFaculty() != null) {
            interestTags.add(user.getFaculty().getName().toLowerCase());
        }

        // Add role-based tags
        if (user.getRole() != null) {
            interestTags.add(user.getRole().toString().toLowerCase());
        }

        return interestTags;
    }

    /**
     * Get user preferred categories (for post-service content filtering)
     */
    public Set<String> getUserPreferredCategories(String userId) {
        Optional<UserEntity> userOpt = userRepository.findById(userId);
        if (userOpt.isEmpty()) {
            return new HashSet<>();
        }

        UserEntity user = userOpt.get();
        Set<String> preferredCategories = new HashSet<>();

        // Add categories based on user profile
        if (user.isStudent()) {
            preferredCategories.add("academic");
            preferredCategories.add("student_life");
            if (user.getMajor() != null) {
                preferredCategories.add(user.getMajor().getName().toLowerCase().replace(" ", "_"));
            }
        } else if (user.isFaculty()) {
            preferredCategories.add("academic");
            preferredCategories.add("research");
            preferredCategories.add("teaching");
        }

        // Add general categories
        preferredCategories.add("general");
        preferredCategories.add("announcements");

        return preferredCategories;
    }

    /**
     * Get user's faculty ID (for post-service group filtering)
     */
    public String getUserFacultyId(String userId) {
        Optional<UserEntity> userOpt = userRepository.findById(userId);
        if (userOpt.isPresent() && userOpt.get().getFaculty() != null) {
            return userOpt.get().getFaculty().getId();
        }
        return null;
    }

    /**
     * Get user's major ID (for post-service group filtering)
     */
    public String getUserMajorId(String userId) {
        Optional<UserEntity> userOpt = userRepository.findById(userId);
        if (userOpt.isPresent() && userOpt.get().getMajor() != null) {
            return userOpt.get().getMajor().getId();
        }
        return null;
    }

    /**
     * Enhanced user search with academic context
     */
    public List<UserDTO> searchUsersWithContext(String query, String faculty, String major,
                                               String batch, String currentUserId, int page, int size) {
        // This would implement complex search with Neo4j queries
        // For now, implementing basic search
        List<UserEntity> users = userRepository.findByFullNameContainingIgnoreCase(query);

        // Apply filters
        if (faculty != null && !faculty.isEmpty()) {
            users = users.stream()
                    .filter(user -> faculty.equals(user.getFacultyId()))
                    .collect(Collectors.toList());
        }

        if (major != null && !major.isEmpty()) {
            users = users.stream()
                    .filter(user -> major.equals(user.getMajorId()))
                    .collect(Collectors.toList());
        }

        if (batch != null && !batch.isEmpty()) {
            users = users.stream()
                    .filter(user -> batch.equals(user.getBatchId()))
                    .collect(Collectors.toList());
        }

        // Apply pagination and convert to DTOs
        return users.stream()
                .skip(page * size)
                .limit(size)
                .map(this::mapToDTO)
                .collect(Collectors.toList());
    }

    /**
     * Send friend request
     */
    /*@Transactional
    public void sendFriendRequest(String fromUserId, String toUserId) {
        // This would typically create a friend request relationship in Neo4j
        // For now, just publishing an event
        userEventPublisher.publishUserRelationshipChangedEvent(
                fromUserId, toUserId, "FRIEND_REQUEST_SENT");

        log.info("Friend request sent from {} to {}", fromUserId, toUserId);
    }
*/
    /**
     * Accept friend request
     */
    /*@Transactional
    public void acceptFriendRequest(String fromUserId, String toUserId) {
        // This would typically update the relationship in Neo4j
        // For now, just publishing an event
        userEventPublisher.publishUserRelationshipChangedEvent(
                toUserId, fromUserId, "FRIEND_REQUEST_ACCEPTED");

        log.info("Friend request accepted: {} and {} are now friends", fromUserId, toUserId);
    }
*/
    /**
     * Get user activity feed (for profile timeline)
     */
    public List<ActivityDTO> getUserActivity(String userId, String viewerId, int page, int size) {
        // This would typically query activity logs or events
        // For now, returning mock activity data
        List<ActivityDTO> activities = new ArrayList<>();

        activities.add(ActivityDTO.builder()
                .id("activity_1")
                .userId(userId)
                .activityType("POST_CREATED")
                .entityType("POST")
                .entityId("post_123")
                .description("Created a new post")
                .timestamp(LocalDateTime.now().minusDays(1))
                .build());

        activities.add(ActivityDTO.builder()
                .id("activity_2")
                .userId(userId)
                .activityType("FRIEND_ADDED")
                .entityType("USER")
                .entityId("user_456")
                .description("Added a new friend")
                .timestamp(LocalDateTime.now().minusDays(2))
                .build());

        return activities.stream()
                .skip(page * size)
                .limit(size)
                .collect(Collectors.toList());
    }

}
