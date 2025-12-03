package com.ctuconnect.dto;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;
import java.util.Set;

/**
 * DTO specifically designed for Cypher query results
 * Contains all user information in a flat structure for easy mapping
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class UserQueryResultDTO {
    // Basic user information
    private String id;
    private String email;
    private String username;
    private String studentId;
    private String fullName;
    private String role;
    private String bio;
    private Boolean isActive;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    // University structure information
    private String college;
    private String faculty;
    private String major;
    private String batch;
    private String gender;

    // Relationship counts
    private Integer friendsCount;
    private Integer sentRequestsCount;
    private Integer receivedRequestsCount;

    // For relationship analysis
    private Integer mutualFriendsCount;
    private Boolean sameCollege;
    private Boolean sameFaculty;
    private Boolean sameMajor;

    // Friend IDs (for specific queries)
    private Set<String> friendIds;

    /**
     * Convert to UserDTO for API responses
     */
    public UserDTO toUserDTO() {
        UserDTO dto = new UserDTO();
        dto.setId(this.id);
        dto.setEmail(this.email);
        dto.setUsername(this.username);
        dto.setStudentId(this.studentId);
        dto.setBatch(this.batch);
        dto.setFullName(this.fullName);
        dto.setRole(this.role);
        dto.setCollege(this.college);
        dto.setFaculty(this.faculty);
        dto.setMajor(this.major);
        dto.setGender(this.gender);
        dto.setBio(this.bio);
        dto.setIsActive(this.isActive);
        dto.setCreatedAt(this.createdAt);
        dto.setUpdatedAt(this.updatedAt);
        dto.setFriendIds(this.friendIds);
        dto.setMutualFriendsCount(this.mutualFriendsCount != null ? this.mutualFriendsCount : 0);
        dto.setSameCollege(this.sameCollege != null ? this.sameCollege : false);
        dto.setSameFaculty(this.sameFaculty != null ? this.sameFaculty : false);
        dto.setSameMajor(this.sameMajor != null ? this.sameMajor : false);
        return dto;
    }
}
