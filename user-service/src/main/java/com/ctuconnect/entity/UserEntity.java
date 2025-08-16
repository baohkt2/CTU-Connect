package com.ctuconnect.entity;

import com.ctuconnect.enums.Role;
import lombok.*;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;
import org.springframework.data.neo4j.core.schema.Relationship;

import java.time.LocalDateTime;
import java.util.HashSet;
import java.util.Set;
@Node("User")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UserEntity {
    @Id
    private String id;
    private String email;
    private String username;
    private String fullName;
    private Boolean isActive;
    private Role role;
    private String bio;
    private Boolean isProfileCompleted = false;
    private String avatarUrl;
    private String backgroundUrl;
    private String studentId;
    private String staffCode;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    @Relationship(type = "ENROLLED_IN", direction = Relationship.Direction.OUTGOING)
    private MajorEntity major;

    @Relationship(type = "IN_BATCH", direction = Relationship.Direction.OUTGOING)
    private BatchEntity batch;

    @Relationship(type = "HAS_GENDER", direction = Relationship.Direction.OUTGOING)
    private GenderEntity gender;

    @Relationship(type = "FRIEND_WITH", direction = Relationship.Direction.OUTGOING)
    private Set<UserEntity> friends = new HashSet<>();

    @Relationship(type = "BELONGS_TO", direction = Relationship.Direction.OUTGOING)
    private CollegeEntity college;

    @Relationship(type = "WORKS_IN", direction = Relationship.Direction.OUTGOING)
    private FacultyEntity faculty;

    @Relationship(type = "HAS_DEGREE", direction = Relationship.Direction.OUTGOING)
    private DegreeEntity degree;

    @Relationship(type = "HAS_POSITION", direction = Relationship.Direction.OUTGOING)
    private PositionEntity position;

    @Relationship(type = "HAS_ACADEMIC", direction = Relationship.Direction.OUTGOING)
    private AcademicEntity academic;

    // Factory method
    public static UserEntity fromAuthService(String authUserId, String email, String username, String roleString) {
        Role resolvedRole;
        try {
            resolvedRole = Role.fromString(roleString);
        } catch (IllegalArgumentException e) {
            resolvedRole = Role.USER;
        }
        return UserEntity.builder()
                .id(authUserId)
                .email(email)
                .username(username)
                .role(resolvedRole)
                .isActive(true)
                .isProfileCompleted(false)
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .build();
    }

    // Utility methods
    public String getFacultyId() { return faculty != null ? faculty.getId() : null; }
    public String getFacultyName() { return faculty != null ? faculty.getName() : null; }
    public String getMajorId() { return major != null ? major.getId() : null; }
    public String getMajorName() { return major != null ? major.getName() : null; }
    public String getCollegeId() { return college != null ? college.getId() : null; }
    public String getCollegeName() { return college != null ? college.getName() : null; }
    public String getBatchId() { return batch != null ? batch.getId() : null; }
    public String getBatchYear() { return batch != null ? String.valueOf(batch.getYear()) : null; }
    public String getGenderId() { return gender != null ? gender.getId() : null; }
    public String getGenderName() { return gender != null ? gender.getName() : null; }
    public String getDegreeId() { return degree != null ? degree.getId() : null; }
    public String getPositionId() { return position != null ? position.getId() : null; }
    public String getAcademicId() { return academic != null ? academic.getId() : null; }
    public Set<String> getFriendIds() {
        return friends == null ? new HashSet<>() : friends.stream().map(UserEntity::getId).collect(java.util.stream.Collectors.toSet());
    }

    // Role checks (giữ 1 bộ)
    public boolean isStudent() {
        return Role.STUDENT.equals(this.role);
    }
    public boolean isFaculty() {
        return Role.LECTURER.equals(this.role);
    }
    public boolean isAdmin() {
        return Role.ADMIN.equals(this.role);
    }

    // Tên code thêm nếu cần
    public String getMajorCode() {
        return major != null ? major.getCode() : null;
    }
    public String getGenderCode() {
        return gender != null ? gender.getCode() : null;
    }
    public String getFacultyCode() {
        return faculty != null ? faculty.getCode() : null;
    }
    public String getCollegeCode() {
        return college != null ? college.getCode() : null;
    }
    public String getDegreeName() {
        return degree != null ? degree.getName() : null;
    }
    public String getDegreeCode() {
        return degree != null ? degree.getCode() : null;
    }
    public String getPositionName() {
        return position != null ? position.getName() : null;
    }
    public String getPositionCode() {
        return position != null ? position.getCode() : null;
    }
    public String getAcademicName() {
        return academic != null ? academic.getName() : null;
    }
    public String getAcademicCode() {
        return academic != null ? academic.getCode() : null;
    }

    public void updateTimestamp() {
        this.updatedAt = LocalDateTime.now();
    }
}
