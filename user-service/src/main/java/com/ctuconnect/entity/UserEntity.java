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
    private String id; // UUID từ auth-service

    private String email;
    private String username;
    private String fullName;
    private Boolean isActive;
    private Role role;
    private String bio;
    private Boolean isProfileCompleted = false; // Mặc định false

    // Ảnh đại diện (tùy chọn)
    private String avatarUrl;
    private String backgroundUrl;

    // ==== Trường riêng của sinh viên ====
    private String studentId;

    // ==== Trường riêng của giảng viên/cán bộ ====
    private String staffCode;
    private String position;        // Giảng viên, Trợ lý, Cán bộ,...
    private String academicTitle;   // Giáo sư, PGS,...
    private String degree;          // Tiến sĩ, Thạc sĩ,...

    // ==== Thời gian ====
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    // ==== RELATIONSHIPS ====

    @Relationship(type = "ENROLLED_IN", direction = Relationship.Direction.OUTGOING)
    private MajorEntity major;

    @Relationship(type = "IN_BATCH", direction = Relationship.Direction.OUTGOING)
    private BatchEntity batch;

    @Relationship(type = "HAS_GENDER", direction = Relationship.Direction.OUTGOING)
    private GenderEntity gender;

    @Relationship(type = "WORKS_IN", direction = Relationship.Direction.OUTGOING)
    private FacultyEntity workingFaculty; // dùng cho FACULTY

    @Relationship(type = "FRIEND", direction = Relationship.Direction.OUTGOING)
    private Set<UserEntity> friends = new HashSet<>();

    // ==== Factory method ====

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

    public void updateTimestamp() {
        this.updatedAt = LocalDateTime.now();
    }

    // ==== Helper logic ====

    public boolean isStudent() {
        return Role.STUDENT.equals(this.role);
    }

    public boolean isFaculty() {
        return Role.FACULTY.equals(this.role);
    }

    public boolean isAdmin() {
        return Role.ADMIN.equals(this.role);
    }

    public String getMajorName() {
        return major != null ? major.getName() : null;
    }

    public String getMajorCode() {
        return major != null ? major.getCode() : null;
    }

    public String getBatchYear() {
        return batch != null ? String.valueOf(batch.getYear()) : null;
    }

    public String getBatch() {
        return getBatchYear();
    }

    public String getGenderCode() {
        return gender != null ? gender.getCode() : null;
    }

    public String getGenderName() {
        return gender != null ? gender.getName() : null;
    }

    public String getFacultyName() {
        if (major != null && major.getFaculty() != null)
            return major.getFaculty().getName();
        return null;
    }

    public String getFacultyCode() {
        if (major != null && major.getFaculty() != null)
            return major.getFaculty().getCode();
        return null;
    }

    public String getCollegeName() {
        if (major != null && major.getFaculty() != null && major.getFaculty().getCollege() != null)
            return major.getFaculty().getCollege().getName();
        return null;
    }

    public String getCollegeCode() {
        if (major != null && major.getFaculty() != null && major.getFaculty().getCollege() != null)
            return major.getFaculty().getCollege().getCode();
        return null;
    }

    public String getDepartment() {
        return workingFaculty != null ? workingFaculty.getName() : getFacultyName();
    }
}
