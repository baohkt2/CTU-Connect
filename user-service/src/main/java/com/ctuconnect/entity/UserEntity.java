package com.ctuconnect.entity;

import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
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
    private String id; // UUID string từ auth-service

    // Username from auth service
    private String username;

    // Active status
    private Boolean isActive;

    // Existing fields
    private String email;
    private String studentId;

    private String batch;
    private String fullName;
    private String role;
    private String college;
    private String faculty;
    private String major;
    private String gender;
    private String bio;

    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    @Relationship(type = "FRIEND", direction = Relationship.Direction.OUTGOING)
    private Set<UserEntity> friends = new HashSet<>();

    // Utility method to ensure ID is set when creating from auth service
    public static UserEntity fromAuthService(String authUserId, String email, String username, String role) {
        return UserEntity.builder()
                .id(authUserId) // Use UUID from auth service
                .email(email)
                .username(username)
                .role(role)
                .isActive(true)
                .createdAt(LocalDateTime.now())
                .updatedAt(LocalDateTime.now())
                .friends(new HashSet<>())
                .build();
    }

    // Method to update timestamp
    public void updateTimestamp() {
        this.updatedAt = LocalDateTime.now();
    }
}