package com.ctuconnect.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.GenericGenerator;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.Column;


@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Entity
@Table(name = "users",
       uniqueConstraints = {
           @UniqueConstraint(columnNames = {"email"}),
           @UniqueConstraint(columnNames = {"username"})
       })
public class UserEntity {

    @Id
    @GeneratedValue(generator = "uuid2")
    @GenericGenerator(name = "uuid2", strategy = "uuid2")
    @Column(columnDefinition = "uuid", updatable = false, nullable = false)
    private UUID id;

    @Column(nullable = false, unique = true, length = 50)
    private String email;

    @Column(nullable = false, unique = true, length = 25)
    private String username;

    @Column(nullable = false)
    private String password;

    @Column(nullable = false, length = 20)
    private String role = "USER";

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;

    @Column(name = "is_active", nullable = false)
    private Boolean isActive = true;

    @OneToMany(mappedBy = "user", cascade = CascadeType.ALL, orphanRemoval = true, fetch = FetchType.LAZY)
    private List<RefreshTokenEntity> refreshTokens = new ArrayList<>();

    @OneToMany(mappedBy = "user", cascade = CascadeType.ALL, orphanRemoval = true, fetch = FetchType.LAZY)
    private List<EmailVerificationEntity> emailVerifications = new ArrayList<>();

    @PrePersist
    protected void onCreate() {
        if (id == null) {
            id = UUID.randomUUID();
        }
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }

    // Custom getter for isActive to follow naming convention
    public boolean isActive() {
        return isActive != null && isActive;
    }

    // Custom setter for isActive
    public void setActive(Boolean active) {
        this.isActive = active;
    }

    // Phương thức để đảm bảo email luôn được lưu dưới dạng lowercase
    public void setEmail(String email) {
        this.email = email != null ? email.toLowerCase().trim() : null;
    }

    // Phương thức để đảm bảo username luôn được lưu dưới dạng lowercase
    public void setUsername(String username) {
        this.username = username != null ? username.toLowerCase().trim() : null;
    }
}
