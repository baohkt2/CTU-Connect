package com.ctuconnect.entity;


import jakarta.persistence.*;
import lombok.Data;

@Entity
@Table (name = "email_verification")
@Data
public class EmailVertify {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "token", nullable = false, unique = true)
    private String token;

    @Column(name = "user_id", nullable = false)
    private Long userId;

    @Column(name = "expiry_date", nullable = false)
    private Long expiryDate;

    @Column(name = "is_verified", nullable = false)
    private boolean isVerified;

    @Column(name = "created_at", nullable = false)
    private java.time.LocalDateTime createdAt = java.time.LocalDateTime.now();
}
