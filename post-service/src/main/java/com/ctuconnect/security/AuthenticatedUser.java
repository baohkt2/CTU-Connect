package com.ctuconnect.security;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Represents the authenticated user information passed from the gateway
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class AuthenticatedUser {
    private String userId;
    private String email;
    private String role;
    private String fullName; // Added for notification messages

    // Constructor for backward compatibility
    public AuthenticatedUser(String userId, String email, String role) {
        this.userId = userId;
        this.email = email;
        this.role = role;
    }

    public String getId() {
        return this.userId;
    }

    public String getFullName() {
        return this.fullName != null ? this.fullName : this.email;
    }

    public boolean hasRole(String role) {
        return this.role != null && this.role.equalsIgnoreCase(role);
    }

    public boolean isAdmin() {
        return hasRole("ADMIN");
    }

    public boolean isUser() {
        return hasRole("USER");
    }

    public boolean isAuthenticated() {
        return userId != null && !userId.isEmpty() &&
               email != null && !email.isEmpty() &&
               role != null && !role.isEmpty();
    }
}
