package com.ctuconnect.dto;

import com.ctuconnect.security.AuthenticatedUser;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * DTO cho response của authentication context
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class AuthContextDTO {
    private String userId;
    private String email;
    private String role;
    private boolean authenticated;

    public static AuthContextDTO fromAuthenticatedUser(AuthenticatedUser user) {
        if (user == null || !user.isAuthenticated()) {
            return new AuthContextDTO(null, null, null, false);
        }
        return new AuthContextDTO(
                user.getUserId(),
                user.getEmail(),
                user.getRole(),
                true
        );
    }
}
