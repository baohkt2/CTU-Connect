package com.ctuconnect.dto;


import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class AuthorInfo {
    private String id;
    private String name;
    private String avatar;
    private String fullName; // Added for consistency
    private String avatarUrl; // Added for consistency

    // Backward compatibility methods
    public String getFullName() {
        return fullName != null ? fullName : name;
    }

    public String getAvatarUrl() {
        return avatarUrl != null ? avatarUrl : avatar;
    }

    public void setFullName(String fullName) {
        this.fullName = fullName;
        this.name = fullName; // Keep both fields in sync
    }

    public void setAvatarUrl(String avatarUrl) {
        this.avatarUrl = avatarUrl;
        this.avatar = avatarUrl; // Keep both fields in sync
    }
}
