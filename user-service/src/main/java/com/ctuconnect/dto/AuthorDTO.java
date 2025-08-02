package com.ctuconnect.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
@Builder
public class AuthorDTO {
    private String id;
    private String name;
    private String avatar;
    private String fullName; // Added for consistency
    private String avatarUrl; // Added for consistency
    private String role; // Added for consistency
}
