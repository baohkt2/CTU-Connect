package com.ctuconnect.dto;

import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RelationshipFilterDTO {
    // Boolean flags for relationship types
    private boolean isFriend;
    private boolean isSameCollege;
    private boolean isSameFaculty;
    private boolean isSameMajor;
    private boolean isSameBatch;

    // Specific filter values
    private String college;
    private String faculty;
    private String major;
    private String batch;
    private String gender;
    private String role;

    // Additional options
    @Builder.Default
    private Integer page = 0;

    @Builder.Default
    private Integer size = 20;
}
