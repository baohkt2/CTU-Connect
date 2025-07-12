package com.ctuconnect.dto;

import lombok.Data;

@Data
public class RelationshipFilterDTO {
    // Boolean flags for relationship types
    private boolean isFriend = false;
    private boolean isSameCollege = false;
    private boolean isSameFaculty = false;
    private boolean isSameMajor = false;
    private boolean isSameBatch = false;

    // Specific filter values
    private String college;
    private String faculty;
    private String major;
    private Integer batch; // Changed to Integer to match Neo4j dataset
    private String gender;
    private String role;

    // Additional options
    private Integer page = 0;
    private Integer size = 20;
}
