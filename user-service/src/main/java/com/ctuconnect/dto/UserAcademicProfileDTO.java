package com.ctuconnect.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

/**
 * Academic profile DTO for recommend-service integration
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UserAcademicProfileDTO implements Serializable {
    private String userId;
    private String major;
    private String faculty;
    private String degree;
    private String batch;
    private String studentId;
}
