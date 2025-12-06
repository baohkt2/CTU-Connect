package com.ctuconnect.dto;

import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import com.fasterxml.jackson.annotation.JsonInclude;

import jakarta.validation.constraints.Size;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class UserUpdateDTO {
    @Size(max = 100, message = "Full name must not exceed 100 characters")
    private String fullName;

    @Size(max = 500, message = "Bio must not exceed 500 characters")
    private String bio;

    @Size(max = 20, message = "Student ID must not exceed 20 characters")
    private String studentId;

    private String majorCode;  // Changed from majorName to majorCode
    private String batchYear;
    private String genderName;
}
