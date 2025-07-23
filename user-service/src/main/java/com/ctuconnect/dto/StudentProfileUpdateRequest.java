package com.ctuconnect.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class StudentProfileUpdateRequest {
    @NotBlank(message = "Full name is required")
    private String fullName;

    private String bio;

    @NotBlank(message = "Student ID is required")
    private String studentId;

    @NotBlank(message = "Major name is required")
    private String majorName; // Đổi từ majorCode sang majorName

    @NotNull(message = "Batch year is required")
    private Integer batchYear;

    @NotBlank(message = "Gender code is required")
    private String genderCode;

    private String avatarUrl;
    private String backgroundUrl;
}
