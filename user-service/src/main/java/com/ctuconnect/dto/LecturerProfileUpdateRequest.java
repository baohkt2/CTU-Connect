package com.ctuconnect.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import jakarta.validation.constraints.NotBlank;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class LecturerProfileUpdateRequest {
    @NotBlank(message = "Full name is required")
    private String fullName;

    private String bio;

    @NotBlank(message = "Staff code is required")
    private String staffCode;

    @NotBlank(message = "Position is required")
    private String positionCode;

    private String academicCode;
    private String degreeCode;

    @NotBlank(message = "Working faculty name is required")
    private String facultyCode; // Đổi từ workingFacultyCode sang workingFacultyName
    private String collegeCode;
    private String majorCode;
    private String batchYear;

    @NotBlank(message = "Gender code is required")
    private String genderCode;

    private String avatarUrl;
    private String backgroundUrl;
}
