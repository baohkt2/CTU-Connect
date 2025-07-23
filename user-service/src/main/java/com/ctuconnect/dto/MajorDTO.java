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
public class MajorDTO {
    @NotBlank(message = "Major code is required")
    private String code;

    @NotBlank(message = "Major name is required")
    private String name;

    @NotBlank(message = "Faculty code is required")
    private String facultyCode;

    private String facultyName;
    private String collegeCode;
    private String collegeName;
}
