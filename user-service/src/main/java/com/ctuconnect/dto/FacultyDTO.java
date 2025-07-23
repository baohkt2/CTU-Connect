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
public class FacultyDTO {
    @NotBlank(message = "Faculty code is required")
    private String code;

    @NotBlank(message = "Faculty name is required")
    private String name;

    @NotBlank(message = "College code is required")
    private String collegeCode;

    private String collegeName;
}
