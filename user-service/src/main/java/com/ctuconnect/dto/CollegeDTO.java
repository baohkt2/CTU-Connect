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
public class CollegeDTO {
    @NotBlank(message = "College code is required")
    private String code;

    @NotBlank(message = "College name is required")
    private String name;
}
