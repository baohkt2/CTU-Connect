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
public class GenderDTO {
    @NotBlank(message = "Gender code is required")
    private String code;

    @NotBlank(message = "Gender name is required")
    private String name;
}
