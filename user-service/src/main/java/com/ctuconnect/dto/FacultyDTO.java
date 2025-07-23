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
    @NotBlank(message = "Faculty name is required")
    private String name; // Sử dụng name làm identifier chính

    private String code; // Code tùy chọn cho frontend

    @NotBlank(message = "College name is required")
    private String collegeName; // Tên college mà faculty thuộc về
}
