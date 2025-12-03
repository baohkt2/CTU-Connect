package com.ctuconnect.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.List;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class FacultyWithMajorsDTO {
    private String name; // Sử dụng name làm identifier chính
    private String code; // Code tùy chọn cho frontend
    private String collegeName; // Tên college mà faculty thuộc về
    private List<MajorDTO> majors;
}
