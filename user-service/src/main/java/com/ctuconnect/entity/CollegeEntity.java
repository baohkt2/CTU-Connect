package com.ctuconnect.entity;

import lombok.*;
import org.springframework.data.neo4j.core.schema.*;

@Node("College")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class CollegeEntity {
    @Id
    private String name; // Sử dụng name làm ID như trong database

    // Có thể thêm code riêng nếu cần
    private String code;
}
