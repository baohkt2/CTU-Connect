package com.ctuconnect.entity;

import lombok.*;
import org.springframework.data.neo4j.core.schema.*;

@Node("Faculty")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class FacultyEntity {
    @Id
    private String name; // Sử dụng name làm ID như trong database

    private String code; // Có thể thêm code riêng nếu cần

    private String college; // Tên college mà faculty thuộc về (theo database structure)

    @Relationship(type = "HAS_FACULTY", direction = Relationship.Direction.INCOMING)
    private CollegeEntity collegeEntity;
}
