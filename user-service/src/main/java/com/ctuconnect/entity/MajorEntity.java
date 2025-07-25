package com.ctuconnect.entity;

import lombok.*;
import org.springframework.data.neo4j.core.schema.*;

@Node("Major")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class MajorEntity {
    @Id
    private String name; // Sử dụng name làm ID như trong database

    private String code;

    private String faculty; // Tên faculty mà major thuộc về (theo database structure)

    @Relationship(type = "HAS_MAJOR", direction = Relationship.Direction.INCOMING)
    private FacultyEntity facultyEntity;
}
