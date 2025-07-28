package com.ctuconnect.entity;

import lombok.*;
import org.springframework.data.neo4j.core.schema.*;

@Node("Major")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class MajorEntity {

    private String name; // Sử dụng name làm ID như trong database
    @Id
    private String code;

    private String faculty; // Tên faculty mà major thuộc về (theo database structure)

    @Relationship(type = "HAS_MAJOR", direction = Relationship.Direction.INCOMING)
    private FacultyEntity facultyEntity;

    @Relationship(type = "ENROLLED_IN", direction = Relationship.Direction.INCOMING)
    private UserEntity users; // Sinh viên thuộc major này
}
