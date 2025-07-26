package com.ctuconnect.entity;

import lombok.*;
import org.springframework.data.neo4j.core.schema.*;

import java.util.List;

@Node("Faculty")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class FacultyEntity {

    private String name; // Sử dụng name làm ID như trong database
    @Id
    private String code;

    private String college; // Tên college mà faculty thuộc về (theo database structure)

    @Relationship(type = "HAS_FACULTY", direction = Relationship.Direction.INCOMING)
    private CollegeEntity collegeEntity;

    @Relationship(type = "HAS_MAJOR", direction = Relationship.Direction.OUTGOING)
    private List<MajorEntity> majors;
}
