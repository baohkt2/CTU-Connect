package com.ctuconnect.entity;

import lombok.*;
import org.springframework.data.neo4j.core.schema.*;

import java.util.List;

@Node("College")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class CollegeEntity {

    private String name; // Sử dụng name làm ID như trong database
    @Id
    private String code;

    @Relationship(type = "HAS_FACULTY", direction = Relationship.Direction.OUTGOING)
    private List<FacultyEntity> faculties;
}
