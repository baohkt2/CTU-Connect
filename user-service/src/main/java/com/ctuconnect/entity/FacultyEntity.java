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
    private String code;

    private String name;

    @Relationship(type = "BELONGS_TO", direction = Relationship.Direction.OUTGOING)
    private CollegeEntity college;
}
