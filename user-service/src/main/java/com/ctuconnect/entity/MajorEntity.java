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
    private String code; // e.g., CNPM01

    private String name;

    @Relationship(type = "BELONGS_TO", direction = Relationship.Direction.OUTGOING)
    private FacultyEntity faculty;
}
