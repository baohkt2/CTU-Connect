package com.ctuconnect.entity;

import lombok.*;
import org.springframework.data.neo4j.core.schema.*;

@Node("Batch")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class BatchEntity  {
    @Id
    private String year;

    @Relationship(type = "IN_BATCH", direction = Relationship.Direction.INCOMING)
    private UserEntity students;
}
