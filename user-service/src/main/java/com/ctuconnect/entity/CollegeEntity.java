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
    private String code;

    private String name;
}
