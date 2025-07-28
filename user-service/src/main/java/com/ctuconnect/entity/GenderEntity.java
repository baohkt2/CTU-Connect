package com.ctuconnect.entity;

import lombok.*;
import org.springframework.data.neo4j.core.schema.*;

@Node("Gender")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class GenderEntity {
    @Id
    private String code; // "M" or "F"

    private String name; // "Nam", "Nữ"

    @Relationship(type = "HAS_GENDER", direction = Relationship.Direction.INCOMING)
    private UserEntity user; // Người dùng có giới tính này
}
