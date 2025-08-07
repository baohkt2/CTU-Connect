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

    private String name;
    @Id
    private String code;

    @Relationship(type = "HAS_FACULTY", direction = Relationship.Direction.OUTGOING)
    private List<FacultyEntity> faculties;

    @Relationship(type = "BELONGS_TO", direction = Relationship.Direction.INCOMING)
    private List<UserEntity> students; // Danh sách sinh viên thuộc college

    public String getId() {
        return code; // Trả về name làm ID
    }
}
