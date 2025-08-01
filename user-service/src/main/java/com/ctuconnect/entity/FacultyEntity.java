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

    private String name;
    @Id
    private String code;

    private String college; // Tên college mà faculty thuộc về (theo database structure)

    @Relationship(type = "HAS_FACULTY", direction = Relationship.Direction.INCOMING)
    private CollegeEntity collegeEntity;

    @Relationship(type = "HAS_MAJOR", direction = Relationship.Direction.OUTGOING)
    private List<MajorEntity> majors;

    @Relationship(type = "WORKS_IN", direction = Relationship.Direction.INCOMING)
    private List<UserEntity> users; // Danh sách nhân viên làm việc tại faculty

    public String getId() {
        return code; // Sử dụng code làm ID duy nhất cho Faculty
    }
}
