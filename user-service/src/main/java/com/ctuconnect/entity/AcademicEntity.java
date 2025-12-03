package com.ctuconnect.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;
import org.springframework.data.neo4j.core.schema.Relationship;

@Node("Academic")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class AcademicEntity {

    private String name; // Tên học vị, ví dụ: "Cử nhân", "Thạc sĩ", "Tiến sĩ"
    @Id
    private String code; // Mã định danh cho học vị, ví dụ: "BACHELOR", "MASTER", "PHD"

    private String description; // Mô tả chi tiết về học vị

    @Relationship(type = "HAS_DEGREE", direction = Relationship.Direction.INCOMING)
    private DegreeEntity degree;

    public String getId() {
        return code;
    }
}
