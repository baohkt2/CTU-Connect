package com.ctuconnect.entity;

import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;
import org.springframework.data.neo4j.core.schema.Relationship;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

import java.util.HashSet;
import java.util.Set;

@Node("Faculty")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FacultyEntity {
    @Id
    @NotBlank(message = "Faculty name is required")
    @Size(max = 100, message = "Faculty name must not exceed 100 characters")
    private String name;

    @Size(max = 10, message = "Faculty code must not exceed 10 characters")
    private String code;

    @Size(max = 500, message = "Description must not exceed 500 characters")
    private String description;

    @Relationship(type = "HAS_FACULTY", direction = Relationship.Direction.INCOMING)
    private CollegeEntity college;

    @Relationship(type = "HAS_MAJOR", direction = Relationship.Direction.OUTGOING)
    @Builder.Default
    private Set<MajorEntity> majors = new HashSet<>();

    // Safe getter methods
    public String getName() {
        return name != null ? name : "";
    }

    public String getCode() {
        return code != null ? code : "";
    }

    public String getDescription() {
        return description != null ? description : "";
    }

    public Set<MajorEntity> getMajors() {
        return majors != null ? majors : new HashSet<>();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        FacultyEntity that = (FacultyEntity) o;
        return name != null && name.equals(that.name);
    }

    @Override
    public int hashCode() {
        return name != null ? name.hashCode() : 0;
    }
}
