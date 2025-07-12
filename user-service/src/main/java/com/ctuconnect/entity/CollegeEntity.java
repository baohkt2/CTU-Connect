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

@Node("College")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CollegeEntity {
    @Id
    @NotBlank(message = "College name is required")
    @Size(max = 100, message = "College name must not exceed 100 characters")
    private String name;

    @Size(max = 10, message = "College code must not exceed 10 characters")
    private String code;

    @Size(max = 500, message = "Description must not exceed 500 characters")
    private String description;

    @Relationship(type = "HAS_COLLEGE", direction = Relationship.Direction.INCOMING)
    private UniversityEntity university;

    @Relationship(type = "HAS_FACULTY", direction = Relationship.Direction.OUTGOING)
    @Builder.Default
    private Set<FacultyEntity> faculties = new HashSet<>();

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

    public Set<FacultyEntity> getFaculties() {
        return faculties != null ? faculties : new HashSet<>();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        CollegeEntity that = (CollegeEntity) o;
        return name != null && name.equals(that.name);
    }

    @Override
    public int hashCode() {
        return name != null ? name.hashCode() : 0;
    }
}
