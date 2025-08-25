package com.ctuconnect.entity;

import jakarta.validation.constraints.Min;
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

@Node("University")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UniversityEntity {
    @Id
    @NotBlank(message = "University name is required")
    @Size(max = 100, message = "University name must not exceed 100 characters")
    private String name;

    @Size(max = 10, message = "University code must not exceed 10 characters")
    private String code;

    @Min(value = 1800, message = "Establishment year must be after 1800")
    private Integer established;

    @Size(max = 500, message = "Description must not exceed 500 characters")
    private String description;

    @Size(max = 200, message = "Address must not exceed 200 characters")
    private String address;

    @Size(max = 100, message = "Website must not exceed 100 characters")
    private String website;

    @Relationship(type = "HAS_COLLEGE", direction = Relationship.Direction.OUTGOING)
    @Builder.Default
    private Set<CollegeEntity> colleges = new HashSet<>();

    // Safe getter methods
    public String getName() {
        return name != null ? name : "";
    }

    public String getCode() {
        return code != null ? code : "";
    }

    public Integer getEstablished() {
        return established != null ? established : 0;
    }

    public String getDescription() {
        return description != null ? description : "";
    }

    public String getAddress() {
        return address != null ? address : "";
    }

    public String getWebsite() {
        return website != null ? website : "";
    }

    public Set<CollegeEntity> getColleges() {
        return colleges != null ? colleges : new HashSet<>();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        UniversityEntity that = (UniversityEntity) o;
        return name != null && name.equals(that.name);
    }

    @Override
    public int hashCode() {
        return name != null ? name.hashCode() : 0;
    }
}
