package com.ctuconnect.entity;

import lombok.Data;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import org.springframework.data.neo4j.core.schema.Id;
import org.springframework.data.neo4j.core.schema.Node;
import org.springframework.data.neo4j.core.schema.Relationship;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;
import java.util.HashSet;
import java.util.Set;

@Node("Batch")
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class BatchEntity {
    @Id
    @NotNull(message = "Batch year is required")
    @Min(value = 1990, message = "Batch year must be after 1990")
    @Max(value = 2050, message = "Batch year must be before 2050")
    private Integer year;

    private String description;

    @Relationship(type = "IN_BATCH", direction = Relationship.Direction.INCOMING)
    @Builder.Default
    private Set<UserEntity> students = new HashSet<>();

    // Safe getter methods
    public Integer getYear() {
        return year != null ? year : 0;
    }

    public String getDescription() {
        return description != null ? description : "";
    }

    public Set<UserEntity> getStudents() {
        return students != null ? students : new HashSet<>();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        BatchEntity that = (BatchEntity) o;
        return year != null && year.equals(that.year);
    }

    @Override
    public int hashCode() {
        return year != null ? year.hashCode() : 0;
    }
}
