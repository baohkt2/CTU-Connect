package com.ctuconnect.repository;

import com.ctuconnect.entity.MajorEntity;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface MajorRepository extends Neo4jRepository<MajorEntity, String> {
    // Tìm major theo faculty name
    @Query("MATCH (f:Faculty {name: $facultyName})-[:HAS_MAJOR]->(m:Major) RETURN m")
    List<MajorEntity> findByFacultyName(String facultyName);

    // Tìm major theo faculty property trong major node
    List<MajorEntity> findByFaculty(String faculty);
}
