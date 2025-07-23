package com.ctuconnect.repository;

import com.ctuconnect.entity.FacultyEntity;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface FacultyRepository extends Neo4jRepository<FacultyEntity, String> {
    // Tìm faculty theo college name
    @Query("MATCH (c:College {name: $collegeName})-[:HAS_FACULTY]->(f:Faculty) RETURN f")
    List<FacultyEntity> findByCollegeName(String collegeName);

    // Tìm faculty theo college property trong faculty node
    List<FacultyEntity> findByCollege(String college);
}
