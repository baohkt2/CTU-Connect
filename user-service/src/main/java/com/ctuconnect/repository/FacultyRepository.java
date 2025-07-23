package com.ctuconnect.repository;

import com.ctuconnect.entity.FacultyEntity;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface FacultyRepository extends Neo4jRepository<FacultyEntity, String> {
    @Query("MATCH (f:Faculty)-[:BELONGS_TO]->(c:College {code: $collegeCode}) RETURN f")
    List<FacultyEntity> findByCollegeCode(String collegeCode);
}
