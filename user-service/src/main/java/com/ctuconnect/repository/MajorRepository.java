package com.ctuconnect.repository;

import com.ctuconnect.entity.MajorEntity;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.stereotype.Repository;
import java.util.List;

@Repository
public interface MajorRepository extends Neo4jRepository<MajorEntity, String> {
    @Query("MATCH (m:Major)-[:BELONGS_TO]->(f:Faculty {code: $facultyCode}) RETURN m")
    List<MajorEntity> findByFacultyCode(String facultyCode);
}
