package com.ctuconnect.repository;

import com.ctuconnect.entity.UniversityEntity;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;

import java.util.List;
import java.util.Optional;

public interface UniversityRepository extends Neo4jRepository<UniversityEntity, String> {

    Optional<UniversityEntity> findByName(String name);

    Optional<UniversityEntity> findByCode(String code);

    boolean existsByName(String name);

    boolean existsByCode(String code);

    @Query("""
        MATCH (u:University)
        OPTIONAL MATCH (u)-[:HAS_COLLEGE]->(c:College)
        RETURN u, collect(c) as colleges
        ORDER BY u.name ASC
        """)
    List<UniversityEntity> findAllWithRelations();
}
