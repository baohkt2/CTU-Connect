package com.ctuconnect.repository;

import com.ctuconnect.entity.CollegeEntity;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface CollegeRepository extends Neo4jRepository<CollegeEntity, String> {

    @Query("MATCH (c:College) RETURN c ORDER BY c.name")
    List<CollegeEntity> findAllWithFacultiesAndMajors();

    @Query("MATCH (c:College {name: $name}) RETURN c")
    Optional<CollegeEntity> findByNameWithFacultiesAndMajors(String name);

    Optional<CollegeEntity> findByName(String name);

    Optional<CollegeEntity> findByCode(String code);
}
