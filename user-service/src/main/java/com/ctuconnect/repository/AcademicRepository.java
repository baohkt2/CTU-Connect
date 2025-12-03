package com.ctuconnect.repository;


import com.ctuconnect.entity.AcademicEntity;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;


@Repository
public interface AcademicRepository extends Neo4jRepository<AcademicEntity, String> {
    Optional<AcademicEntity> findByName(String name);
    Optional<AcademicEntity> findByCode(String code);
}
