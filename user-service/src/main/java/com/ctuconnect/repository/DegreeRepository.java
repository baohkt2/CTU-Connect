package com.ctuconnect.repository;


import com.ctuconnect.entity.BatchEntity;
import com.ctuconnect.entity.DegreeEntity;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface DegreeRepository extends Neo4jRepository<DegreeEntity, String> {
    Optional<DegreeEntity> findByName(String name);
    Optional<DegreeEntity> findByCode(String code);
}
