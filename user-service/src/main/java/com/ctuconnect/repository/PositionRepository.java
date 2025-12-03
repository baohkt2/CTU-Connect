package com.ctuconnect.repository;

import com.ctuconnect.entity.BatchEntity;
import com.ctuconnect.entity.PositionEntity;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface PositionRepository extends Neo4jRepository<PositionEntity, String> {
    Optional<PositionEntity> findByName(String name);

    Optional<PositionEntity> findByCode(String code);
}
