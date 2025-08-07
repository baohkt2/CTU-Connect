package com.ctuconnect.repository;

import com.ctuconnect.entity.GenderEntity;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface GenderRepository extends Neo4jRepository<GenderEntity, String> {

    Optional<GenderEntity> findByCode(String code);

    Optional<GenderEntity> findByName(String name);
}
