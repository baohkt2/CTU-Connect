package com.ctuconnect.repository;

import com.ctuconnect.entity.GenderEntity;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface GenderRepository extends Neo4jRepository<GenderEntity, String> {
}
