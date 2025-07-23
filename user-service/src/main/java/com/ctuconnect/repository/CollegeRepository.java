package com.ctuconnect.repository;

import com.ctuconnect.entity.CollegeEntity;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface CollegeRepository extends Neo4jRepository<CollegeEntity, String> {
}
