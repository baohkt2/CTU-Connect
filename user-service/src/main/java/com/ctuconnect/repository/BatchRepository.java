package com.ctuconnect.repository;

import com.ctuconnect.entity.BatchEntity;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface BatchRepository extends Neo4jRepository<BatchEntity, Integer> {

    List<BatchEntity> findAllByOrderByYearDesc();

    Optional<BatchEntity> findByYear(Integer year);
}
