package com.ctuconnect.repository;

import com.ctuconnect.entity.FacultyEntity;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface FacultyRepository extends Neo4jRepository<FacultyEntity, String> {

    @Query("MATCH (f:Faculty) RETURN f ORDER BY f.name")
    List<FacultyEntity> findAllWithCollegeAndMajors();

    @Query("MATCH (f:Faculty {college: $collegeName}) RETURN f ORDER BY f.name")
    List<FacultyEntity> findByCollegeWithMajors(String collegeName);

    @Query("MATCH (f:Faculty {name: $name}) RETURN f")
    Optional<FacultyEntity> findByNameWithCollegeAndMajors(String name);

    Optional<FacultyEntity> findByName(String name);

    Optional<FacultyEntity> findByCode(String code);

    List<FacultyEntity> findByCollege(String college);


}
