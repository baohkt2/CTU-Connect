package com.ctuconnect.repository;

import com.ctuconnect.entity.MajorEntity;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface MajorRepository extends Neo4jRepository<MajorEntity, String> {

    @Query("MATCH (m:Major) RETURN m ORDER BY m.name")
    List<MajorEntity> findAllWithFacultyAndCollege();

    @Query("MATCH (m:Major {faculty: $facultyName}) RETURN m ORDER BY m.name")
    List<MajorEntity> findByFacultyWithFacultyAndCollege(String facultyName);

    @Query("MATCH (m:Major {name: $name}) RETURN m")
    Optional<MajorEntity> findByNameWithFacultyAndCollege(String name);

    @Query("MATCH (m:Major {code: $code}) RETURN m")
    Optional<MajorEntity> findByCode(String code);

    Optional<MajorEntity> findByName(String name);

    List<MajorEntity> findByFaculty(String faculty);


}
