package com.ctuconnect.repository;

import com.ctuconnect.entity.UserEntity;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface UserRepository extends Neo4jRepository<UserEntity, String> {

    Optional<UserEntity> findByEmail(String email);

    Optional<UserEntity> findByUsername(String username);

    boolean existsByEmail(String email);

    boolean existsByUsername(String username);

    @Query("MATCH (u:User) WHERE u.email = $identifier OR u.username = $identifier RETURN u")
    Optional<UserEntity> findByEmailOrUsername(@Param("identifier") String identifier);

   /* @Query("""
    MATCH (u:User {id: $userId})
    OPTIONAL MATCH (u)-[:ENROLLED_IN]->(m:Major)
    OPTIONAL MATCH (u)-[:IN_BATCH]->(b:Batch)
    OPTIONAL MATCH (u)-[:HAS_GENDER]->(g:Gender)
    OPTIONAL MATCH (u)-[:WORKS_IN]->(f:Faculty)
    OPTIONAL MATCH (u)-[:BELONGS_TO]->(c:College)
    OPTIONAL MATCH (u)-[:HAS_DEGREE]->(d:Degree)
    OPTIONAL MATCH (u)-[:HAS_POSITION]->(p:Position)
    OPTIONAL MATCH (u)-[:HAS_ACADEMIC]->(a:Academic)
    OPTIONAL MATCH (u)-[:FRIEND]-(friend:User)
    RETURN u, m, b, g, f, c, d, p, a, collect(friend) as friends
""")
    Optional<UserEntity> findUserWithAllRelations(@Param("userId") String userId);*/
   // Simple query - Spring Data Neo4j will automatically load relationships
   @Query("MATCH (u:User {id: $userId}) RETURN u")
   Optional<UserEntity> findUserWithAllRelations(@Param("userId") String userId);

// In your UserRepository interface

    // In your UserRepository interface - CORRECTED QUERIES

    // Fixed: Clear all student profile relationships
    @Modifying
    @Query("MATCH (u:User {id: $userId}) " +
            "OPTIONAL MATCH (u)-[r1:ENROLLED_IN]->() DELETE r1 " +
            "WITH u " +
            "OPTIONAL MATCH (u)-[r2:WORKS_IN]->() DELETE r2 " +
            "WITH u " +
            "OPTIONAL MATCH (u)-[r3:BELONGS_TO]->() DELETE r3 " +
            "WITH u " +
            "OPTIONAL MATCH (u)-[r4:IN_BATCH]->() DELETE r4 " +
            "WITH u " +
            "OPTIONAL MATCH (u)-[r5:HAS_GENDER]->() DELETE r5")
    void clearStudentProfileRelationships(@Param("userId") String userId);

    // Fixed: Clear all lecturer profile relationships
    @Modifying
    @Query("MATCH (u:User {id: $userId}) " +
            "OPTIONAL MATCH (u)-[r1:HAS_DEGREE]->() DELETE r1 " +
            "WITH u " +
            "OPTIONAL MATCH (u)-[r2:HAS_ACADEMIC]->() DELETE r2 " +
            "WITH u " +
            "OPTIONAL MATCH (u)-[r3:HAS_POSITION]->() DELETE r3 " +
            "WITH u " +
            "OPTIONAL MATCH (u)-[r4:WORKS_IN]->() DELETE r4 " +
            "WITH u " +
            "OPTIONAL MATCH (u)-[r5:BELONGS_TO]->() DELETE r5 " +
            "WITH u " +
            "OPTIONAL MATCH (u)-[r6:HAS_GENDER]->() DELETE r6")
    void clearLecturerProfileRelationships(@Param("userId") String userId);

    // BETTER APPROACH: Single efficient query using relationship type array
    @Modifying
    @Query("MATCH (u:User {id: $userId}) " +
            "OPTIONAL MATCH (u)-[r]->() " +
            "WHERE type(r) IN ['ENROLLED_IN', 'WORKS_IN', 'BELONGS_TO', 'IN_BATCH', 'HAS_GENDER'] " +
            "DELETE r")
    void clearStudentRelationships(@Param("userId") String userId);

    @Modifying
    @Query("MATCH (u:User {id: $userId}) " +
            "OPTIONAL MATCH (u)-[r]->() " +
            "WHERE type(r) IN ['HAS_DEGREE', 'HAS_ACADEMIC', 'HAS_POSITION', 'WORKS_IN', 'BELONGS_TO', 'HAS_GENDER'] " +
            "DELETE r")
    void clearLecturerRelationships(@Param("userId") String userId);

    // MOST EFFICIENT: Generic method for any relationship types
    @Modifying
    @Query("MATCH (u:User {id: $userId}) " +
            "OPTIONAL MATCH (u)-[r]->() " +
            "WHERE type(r) IN $relationshipTypes " +
            "DELETE r")
    void clearSpecificRelationships(@Param("userId") String userId, @Param("relationshipTypes") List<String> relationshipTypes);

    // Simple individual relationship deletion (your original working method)
    @Modifying
    @Query("MATCH (u:User {id: $userId})-[r]->(n) " +
            "WHERE type(r) = $relationshipType " +
            "DELETE r")
    void deleteRelationship(@Param("userId") String userId, @Param("relationshipType") String relationshipType);
    // ========================= RELATIONSHIP UPDATE METHODS =========================

    @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:ENROLLED_IN]->(m:Major)
        DELETE r
        WITH u
        MATCH (newMajor:Major {id: $majorId})
        MERGE (u)-[:ENROLLED_IN]->(newMajor)
        """)
    void updateUserMajor(@Param("userId") String userId, @Param("majorId") String majorId);

    @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:IN_BATCH]->(b:Batch)
        DELETE r
        WITH u
        MATCH (newBatch:Batch {id: $batchId})
        MERGE (u)-[:IN_BATCH]->(newBatch)
        """)
    void updateUserBatch(@Param("userId") String userId, @Param("batchId") String batchId);

    @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:HAS_GENDER]->(g:Gender)
        DELETE r
        WITH u
        MATCH (newGender:Gender {id: $genderId})
        MERGE (u)-[:HAS_GENDER]->(newGender)
        """)
    void updateUserGender(@Param("userId") String userId, @Param("genderId") String genderId);

    @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:ENROLLED_IN]->(f:Faculty)
        DELETE r
        WITH u
        MATCH (newFaculty:Faculty {id: $facultyId})
        MERGE (u)-[:ENROLLED_IN]->(newFaculty)
        """)
    void updateUserFaculty(@Param("userId") String userId, @Param("facultyId") String facultyId);

    @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:ENROLLED_IN]->(c:College)
        DELETE r
        WITH u
        MATCH (newCollege:College {id: $collegeId})
        MERGE (u)-[:ENROLLED_IN]->(newCollege)
        """)
    void updateUserCollege(@Param("userId") String userId, @Param("collegeId") String collegeId);

    @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:WORKS_IN]->(f:Faculty)
        DELETE r
        WITH u
        MATCH (newFaculty:Faculty {id: $facultyId})
        MERGE (u)-[:WORKS_IN]->(newFaculty)
        """)
    void updateUserWorkingFaculty(@Param("userId") String userId, @Param("facultyId") String facultyId);

    @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:WORKS_IN]->(c:College)
        DELETE r
        WITH u
        MATCH (newCollege:College {id: $collegeId})
        MERGE (u)-[:WORKS_IN]->(newCollege)
        """)
    void updateUserWorkingCollege(@Param("userId") String userId, @Param("collegeId") String collegeId);

    @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:HAS_DEGREE]->(d:Degree)
        DELETE r
        WITH u
        MATCH (newDegree:Degree {id: $degreeId})
        MERGE (u)-[:HAS_DEGREE]->(newDegree)
        """)
    void updateUserDegree(@Param("userId") String userId, @Param("degreeId") String degreeId);

    @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:HAS_POSITION]->(p:Position)
        DELETE r
        WITH u
        MATCH (newPosition:Position {id: $positionId})
        MERGE (u)-[:HAS_POSITION]->(newPosition)
        """)
    void updateUserPosition(@Param("userId") String userId, @Param("positionId") String positionId);

    @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:HAS_ACADEMIC]->(a:Academic)
        DELETE r
        WITH u
        MATCH (newAcademic:Academic {id: $academicId})
        MERGE (u)-[:HAS_ACADEMIC]->(newAcademic)
        """)
    void updateUserAcademic(@Param("userId") String userId, @Param("academicId") String academicId);

    // ========================= FRIENDSHIPS =========================

    @Query("""
                MATCH (u:User {id: $userId})-[:FRIEND]-(friend:User)
                RETURN friend
            """)
    List<UserEntity> findFriends(@Param("userId") String userId);

    @Query("""
                MATCH (u1:User {id: $userId1})-[:FRIEND]-(friend:User),
                      (u2:User {id: $userId2})-[:FRIEND]-(friend)
                WHERE friend.id <> $userId1 AND friend.id <> $userId2
                RETURN DISTINCT friend
            """)
    List<UserEntity> findMutualFriends(@Param("userId1") String userId1, @Param("userId2") String userId2);

    @Query("""
                MATCH (u1:User {id: $userId1}), (u2:User {id: $userId2})
                WHERE u1.id <> u2.id
                MERGE (u1)-[:FRIEND_REQUEST {createdAt: datetime(), status: 'PENDING'}]->(u2)
            """)
    void sendFriendRequest(@Param("userId1") String userId1, @Param("userId2") String userId2);

    @Query("""
                MATCH (u1:User {id: $userId1})-[r:FRIEND_REQUEST {status: 'PENDING'}]->(u2:User {id: $userId2})
                DELETE r
                MERGE (u1)-[:FRIEND {since: datetime()}]-(u2)
            """)
    void acceptFriendRequest(@Param("userId1") String userId1, @Param("userId2") String userId2);

    @Query("""
                MATCH (u1:User {id: $userId1})-[r:FRIEND_REQUEST]->(u2:User {id: $userId2})
                DELETE r
            """)
    void rejectFriendRequest(@Param("userId1") String userId1, @Param("userId2") String userId2);

    @Query("""
                MATCH (u1:User {id: $userId1})-[r:FRIEND]-(u2:User {id: $userId2})
                DELETE r
            """)
    void deleteFriendship(@Param("userId1") String userId1, @Param("userId2") String userId2);

    @Query("""
                MATCH (u1:User {id: $userId1})-[:FRIEND]-(u2:User {id: $userId2})
                RETURN COUNT(*) > 0
            """)
    boolean areFriends(@Param("userId1") String userId1, @Param("userId2") String userId2);

    @Query("""
                MATCH (u1:User {id: $userId1})-[:FRIEND_REQUEST {status: 'PENDING'}]->(u2:User {id: $userId2})
                RETURN COUNT(*) > 0
            """)
    boolean hasPendingFriendRequest(@Param("userId1") String userId1, @Param("userId2") String userId2);

    // Get friend requests sent TO this user
    @Query("""
                MATCH (sender:User)-[:FRIEND_REQUEST {status: 'PENDING'}]->(u:User {id: $userId})
                RETURN sender
            """)
    List<UserEntity> findIncomingFriendRequests(@Param("userId") String userId);

    // Get friend requests sent BY this user
    @Query("""
                MATCH (u:User {id: $userId})-[:FRIEND_REQUEST {status: 'PENDING'}]->(receiver:User)
                RETURN receiver
            """)
    List<UserEntity> findOutgoingFriendRequests(@Param("userId") String userId);

    // ========================= FILTER QUERIES =========================

    @Query("MATCH (u:User)-[:ENROLLED_IN]->(m:Major {name: $major}) RETURN u")
    List<UserEntity> findByMajor(@Param("major") String major);

    @Query("MATCH (u:User)-[:IN_BATCH]->(b:Batch {year: $batch}) RETURN u")
    List<UserEntity> findByBatch(@Param("batch") int batch);

    @Query("""
                MATCH (u:User)-[:ENROLLED_IN]->(:Major)-[:BELONGS_TO]->(:Faculty {name: $faculty})
                RETURN u
            """)
    List<UserEntity> findByFaculty(@Param("faculty") String faculty);

    @Query("""
                MATCH (u:User)-[:ENROLLED_IN]->(:Major)-[:BELONGS_TO]->(:Faculty)-[:PART_OF]->(:College {name: $college})
                RETURN u
            """)
    List<UserEntity> findByCollege(@Param("college") String college);

    // ========================= SAME ATTRIBUTE QUERIES =========================

    @Query("""
                MATCH (u:User {id: $userId})-[:ENROLLED_IN]->(m:Major)<-[:ENROLLED_IN]-(other:User)
                WHERE u.id <> other.id
                RETURN other
            """)
    List<UserEntity> findUsersWithSameMajor(@Param("userId") String userId);

    @Query("""
                MATCH (u:User {id: $userId})-[:IN_BATCH]->(b:Batch)<-[:IN_BATCH]-(other:User)
                WHERE u.id <> other.id
                RETURN other
            """)
    List<UserEntity> findUsersWithSameBatch(@Param("userId") String userId);

    @Query("""
                MATCH (u:User {id: $userId})-[:ENROLLED_IN]->(:Major)-[:BELONGS_TO]->(f:Faculty)
                MATCH (other:User)-[:ENROLLED_IN]->(:Major)-[:BELONGS_TO]->(f)
                WHERE other.id <> $userId
                RETURN other
            """)
    List<UserEntity> findUsersWithSameFaculty(@Param("userId") String userId);

    @Query("""
                MATCH (u:User {id: $userId})-[:ENROLLED_IN]->(:Major)-[:BELONGS_TO]->(:Faculty)-[:PART_OF]->(c:College)
                MATCH (other:User)-[:ENROLLED_IN]->(:Major)-[:BELONGS_TO]->(:Faculty)-[:PART_OF]->(c)
                WHERE other.id <> $userId
                RETURN other
            """)
    List<UserEntity> findUsersWithSameCollege(@Param("userId") String userId);

    // ========================= ADVANCED FILTERING =========================

    @Query("""
                MATCH (u:User {id: $userId})
                OPTIONAL MATCH (u)-[:ENROLLED_IN]->(uMajor:Major)
                OPTIONAL MATCH (u)-[:IN_BATCH]->(uBatch:Batch)
                OPTIONAL MATCH (uMajor)-[:BELONGS_TO]->(uFaculty:Faculty)
                OPTIONAL MATCH (uFaculty)-[:PART_OF]->(uCollege:College)
            
                MATCH (other:User)
                OPTIONAL MATCH (other)-[:ENROLLED_IN]->(oMajor:Major)
                OPTIONAL MATCH (other)-[:IN_BATCH]->(oBatch:Batch)
                OPTIONAL MATCH (oMajor)-[:BELONGS_TO]->(oFaculty:Faculty)
                OPTIONAL MATCH (oFaculty)-[:PART_OF]->(oCollege:College)
            
                WHERE u.id <> other.id
                  AND ($isSameMajor = false OR uMajor.name = oMajor.name)
                  AND ($isSameBatch = false OR uBatch.year = oBatch.year)
                  AND ($isSameFaculty = false OR uFaculty.name = oFaculty.name)
                  AND ($isSameCollege = false OR uCollege.name = oCollege.name)
            
                RETURN DISTINCT other
            """)
    List<UserEntity> findUsersWithFilters(
            @Param("userId") String userId,
            @Param("isSameCollege") boolean isSameCollege,
            @Param("isSameFaculty") boolean isSameFaculty,
            @Param("isSameMajor") boolean isSameMajor,
            @Param("isSameBatch") boolean isSameBatch
    );

    @Query("""
                MATCH (u:User {id: $userId})-[:FRIEND]-(friend:User)
                OPTIONAL MATCH (u)-[:ENROLLED_IN]->(uMajor:Major)
                OPTIONAL MATCH (friend)-[:ENROLLED_IN]->(fMajor:Major)
                OPTIONAL MATCH (uMajor)-[:BELONGS_TO]->(uFaculty:Faculty)
                OPTIONAL MATCH (fMajor)-[:BELONGS_TO]->(fFaculty:Faculty)
                OPTIONAL MATCH (uFaculty)-[:PART_OF]->(uCollege:College)
                OPTIONAL MATCH (fFaculty)-[:PART_OF]->(fCollege:College)
                OPTIONAL MATCH (u)-[:IN_BATCH]->(uBatch:Batch)
                OPTIONAL MATCH (friend)-[:IN_BATCH]->(fBatch:Batch)
            
                WHERE ($isSameMajor = false OR uMajor.name = fMajor.name)
                  AND ($isSameFaculty = false OR uFaculty.name = fFaculty.name)
                  AND ($isSameCollege = false OR uCollege.name = fCollege.name)
                  AND ($isSameBatch = false OR uBatch.year = fBatch.year)
            
                RETURN DISTINCT friend
            """)
    List<UserEntity> findFriendsWithFilters(
            @Param("userId") String userId,
            @Param("isSameCollege") boolean isSameCollege,
            @Param("isSameFaculty") boolean isSameFaculty,
            @Param("isSameMajor") boolean isSameMajor,
            @Param("isSameBatch") boolean isSameBatch
    );

    // ========================= FRIEND SUGGESTIONS =========================

    @Query("""
                MATCH (u:User {id: $userId})-[:FRIEND]-(friend:User)-[:FRIEND]-(suggestion:User)
                WHERE u.id <> suggestion.id 
                  AND NOT (u)-[:FRIEND]-(suggestion)
                  AND NOT (u)-[:FRIEND_REQUEST]-(suggestion)
                RETURN suggestion, COUNT(*) AS mutualFriends
                ORDER BY mutualFriends DESC
                LIMIT 10
            """)
    List<UserEntity> findFriendSuggestions(@Param("userId") String userId);
}

