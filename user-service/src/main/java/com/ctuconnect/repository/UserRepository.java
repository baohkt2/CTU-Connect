package com.ctuconnect.repository;

import com.ctuconnect.entity.UserEntity;
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
        RETURN suggestion, COUNT(*) as mutualFriends
        ORDER BY mutualFriends DESC
        LIMIT 10
    """)
    List<UserEntity> findFriendSuggestions(@Param("userId") String userId);
}
