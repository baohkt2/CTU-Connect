package com.ctuconnect.repository;

import com.ctuconnect.entity.UserEntity;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;

import java.util.List;
import java.util.Optional;

public interface UserRepository extends Neo4jRepository<UserEntity, String> {

    // Basic user queries
    Optional<UserEntity> findByEmail(String email);

    boolean existsByEmail(String email);

    Optional<UserEntity> findByStudentId(String studentId);

    boolean existsByStudentId(String studentId);

    List<UserEntity> findByIsActiveTrue();

    // Simple user profile query - fetch user with all direct relationships
    // Explicitly return all properties to ensure avatarUrl is loaded
    @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[:ENROLLED_IN]->(m:Major)
        OPTIONAL MATCH (u)-[:IN_BATCH]->(b:Batch)
        OPTIONAL MATCH (u)-[:HAS_GENDER]->(g:Gender)
        RETURN u, m, b, g
        """)
    Optional<UserEntity> findUserWithRelationships(@Param("userId") String userId);

    // Enhanced user search with relationship context
    @Query(value = """
    MATCH (u:User)
    WHERE u.isActive = true
    AND (
        toLower(u.fullName) CONTAINS toLower($searchTerm) OR
        toLower(u.username) CONTAINS toLower($searchTerm) OR
        toLower(u.email) CONTAINS toLower($searchTerm) OR
        toLower(u.studentId) CONTAINS toLower($searchTerm)
    )
    AND ($currentUserId IS NULL OR u.id <> $currentUserId)
    OPTIONAL MATCH (u)-[:ENROLLED_IN]->(m:Major)
    OPTIONAL MATCH (m)-[:HAS_MAJOR]-(f:Faculty)
    OPTIONAL MATCH (f)-[:HAS_FACULTY]-(c:College)
    OPTIONAL MATCH (u)-[:IN_BATCH]->(b:Batch)
    OPTIONAL MATCH (u)-[:HAS_GENDER]->(g:Gender)
    OPTIONAL MATCH (u)-[:IS_FRIENDS_WITH]-(friend:User)
    OPTIONAL MATCH (currentUser:User {id: $currentUserId})
    OPTIONAL MATCH (currentUser)-[:IS_FRIENDS_WITH]-(u)
    OPTIONAL MATCH (currentUser)-[:SENT_FRIEND_REQUEST_TO]->(u)
    OPTIONAL MATCH (u)-[:SENT_FRIEND_REQUEST_TO]->(currentUser)
    OPTIONAL MATCH (currentUser)-[:IS_FRIENDS_WITH]-(mutual:User)-[:IS_FRIENDS_WITH]-(u)
    WHERE currentUser.id <> u.id
    RETURN u as user,
           c.name as college,
           f.name as faculty,
           m.name as major,
           b.year as batch,
           g.name as gender,
           count(DISTINCT friend) as friendsCount,
           count(DISTINCT mutual) as mutualFriendsCount,
           CASE WHEN currentUser IS NULL THEN false ELSE EXISTS((currentUser)-[:IS_FRIENDS_WITH]-(u)) END as isFriend,
           CASE WHEN currentUser IS NULL THEN false ELSE EXISTS((currentUser)-[:SENT_FRIEND_REQUEST_TO]->(u)) END as requestSent,
           CASE WHEN currentUser IS NULL THEN false ELSE EXISTS((u)-[:SENT_FRIEND_REQUEST_TO]->(currentUser)) END as requestReceived
    ORDER BY 
        CASE WHEN currentUser IS NULL THEN 0 ELSE count(DISTINCT mutual) END DESC,
        u.fullName ASC
""",
            countQuery = """
    MATCH (u:User)
    WHERE u.isActive = true
    AND (
        toLower(u.fullName) CONTAINS toLower($searchTerm) OR
        toLower(u.username) CONTAINS toLower($searchTerm) OR
        toLower(u.email) CONTAINS toLower($searchTerm) OR
        toLower(u.studentId) CONTAINS toLower($searchTerm)
    )
    AND ($currentUserId IS NULL OR u.id <> $currentUserId)
    RETURN count(u)
""")
    Page<UserSearchProjection> searchUsers(@Param("searchTerm") String searchTerm,
                                           @Param("currentUserId") String currentUserId,
                                           Pageable pageable);

    // Find friends of a user
    @Query(value = """
    MATCH (u:User {id: $userId})-[:IS_FRIENDS_WITH]-(friend:User)
    WHERE friend.isActive = true
    OPTIONAL MATCH (friend)-[:ENROLLED_IN]->(m:Major)
    OPTIONAL MATCH (m)-[:HAS_MAJOR]-(f:Faculty)
    OPTIONAL MATCH (f)-[:HAS_FACULTY]-(c:College)
    OPTIONAL MATCH (friend)-[:IN_BATCH]->(b:Batch)
    OPTIONAL MATCH (friend)-[:HAS_GENDER]->(g:Gender)
    OPTIONAL MATCH (friend)-[:IS_FRIENDS_WITH]-(friendOfFriend:User)
    RETURN friend as user,
           c.name as college,
           f.name as faculty,
           m.name as major,
           b.year as batch,
           g.name as gender,
           count(DISTINCT friendOfFriend) as friendsCount
    ORDER BY friend.fullName ASC
""",
            countQuery = """
    MATCH (u:User {id: $userId})-[:IS_FRIENDS_WITH]-(friend:User)
    WHERE friend.isActive = true
    RETURN count(friend)
""")
    Page<UserSearchProjection> findFriends(@Param("userId") String userId, Pageable pageable);

    // Find sent friend requests
    @Query("""
        MATCH (u:User {id: $userId})-[:SENT_FRIEND_REQUEST_TO]->(target:User)
        WHERE target.isActive = true
        
        OPTIONAL MATCH (target)-[:ENROLLED_IN]->(m:Major)
        OPTIONAL MATCH (m)-[:HAS_MAJOR]-(f:Faculty)
        OPTIONAL MATCH (f)-[:HAS_FACULTY]-(c:College)
        OPTIONAL MATCH (target)-[:IN_BATCH]->(b:Batch)
        OPTIONAL MATCH (target)-[:HAS_GENDER]->(g:Gender)
        OPTIONAL MATCH (u)-[:IS_FRIENDS_WITH]-(mutual:User)-[:IS_FRIENDS_WITH]-(target)
        
        RETURN target as user,
               c.name as college,
               f.name as faculty,
               m.name as major,
               b.year as batch,
               g.name as gender,
               count(DISTINCT mutual) as mutualFriendsCount,
               'SENT' as requestType
        ORDER BY target.fullName ASC
        """)
    List<FriendRequestProjection> findSentFriendRequests(@Param("userId") String userId);

    // Find received friend requests
    @Query("""
        MATCH (requester:User)-[:SENT_FRIEND_REQUEST_TO]->(u:User {id: $userId})
        WHERE requester.isActive = true
        
        OPTIONAL MATCH (requester)-[:ENROLLED_IN]->(m:Major)
        OPTIONAL MATCH (m)-[:HAS_MAJOR]-(f:Faculty)
        OPTIONAL MATCH (f)-[:HAS_FACULTY]-(c:College)
        OPTIONAL MATCH (requester)-[:IN_BATCH]->(b:Batch)
        OPTIONAL MATCH (requester)-[:HAS_GENDER]->(g:Gender)
        OPTIONAL MATCH (u)-[:IS_FRIENDS_WITH]-(mutual:User)-[:IS_FRIENDS_WITH]-(requester)
        
        RETURN requester as user,
               c.name as college,
               f.name as faculty,
               m.name as major,
               b.year as batch,
               g.name as gender,
               count(DISTINCT mutual) as mutualFriendsCount,
               'RECEIVED' as requestType
        ORDER BY requester.fullName ASC
        """)
    List<FriendRequestProjection> findReceivedFriendRequests(@Param("userId") String userId);

    // Send friend request
    @Query("""
        MATCH (sender:User {id: $senderId}), (receiver:User {id: $receiverId})
        WHERE sender.isActive = true AND receiver.isActive = true
        AND NOT (sender)-[:IS_FRIENDS_WITH]-(receiver)
        AND NOT (sender)-[:SENT_FRIEND_REQUEST_TO]->(receiver)
        AND NOT (receiver)-[:SENT_FRIEND_REQUEST_TO]->(sender)
        CREATE (sender)-[:SENT_FRIEND_REQUEST_TO]->(receiver)
        RETURN count(*) > 0 as success
        """)
    boolean sendFriendRequest(@Param("senderId") String senderId, @Param("receiverId") String receiverId);

    // Accept friend request
    @Query("""
        MATCH (requester:User {id: $requesterId})-[r:SENT_FRIEND_REQUEST_TO]->(accepter:User {id: $accepterId})
        WHERE requester.isActive = true AND accepter.isActive = true
        DELETE r
        CREATE (requester)-[:IS_FRIENDS_WITH]-(accepter)
        RETURN count(*) > 0 as success
        """)
    boolean acceptFriendRequest(@Param("requesterId") String requesterId, @Param("accepterId") String accepterId);

    // Reject friend request
    @Query("""
        MATCH (requester:User {id: $requesterId})-[r:SENT_FRIEND_REQUEST_TO]->(rejecter:User {id: $rejecterId})
        DELETE r
        RETURN count(*) > 0 as success
        """)
    boolean rejectFriendRequest(@Param("requesterId") String requesterId, @Param("rejecterId") String rejecterId);

    // Remove friend
    @Query("""
        MATCH (user1:User {id: $userId1})-[r:IS_FRIENDS_WITH]-(user2:User {id: $userId2})
        DELETE r
        RETURN count(*) > 0 as success
        """)
    boolean removeFriend(@Param("userId1") String userId1, @Param("userId2") String userId2);

    // Find users by college
    @Query(value = """
    MATCH (u:User)-[:ENROLLED_IN]->(m:Major)-[:HAS_MAJOR]-(f:Faculty)-[:HAS_FACULTY]-(c:College {name: $collegeName})
    WHERE u.isActive = true AND ($currentUserId IS NULL OR u.id <> $currentUserId)
    OPTIONAL MATCH (u)-[:IN_BATCH]->(b:Batch)
    OPTIONAL MATCH (u)-[:HAS_GENDER]->(g:Gender)
    OPTIONAL MATCH (u)-[:IS_FRIENDS_WITH]-(friend:User)
    RETURN u as user,
           c.name as college,
           f.name as faculty,
           m.name as major,
           b.year as batch,
           g.name as gender,
           count(DISTINCT friend) as friendsCount
    ORDER BY u.fullName ASC
""",
            countQuery = """
    MATCH (u:User)-[:ENROLLED_IN]->(m:Major)-[:HAS_MAJOR]-(f:Faculty)-[:HAS_FACULTY]-(c:College {name: $collegeName})
    WHERE u.isActive = true AND ($currentUserId IS NULL OR u.id <> $currentUserId)
    RETURN count(u)
""")
    Page<UserSearchProjection> findUsersByCollege(@Param("collegeName") String collegeName,
                                                  @Param("currentUserId") String currentUserId,
                                                  Pageable pageable);

    // Find users by faculty
    @Query(value = """
    MATCH (u:User)-[:ENROLLED_IN]->(m:Major)-[:HAS_MAJOR]-(f:Faculty {name: $facultyName})
    WHERE u.isActive = true AND ($currentUserId IS NULL OR u.id <> $currentUserId)
    OPTIONAL MATCH (f)-[:HAS_FACULTY]-(c:College)
    OPTIONAL MATCH (u)-[:IN_BATCH]->(b:Batch)
    OPTIONAL MATCH (u)-[:HAS_GENDER]->(g:Gender)
    OPTIONAL MATCH (u)-[:IS_FRIENDS_WITH]-(friend:User)
    RETURN u as user,
           c.name as college,
           f.name as faculty,
           m.name as major,
           b.year as batch,
           g.name as gender,
           count(DISTINCT friend) as friendsCount
    ORDER BY u.fullName ASC
""",
            countQuery = """
    MATCH (u:User)-[:ENROLLED_IN]->(m:Major)-[:HAS_MAJOR]-(f:Faculty {name: $facultyName})
    WHERE u.isActive = true AND ($currentUserId IS NULL OR u.id <> $currentUserId)
    RETURN count(u)
""")
    Page<UserSearchProjection> findUsersByFaculty(@Param("facultyName") String facultyName,
                                                  @Param("currentUserId") String currentUserId,
                                                  Pageable pageable);

    // Find users by major
    @Query(value = """
    MATCH (u:User)-[:ENROLLED_IN]->(m:Major {name: $majorName})
    WHERE u.isActive = true AND ($currentUserId IS NULL OR u.id <> $currentUserId)
    OPTIONAL MATCH (m)-[:HAS_MAJOR]-(f:Faculty)
    OPTIONAL MATCH (f)-[:HAS_FACULTY]-(c:College)
    OPTIONAL MATCH (u)-[:IN_BATCH]->(b:Batch)
    OPTIONAL MATCH (u)-[:HAS_GENDER]->(g:Gender)
    OPTIONAL MATCH (u)-[:IS_FRIENDS_WITH]-(friend:User)
    RETURN u as user,
           c.name as college,
           f.name as faculty,
           m.name as major,
           b.year as batch,
           g.name as gender,
           count(DISTINCT friend) as friendsCount
    ORDER BY u.fullName ASC
""",
            countQuery = """
    MATCH (u:User)-[:ENROLLED_IN]->(m:Major {name: $majorName})
    WHERE u.isActive = true AND ($currentUserId IS NULL OR u.id <> $currentUserId)
    RETURN count(u)
""")
    Page<UserSearchProjection> findUsersByMajor(@Param("majorName") String majorName,
                                                @Param("currentUserId") String currentUserId,
                                                Pageable pageable);

    // Find users by batch
    @Query(value = """
    MATCH (u:User)-[:IN_BATCH]->(b:Batch {year: $batchYear})
    WHERE u.isActive = true AND ($currentUserId IS NULL OR u.id <> $currentUserId)
    OPTIONAL MATCH (u)-[:ENROLLED_IN]->(m:Major)
    OPTIONAL MATCH (m)-[:HAS_MAJOR]-(f:Faculty)
    OPTIONAL MATCH (f)-[:HAS_FACULTY]-(c:College)
    OPTIONAL MATCH (u)-[:HAS_GENDER]->(g:Gender)
    OPTIONAL MATCH (u)-[:IS_FRIENDS_WITH]-(friend:User)
    RETURN u as user,
           c.name as college,
           f.name as faculty,
           m.name as major,
           b.year as batch,
           g.name as gender,
           count(DISTINCT friend) as friendsCount
    ORDER BY u.fullName ASC
""",
            countQuery = """
    MATCH (u:User)-[:IN_BATCH]->(b:Batch {year: $batchYear})
    WHERE u.isActive = true AND ($currentUserId IS NULL OR u.id <> $currentUserId)
    RETURN count(u)
""")
    Page<UserSearchProjection> findUsersByBatch(@Param("batchYear") Integer batchYear,
                                                @Param("currentUserId") String currentUserId,
                                                Pageable pageable);

    // Friend suggestion query based on mutual friends
    @Query("""
        MATCH (u:User {id: $userId})-[:IS_FRIENDS_WITH]-(friend:User)-[:IS_FRIENDS_WITH]-(suggestion:User)
        WHERE suggestion.isActive = true 
        AND suggestion.id <> $userId
        AND NOT (u)-[:IS_FRIENDS_WITH]-(suggestion)
        AND NOT (u)-[:SENT_FRIEND_REQUEST_TO]->(suggestion)
        AND NOT (suggestion)-[:SENT_FRIEND_REQUEST_TO]->(u)
        RETURN DISTINCT suggestion
        ORDER BY suggestion.fullName ASC
        LIMIT 50
        """)
    List<UserEntity> findFriendSuggestions(@Param("userId") String userId);

    // Check if two users are friends
    @Query("""
        MATCH (u1:User {id: $userId1})-[:IS_FRIENDS_WITH]-(u2:User {id: $userId2})
        RETURN count(*) > 0
        """)
    boolean areFriends(@Param("userId1") String userId1, @Param("userId2") String userId2);

    // Find mutual friends between two users
    @Query("""
        MATCH (u1:User {id: $userId1})-[:IS_FRIENDS_WITH]-(mutual:User)-[:IS_FRIENDS_WITH]-(u2:User {id: $userId2})
        WHERE mutual.isActive = true
        RETURN DISTINCT mutual
        ORDER BY mutual.fullName ASC
        """)
    List<UserEntity> findMutualFriends(@Param("userId1") String userId1, @Param("userId2") String userId2);

    // Check if there's a pending friend request between two users
    @Query("""
        MATCH (u1:User {id: $userId1})-[:SENT_FRIEND_REQUEST_TO]->(u2:User {id: $userId2})
        RETURN count(*) > 0
        """)
    boolean hasPendingFriendRequest(@Param("userId1") String userId1, @Param("userId2") String userId2);

    // Find users by faculty ID (name)
    @Query("""
        MATCH (u:User)-[:ENROLLED_IN]->(m:Major)-[:HAS_MAJOR]-(f:Faculty {name: $facultyId})
        WHERE u.isActive = true
        RETURN DISTINCT u
        ORDER BY u.fullName ASC
        """)
    List<UserEntity> findUsersByFacultyId(@Param("facultyId") String facultyId);

    // Find users by major ID (name)
    @Query("""
        MATCH (u:User)-[:ENROLLED_IN]->(m:Major {name: $majorId})
        WHERE u.isActive = true
        RETURN DISTINCT u
        ORDER BY u.fullName ASC
        """)
    List<UserEntity> findUsersByMajorId(@Param("majorId") String majorId);

    // Update relationship methods - properly handle Neo4j relationship updates
    
    @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:ENROLLED_IN]->(:Major)
        DELETE r
        WITH u
        MATCH (m:Major {code: $majorCode})
        MERGE (u)-[:ENROLLED_IN]->(m)
        """)
    void updateUserMajor(@Param("userId") String userId, @Param("majorCode") String majorCode);
    
    @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:IN_BATCH]->(:Batch)
        DELETE r
        WITH u
        MATCH (b:Batch {year: $batchYear})
        MERGE (u)-[:IN_BATCH]->(b)
        """)
    void updateUserBatch(@Param("userId") String userId, @Param("batchYear") String batchYear);
    
    @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:HAS_GENDER]->(:Gender)
        DELETE r
        WITH u
        MATCH (g:Gender {name: $genderName})
        MERGE (u)-[:HAS_GENDER]->(g)
        """)
    void updateUserGender(@Param("userId") String userId, @Param("genderName") String genderName);

    // Interface for projections
    interface UserProfileProjection {
        String getUserId();
        String getEmail();
        String getUsername();
        String getStudentId();
        String getFullName();
        String getBio();
        String getRole();
        Boolean getIsActive();
        java.time.LocalDateTime getCreatedAt();
        java.time.LocalDateTime getUpdatedAt();
        String getCollege();
        String getFaculty();
        String getMajor();
        String getBatch();
        String getGender();
        Long getFriendsCount();
        Long getSentRequestsCount();
        Long getReceivedRequestsCount();
    }

    interface UserSearchProjection {
        UserEntity getUser();
        String getCollege();
        String getFaculty();
        String getMajor();
        String getBatch();
        String getGender();
        Long getFriendsCount();
        Long getMutualFriendsCount();
        Boolean getIsFriend();
        Boolean getRequestSent();
        Boolean getRequestReceived();
    }

    interface FriendRequestProjection {
        UserEntity getUser();
        String getCollege();
        String getFaculty();
        String getMajor();
        Integer getBatch();
        String getGender();
        Long getMutualFriendsCount();
        String getRequestType();
    }
}

