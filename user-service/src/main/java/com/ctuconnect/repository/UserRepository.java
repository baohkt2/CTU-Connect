package com.ctuconnect.repository;

import com.ctuconnect.entity.FacultyEntity;
import com.ctuconnect.entity.UserEntity;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface UserRepository extends Neo4jRepository<UserEntity, String> {

 // Tìm kiếm người dùng bằng email hoặc username
 Optional<UserEntity> findByEmail(String email);

 Optional<UserEntity> findByUsername(String username);

 boolean existsByEmail(String email);

 boolean existsByUsername(String username);

 @Query("MATCH (u:User) WHERE u.email = $identifier OR u.username = $identifier RETURN u")
 Optional<UserEntity> findByEmailOrUsername(@Param("identifier") String identifier);

 // Lấy user by id
 @Query("MATCH (u:User {id: $userId}) RETURN u")
 Optional<UserEntity> findUserWithAllRelations(@Param("userId") String userId);

 // Xóa quan hệ của profile học sinh (student)
 @Modifying
 @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r]->()
        WHERE type(r) IN ['ENROLLED_IN', 'WORKS_IN', 'BELONGS_TO', 'IN_BATCH', 'HAS_GENDER']
        DELETE r
        """)
 void clearStudentRelationships(@Param("userId") String userId);

 // Xóa quan hệ của profile giảng viên (lecturer)
 @Modifying
 @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r]->()
        WHERE type(r) IN ['HAS_DEGREE', 'HAS_ACADEMIC', 'HAS_POSITION', 'WORKS_IN', 'BELONGS_TO', 'HAS_GENDER']
        DELETE r
        """)
 void clearLecturerRelationships(@Param("userId") String userId);

 // Xóa quan hệ cụ thể theo danh sách quan hệ truyền vào
 @Modifying
 @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r]->()
        WHERE type(r) IN $relationshipTypes
        DELETE r
        """)
 void clearSpecificRelationships(@Param("userId") String userId, @Param("relationshipTypes") List<String> relationshipTypes);

 // Xóa quan hệ đơn lẻ theo loại
 @Modifying
 @Query("""
        MATCH (u:User {id: $userId})-[r]->(n)
        WHERE type(r) = $relationshipType
        DELETE r
        """)
 void deleteRelationship(@Param("userId") String userId, @Param("relationshipType") String relationshipType);

 // Cập nhật các quan hệ profile
 @Modifying
 @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:ENROLLED_IN]->()
        DELETE r
        WITH u
        MATCH (newMajor:Major {id: $majorId})
        MERGE (u)-[:ENROLLED_IN]->(newMajor)
        """)
 void updateUserMajor(@Param("userId") String userId, @Param("majorId") String majorId);

 @Modifying
 @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:IN_BATCH]->()
        DELETE r
        WITH u
        MATCH (newBatch:Batch {id: $batchId})
        MERGE (u)-[:IN_BATCH]->(newBatch)
        """)
 void updateUserBatch(@Param("userId") String userId, @Param("batchId") String batchId);

 @Modifying
 @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:HAS_GENDER]->()
        DELETE r
        WITH u
        MATCH (newGender:Gender {id: $genderId})
        MERGE (u)-[:HAS_GENDER]->(newGender)
        """)
 void updateUserGender(@Param("userId") String userId, @Param("genderId") String genderId);

 @Modifying
 @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:BELONGS_TO]->()
        DELETE r
        WITH u
        MATCH (newFaculty:Faculty {id: $facultyId})
        MERGE (u)-[:BELONGS_TO]->(newFaculty)
        """)
 void updateUserFaculty(@Param("userId") String userId, @Param("facultyId") String facultyId);

 @Modifying
 @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:STUDIES_AT]->()
        DELETE r
        WITH u
        MATCH (newCollege:College {id: $collegeId})
        MERGE (u)-[:STUDIES_AT]->(newCollege)
        """)
 void updateUserCollege(@Param("userId") String userId, @Param("collegeId") String collegeId);

 @Modifying
 @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:WORKS_IN]->()
        DELETE r
        WITH u
        MATCH (newFaculty:Faculty {id: $facultyId})
        MERGE (u)-[:WORKS_IN]->(newFaculty)
        """)
 void updateUserWorkingFaculty(@Param("userId") String userId, @Param("facultyId") String facultyId);

 @Modifying
 @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:EMPLOYED_AT]->()
        DELETE r
        WITH u
        MATCH (newCollege:College {id: $collegeId})
        MERGE (u)-[:EMPLOYED_AT]->(newCollege)
        """)
 void updateUserWorkingCollege(@Param("userId") String userId, @Param("collegeId") String collegeId);

 @Modifying
 @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:HAS_DEGREE]->()
        DELETE r
        WITH u
        MATCH (newDegree:Degree {id: $degreeId})
        MERGE (u)-[:HAS_DEGREE]->(newDegree)
        """)
 void updateUserDegree(@Param("userId") String userId, @Param("degreeId") String degreeId);

 @Modifying
 @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:HAS_POSITION]->()
        DELETE r
        WITH u
        MATCH (newPosition:Position {id: $positionId})
        MERGE (u)-[:HAS_POSITION]->(newPosition)
        """)
 void updateUserPosition(@Param("userId") String userId, @Param("positionId") String positionId);

 @Modifying
 @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[r:HAS_ACADEMIC]->()
        DELETE r
        WITH u
        MATCH (newAcademic:Academic {id: $academicId})
        MERGE (u)-[:HAS_ACADEMIC]->(newAcademic)
        """)
 void updateUserAcademic(@Param("userId") String userId, @Param("academicId") String academicId);

 // ========================= FRIEND RELATIONSHIP QUERIES =========================

 // Kiểm tra xem 2 user có phải bạn bè không
 @Query("""
        MATCH (u1:User {id: $userId1})-[:FRIEND_WITH]-(u2:User {id: $userId2})
        RETURN COUNT(*) > 0
        """)
 boolean areFriends(@Param("userId1") String userId1, @Param("userId2") String userId2);

 // Kiểm tra có friend request pending không
 @Query("""
        MATCH (u1:User {id: $fromUserId})-[:FRIEND_REQUEST]->(u2:User {id: $toUserId})
        RETURN COUNT(*) > 0
        """)
 boolean hasPendingFriendRequest(@Param("fromUserId") String fromUserId, @Param("toUserId") String toUserId);

 // Gửi friend request
 @Modifying
 @Query("""
        MATCH (u1:User {id: $fromUserId}), (u2:User {id: $toUserId})
        MERGE (u1)-[:FRIEND_REQUEST]->(u2)
        """)
 void sendFriendRequest(@Param("fromUserId") String fromUserId, @Param("toUserId") String toUserId);

 // Chấp nhận friend request
 @Modifying
 @Query("""
        MATCH (u1:User {id: $fromUserId})-[r:FRIEND_REQUEST]->(u2:User {id: $toUserId})
        DELETE r
        CREATE (u1)-[:FRIEND_WITH]->(u2)
        CREATE (u2)-[:FRIEND_WITH]->(u1)
        """)
 void acceptFriendRequest(@Param("fromUserId") String fromUserId, @Param("toUserId") String toUserId);

 // Từ chối friend request
 @Modifying
 @Query("""
        MATCH (u1:User {id: $fromUserId})-[r:FRIEND_REQUEST]->(u2:User {id: $toUserId})
        DELETE r
        """)
 void rejectFriendRequest(@Param("fromUserId") String fromUserId, @Param("toUserId") String toUserId);

 // Xóa friendship
 @Modifying
 @Query("""
        MATCH (u1:User {id: $userId1})-[r:FRIEND_WITH]-(u2:User {id: $userId2})
        DELETE r
        """)
 void deleteFriendship(@Param("userId1") String userId1, @Param("userId2") String userId2);

 // Lấy danh sách bạn bè
 @Query("""
        MATCH (u:User {id: $userId})-[:FRIEND_WITH]-(friend:User)
        RETURN friend
        """)
 List<UserEntity> findFriends(@Param("userId") String userId);

 // Lấy friend requests nhận được
 @Query("""
        MATCH (requester:User)-[:FRIEND_REQUEST]->(u:User {id: $userId})
        RETURN requester
        """)
 List<UserEntity> findIncomingFriendRequests(@Param("userId") String userId);

 // Lấy friend requests đã gửi
 @Query("""
        MATCH (u:User {id: $userId})-[:FRIEND_REQUEST]->(receiver:User)
        RETURN receiver
        """)
 List<UserEntity> findOutgoingFriendRequests(@Param("userId") String userId);

 // Lấy bạn chung giữa 2 user
 @Query("""
        MATCH (u1:User {id: $userId1})-[:FRIEND_WITH]-(mutual:User)-[:FRIEND_WITH]-(u2:User {id: $userId2})
        RETURN mutual
        """)
 List<UserEntity> findMutualFriends(@Param("userId1") String userId1, @Param("userId2") String userId2);

 // Gợi ý kết bạn dựa trên bạn chung và profile tương tự
 @Query("""
        MATCH (u:User {id: $userId})
        MATCH (u)-[:FRIEND_WITH]-(friend)-[:FRIEND_WITH]-(suggestion:User)
        WHERE NOT (u)-[:FRIEND_WITH]-(suggestion) AND u.id <> suggestion.id
        AND NOT (u)-[:FRIEND_REQUEST]-(suggestion)
        RETURN suggestion, COUNT(*) as mutualFriends
        ORDER BY mutualFriends DESC
        LIMIT 20
        """)
 List<UserEntity> findFriendSuggestions(@Param("userId") String userId);

 // Lọc user theo tiêu chí (cùng college, faculty, major, batch)
 @Query("""
        MATCH (u:User {id: $userId})
        OPTIONAL MATCH (u)-[:STUDIES_AT|EMPLOYED_AT]->(college:College)
        OPTIONAL MATCH (u)-[:BELONGS_TO|WORKS_IN]->(faculty:Faculty)
        OPTIONAL MATCH (u)-[:ENROLLED_IN]->(major:Major)
        OPTIONAL MATCH (u)-[:IN_BATCH]->(batch:Batch)
        
        MATCH (candidate:User)
        WHERE candidate.id <> $userId
        AND (NOT $sameCollege OR (candidate)-[:STUDIES_AT|EMPLOYED_AT]->(college))
        AND (NOT $sameFaculty OR (candidate)-[:BELONGS_TO|WORKS_IN]->(faculty))
        AND (NOT $sameMajor OR (candidate)-[:ENROLLED_IN]->(major))
        AND (NOT $sameBatch OR (candidate)-[:IN_BATCH]->(batch))
        
        RETURN candidate
        """)
 List<UserEntity> findUsersWithFilters(
         @Param("userId") String userId,
         @Param("sameCollege") boolean sameCollege,
         @Param("sameFaculty") boolean sameFaculty,
         @Param("sameMajor") boolean sameMajor,
         @Param("sameBatch") boolean sameBatch
 );

 /**
  * Find users by faculty ID (for post-service news feed algorithm)
  */
 @Query("MATCH (u:User)-[:WORKS_IN|BELONGS_TO]->(f:Faculty {id: $facultyId}) RETURN u")
 List<UserEntity> findUsersByFacultyId(@Param("facultyId") String facultyId);

 /**
  * Find users by major ID (for post-service news feed algorithm)
  */
 @Query("MATCH (u:User)-[:ENROLLED_IN]->(m:Major {id: $majorId}) RETURN u")
 List<UserEntity> findUsersByMajorId(@Param("majorId") String majorId);

 /**
  * Find users by full name containing (for search functionality)
  * Uses case-insensitive search with Neo4j CONTAINS operator
  */
 @Query("MATCH (u:User) WHERE toLower(u.fullName) CONTAINS toLower($fullName) RETURN u")
 List<UserEntity> findByFullNameContainingIgnoreCase(@Param("fullName") String fullName);

 /**
  * Find users by faculty name (for search functionality)
  */
 @Query("MATCH (u:User)-[:BELONGS_TO|WORKS_IN]->(f:Faculty {name: $facultyName}) RETURN u")
 List<UserEntity> findUsersByFaculty(@Param("facultyName") String facultyName);

 /**
  * Find users by major name (for search functionality)
  */
 @Query("MATCH (u:User)-[:ENROLLED_IN]->(m:Major {name: $majorName}) RETURN u")
 List<UserEntity> findUsersByMajor(@Param("majorName") String majorName);
}
