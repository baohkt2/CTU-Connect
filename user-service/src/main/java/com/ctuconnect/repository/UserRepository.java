package com.ctuconnect.repository;

import com.ctuconnect.entity.UserEntity;
import org.springframework.data.neo4j.repository.Neo4jRepository;
import org.springframework.data.neo4j.repository.query.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface UserRepository extends Neo4jRepository<UserEntity, String> {
    // Các phương thức này giờ đây hoạt động chính xác vì trường 'email' đã tồn tại trong UserEntity
    Optional<UserEntity> findByEmail(String email);
    boolean existsByEmail(String email);

    // Đã cải tiến: Sử dụng @Param để liên kết tường minh tham số với biến trong truy vấn
    @Query("MATCH (u:User {id: $userId})-[:FRIEND]->(friend:User) RETURN friend")
    List<UserEntity> findFriends(@Param("userId") String userId);

    // Đã cải tiến: Sử dụng @Param để tăng tính rõ ràng
    @Query("MATCH (u1:User {id: $userId1})-[:FRIEND]->(friend:User)<-[:FRIEND]-(u2:User {id: $userId2}) RETURN friend")
    List<UserEntity> findMutualFriends(@Param("userId1") String userId1, @Param("userId2") String userId2);

    // Find users by college
    @Query("MATCH (u:User) WHERE u.college = $college RETURN u")
    List<UserEntity> findByCollege(@Param("college") String college);

    // Find users by faculty
    @Query("MATCH (u:User) WHERE u.faculty = $faculty RETURN u")
    List<UserEntity> findByFaculty(@Param("faculty") String faculty);

    // Find users by major
    @Query("MATCH (u:User) WHERE u.major = $major RETURN u")
    List<UserEntity> findByMajor(@Param("major") String major);

    // Find users by batch
    @Query("MATCH (u:User) WHERE u.batch = $batch RETURN u")
    List<UserEntity> findByBatch(@Param("batch") String batch);

    // Find users with same college as the specified user
    @Query("MATCH (u:User {id: $userId}), (other:User) WHERE other.college = u.college AND other.id <> $userId RETURN other")
    List<UserEntity> findUsersWithSameCollege(@Param("userId") String userId);

    // Find users with same faculty as the specified user
    @Query("MATCH (u:User {id: $userId}), (other:User) WHERE other.faculty = u.faculty AND other.id <> $userId RETURN other")
    List<UserEntity> findUsersWithSameFaculty(@Param("userId") String userId);

    // Find users with same major as the specified user
    @Query("MATCH (u:User {id: $userId}), (other:User) WHERE other.major = u.major AND other.id <> $userId RETURN other")
    List<UserEntity> findUsersWithSameMajor(@Param("userId") String userId);

    // Find users with same batch as the specified user
    @Query("MATCH (u:User {id: $userId}), (other:User) WHERE other.batch = u.batch AND other.id <> $userId RETURN other")
    List<UserEntity> findUsersWithSameBatch(@Param("userId") String userId);

    // Find users with combined filters
    @Query("MATCH (u:User {id: $userId}), (other:User) " +
           "WHERE other.id <> $userId " +
           "AND ($isSameCollege = false OR other.college = u.college) " +
           "AND ($isSameFaculty = false OR other.faculty = u.faculty) " +
           "AND ($isSameMajor = false OR other.major = u.major) " +
           "AND ($isSameBatch = false OR other.batch = u.batch) " +
           "RETURN other")
    List<UserEntity> findUsersWithFilters(
            @Param("userId") String userId,
            @Param("isSameCollege") boolean isSameCollege,
            @Param("isSameFaculty") boolean isSameFaculty,
            @Param("isSameMajor") boolean isSameMajor,
            @Param("isSameBatch") boolean isSameBatch);

    // Find friends with combined filters
    @Query("MATCH (u:User {id: $userId})-[:FRIEND]->(friend:User) " +
           "WHERE ($isSameCollege = false OR friend.college = u.college) " +
           "AND ($isSameFaculty = false OR friend.faculty = u.faculty) " +
           "AND ($isSameMajor = false OR friend.major = u.major) " +
           "AND ($isSameBatch = false OR friend.batch = u.batch) " +
           "RETURN friend")
    List<UserEntity> findFriendsWithFilters(
            @Param("userId") String userId,
            @Param("isSameCollege") boolean isSameCollege,
            @Param("isSameFaculty") boolean isSameFaculty,
            @Param("isSameMajor") boolean isSameMajor,
            @Param("isSameBatch") boolean isSameBatch);
}