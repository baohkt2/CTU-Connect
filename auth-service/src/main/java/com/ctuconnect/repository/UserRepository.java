package com.ctuconnect.repository;

import com.ctuconnect.entity.UserEntity;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

@Repository
public interface UserRepository extends JpaRepository<UserEntity, UUID> {

    @Query("SELECT u FROM UserEntity u WHERE LOWER(u.email) = LOWER(:email)")
    Optional<UserEntity> findByEmail(@Param("email") String email);

    @Query("SELECT u FROM UserEntity u WHERE LOWER(u.username) = LOWER(:username)")
    Optional<UserEntity> findByUsername(@Param("username") String username);

    @Query("SELECT CASE WHEN COUNT(u) > 0 THEN true ELSE false END FROM UserEntity u WHERE LOWER(u.email) = LOWER(:email)")
    boolean existsByEmail(@Param("email") String email);

    @Query("SELECT CASE WHEN COUNT(u) > 0 THEN true ELSE false END FROM UserEntity u WHERE LOWER(u.username) = LOWER(:username)")
    boolean existsByUsername(@Param("username") String username);

    // Admin functionality methods
    long countByIsActive(boolean isActive);

    long countByCreatedAtAfter(LocalDateTime date);

    @Query("SELECT COUNT(u) FROM UserEntity u WHERE u.id NOT IN " +
           "(SELECT ev.user.id FROM EmailVerificationEntity ev WHERE ev.isVerified = true)")
    long countUnverifiedUsers();

    @Query("SELECT u.role, COUNT(u) FROM UserEntity u GROUP BY u.role")
    List<Object[]> countUsersByRole();

    List<UserEntity> findTop10ByOrderByCreatedAtDesc();

    @Query("SELECT u FROM UserEntity u WHERE LOWER(u.email) LIKE LOWER(CONCAT('%', :email, '%')) OR LOWER(u.username) LIKE LOWER(CONCAT('%', :username, '%'))")
    Page<UserEntity> findByEmailContainingIgnoreCaseOrUsernameContainingIgnoreCase(
            @Param("email") String email, @Param("username") String username, Pageable pageable);

    Page<UserEntity> findByRole(String role, Pageable pageable);

    Page<UserEntity> findByIsActive(boolean isActive, Pageable pageable);

    List<UserEntity> findByRole(String role);

    List<UserEntity> findByIsActive(boolean isActive);

    @Query("SELECT u FROM UserEntity u WHERE LOWER(u.email) = LOWER(:identifier) OR LOWER(u.username) = LOWER(:identifier)")
    Optional<UserEntity> findByEmailOrUsername(@Param("identifier") String identifier);

    @Query("SELECT u FROM UserEntity u WHERE u.id NOT IN " +
           "(SELECT ev.user.id FROM EmailVerificationEntity ev WHERE ev.isVerified = true)")
    List<UserEntity> findUnverifiedUsers();
}
