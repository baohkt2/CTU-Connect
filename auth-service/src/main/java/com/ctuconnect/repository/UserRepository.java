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

@Repository
public interface UserRepository extends JpaRepository<UserEntity, Long> {

    Optional<UserEntity> findByEmail(String email);

    Optional<UserEntity> findByUsername(String username);

    boolean existsByEmail(String email);

    boolean existsByUsername(String username);

    // Admin functionality methods
    long countByIsActive(boolean isActive);

    long countByCreatedAtAfter(LocalDateTime date);

    @Query("SELECT COUNT(u) FROM UserEntity u WHERE u.id NOT IN " +
           "(SELECT ev.user.id FROM EmailVerificationEntity ev WHERE ev.isVerified = true)")
    long countUnverifiedUsers();

    @Query("SELECT u.role, COUNT(u) FROM UserEntity u GROUP BY u.role")
    List<Object[]> countUsersByRole();

    List<UserEntity> findTop10ByOrderByCreatedAtDesc();

    Page<UserEntity> findByEmailContainingIgnoreCaseOrUsernameContainingIgnoreCase(
            String email, String username, Pageable pageable);

    Page<UserEntity> findByRole(String role, Pageable pageable);

    Page<UserEntity> findByIsActive(boolean isActive, Pageable pageable);

    List<UserEntity> findByRole(String role);

    List<UserEntity> findByIsActive(boolean isActive);

    @Query("SELECT u FROM UserEntity u WHERE u.id NOT IN " +
           "(SELECT ev.user.id FROM EmailVerificationEntity ev WHERE ev.isVerified = true)")
    List<UserEntity> findUnverifiedUsers();
}
