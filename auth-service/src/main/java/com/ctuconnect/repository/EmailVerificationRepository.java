package com.ctuconnect.repository;

import com.ctuconnect.entity.EmailVerificationEntity;
import com.ctuconnect.entity.UserEntity;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;
import java.util.UUID;

@Repository
public interface EmailVerificationRepository extends JpaRepository<EmailVerificationEntity, UUID> {

    Optional<EmailVerificationEntity> findByToken(String token);

    Optional<EmailVerificationEntity> findByUser(UserEntity user);

    // Admin functionality methods
    void deleteByUser(UserEntity user);
}
