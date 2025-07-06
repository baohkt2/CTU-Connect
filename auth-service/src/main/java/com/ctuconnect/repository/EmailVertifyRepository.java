package com.ctuconnect.repository;

import com.ctuconnect.entity.EmailVertify;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface EmailVertifyRepository extends JpaRepository<EmailVertify, Long>{
    boolean existsByToken(String token);
    Optional<EmailVertify> findByToken(String token);

    boolean existsByTokenAndExpiryDateLessThan(String token, Long currentTime);
    void deleteByToken(String token);

    // This method needs a custom implementation as it can't be derived by Spring Data JPA
    // @Query or a custom implementation is needed
    // String GenerateToken(String email);
}
