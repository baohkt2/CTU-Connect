package com.ctuconnect.repository;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;
import com.ctuconnect.dto.AuthorInfo;
import com.ctuconnect.entity.InteractionEntity;

import java.util.List;
import java.util.Optional;

@Repository
public interface InteractionRepository extends MongoRepository<InteractionEntity, String> {

    // Find by postId and author.id and type
    @Query("{ 'postId': ?0, 'author.id': ?1, 'type': ?2 }")
    Optional<InteractionEntity> findByPostIdAndAuthor_IdAndType(String postId, String authorId, InteractionEntity.InteractionType type);

    // Find by postId and author object and type
    Optional<InteractionEntity> findByPostIdAndAuthorAndType(String postId, AuthorInfo author, InteractionEntity.InteractionType type);

    List<InteractionEntity> findByPostId(String postId);

    @Query("{ 'author.id': ?0 }")
    List<InteractionEntity> findByAuthor_Id(String authorId);

    Page<InteractionEntity> findByPostIdAndType(String postId, InteractionEntity.InteractionType type, Pageable pageable);

    long countByPostIdAndType(String postId, InteractionEntity.InteractionType type);

    @Query(value = "{ 'author.id': ?0, 'type': ?1 }", count = true)
    long countByAuthor_IdAndType(String authorId, InteractionEntity.InteractionType type);

    void deleteByPostId(String postId);

    // Additional useful queries
    @Query("{ 'postId': ?0, 'type': 'LIKE' }")
    List<InteractionEntity> findLikesByPostId(String postId);

    @Query("{ 'author.id': ?0, 'type': 'LIKE' }")
    List<InteractionEntity> findLikesByAuthor_Id(String authorId);

    @Query("{ 'postId': ?0, 'author.id': ?1 }")
    List<InteractionEntity> findByPostIdAndAuthor_Id(String postId, String authorId);

    // Find existing interaction by postId and userId (using author.id field)
    @Query("{ 'postId': ?0, 'author.id': ?1, 'type': ?2 }")
    Optional<InteractionEntity> findByPostIdAndUserIdAndType(String postId, String userId, InteractionEntity.InteractionType type);

    // Additional methods for user profile synchronization
}
