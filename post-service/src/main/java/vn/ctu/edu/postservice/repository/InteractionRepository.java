package vn.ctu.edu.postservice.repository;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;
import vn.ctu.edu.postservice.entity.InteractionEntity;

import java.util.List;
import java.util.Optional;

@Repository
public interface InteractionRepository extends MongoRepository<InteractionEntity, String> {

    Optional<InteractionEntity> findByPostIdAndUserIdAndType(String postId, String userId, InteractionEntity.InteractionType type);

    List<InteractionEntity> findByPostId(String postId);

    List<InteractionEntity> findByUserId(String userId);

    Page<InteractionEntity> findByPostIdAndType(String postId, InteractionEntity.InteractionType type, Pageable pageable);

    long countByPostIdAndType(String postId, InteractionEntity.InteractionType type);

    long countByUserIdAndType(String userId, InteractionEntity.InteractionType type);

    void deleteByPostId(String postId);
}
