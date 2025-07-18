package vn.ctu.edu.postservice.repository;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;
import vn.ctu.edu.postservice.entity.CommentEntity;

import java.util.List;

@Repository
public interface CommentRepository extends MongoRepository<CommentEntity, String> {

    Page<CommentEntity> findByPostId(String postId, Pageable pageable);

    List<CommentEntity> findByPostIdAndParentCommentIdIsNull(String postId);

    List<CommentEntity> findByParentCommentId(String parentCommentId);

    long countByPostId(String postId);

    long countByAuthorId(String authorId);

    void deleteByPostId(String postId);
}
