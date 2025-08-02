package com.ctuconnect.repository;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;
import com.ctuconnect.entity.CommentEntity;

import java.util.List;

@Repository
public interface CommentRepository extends MongoRepository<CommentEntity, String> {

    Page<CommentEntity> findByPostId(String postId, Pageable pageable);

    List<CommentEntity> findByPostIdAndParentCommentIdIsNull(String postId);

    List<CommentEntity> findByParentCommentId(String parentCommentId);

    long countByPostId(String postId);

    // Fix: Use MongoDB query for nested author object
    @Query(value = "{ 'author.id': ?0 }", count = true)
    long countByAuthor_Id(String authorId);

    void deleteByPostId(String postId);

    // Additional useful queries for nested author structure
    @Query("{ 'author.id': ?0 }")
    Page<CommentEntity> findByAuthor_Id(String authorId, Pageable pageable);

    // Add method that AdminController is calling (without underscore)
    @Query(value = "{ 'author.id': ?0 }", count = true)
    long countByAuthorId(String authorId);
}
