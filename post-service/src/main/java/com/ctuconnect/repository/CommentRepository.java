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

    // Basic queries
    Page<CommentEntity> findByPostId(String postId, Pageable pageable);

    // Root comments only (depth = 0 or null)
    @Query("{ 'postId': ?0, $or: [ { 'depth': { $exists: false } }, { 'depth': 0 } ] }")
    List<CommentEntity> findRootCommentsByPostId(String postId);

    @Query("{ 'postId': ?0, $or: [ { 'depth': { $exists: false } }, { 'depth': 0 } ] }")
    Page<CommentEntity> findRootCommentsByPostId(String postId, Pageable pageable);

    // Direct replies to a comment (depth-aware)
    @Query("{ 'parentCommentId': ?0, 'depth': { $lt: ?1 } }")
    List<CommentEntity> findDirectReplies(String parentCommentId, int maxDepth);

    // All replies under a root comment (flattened)
    List<CommentEntity> findByRootCommentIdOrderByCreatedAtAsc(String rootCommentId);

    // Flattened comments for a specific root comment
    @Query("{ 'rootCommentId': ?0, 'depth': { $gte: ?1 } }")
    List<CommentEntity> findFlattenedReplies(String rootCommentId, int minDepth);

    // Legacy support
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

    // Method for user profile synchronization - find all comments by author ID
    @Query("{ 'author.id': ?0 }")
    List<CommentEntity> findByAuthor_Id(String authorId);

    // Enhanced depth-based queries
    @Query("{ 'postId': ?0, 'parentCommentId': ?1, 'depth': ?2 }")
    List<CommentEntity> findByPostIdAndParentCommentIdAndDepth(String postId, String parentCommentId, int depth);

    // Count replies for a comment (including flattened ones)
    @Query(value = "{ $or: [ { 'parentCommentId': ?0 }, { 'rootCommentId': ?0 } ] }", count = true)
    long countTotalReplies(String commentId);

    // Delete all replies under a comment (including flattened)
    @Query("{ $or: [ { 'parentCommentId': ?0 }, { 'rootCommentId': ?0 } ] }")
    void deleteAllRepliesUnderComment(String commentId);
}
