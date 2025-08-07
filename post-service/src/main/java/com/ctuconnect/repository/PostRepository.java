package com.ctuconnect.repository;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;
import com.ctuconnect.entity.PostEntity;

import java.util.List;

@Repository
public interface PostRepository extends MongoRepository<PostEntity, String> {

    // Find by author ID (using nested author object)
    @Query("{ 'author.id': ?0 }")
    Page<PostEntity> findByAuthor_Id(String authorId, Pageable pageable);

    // Find all posts by author ID (for data consistency updates)
    @Query("{ 'author.id': ?0 }")
    List<PostEntity> findByAuthor_Id(String authorId);

    Page<PostEntity> findByCategory(String category, Pageable pageable);

    Page<PostEntity> findByTagsIn(List<String> tags, Pageable pageable);

    @Query("{ 'title': { $regex: ?0, $options: 'i' } }")
    Page<PostEntity> findByTitleContainingIgnoreCase(String title, Pageable pageable);

    @Query("{ '$or': [ { 'title': { $regex: ?0, $options: 'i' } }, { 'content': { $regex: ?1, $options: 'i' } } ] }")
    Page<PostEntity> findByTitleContainingOrContentContaining(String titleTerm, String contentTerm, Pageable pageable);

    // Enhanced search with category and content filters
    @Query("{ 'category': ?0, '$or': [ { 'title': { $regex: ?1, $options: 'i' } }, { 'content': { $regex: ?2, $options: 'i' } } ] }")
    Page<PostEntity> findByCategoryAndTitleContainingOrContentContaining(String category, String titleTerm, String contentTerm, Pageable pageable);

    // Find by visibility
    Page<PostEntity> findByVisibility(String visibility, Pageable pageable);

    // Find by author and visibility
    @Query("{ 'author.id': ?0, 'visibility': ?1 }")
    Page<PostEntity> findByAuthor_IdAndVisibility(String authorId, String visibility, Pageable pageable);

    List<PostEntity> findTop10ByOrderByStatsViewsDesc();

    List<PostEntity> findTop10ByOrderByStatsLikesDesc();

    @Query("{ 'author.id': ?0 }")
    long countByAuthor_Id(String authorId);

    // Add method that AdminController is calling (without underscore)
    @Query("{ 'author.id': ?0 }")
    long countByAuthorId(String authorId);

    List<PostEntity> findByAuthorIdOrderByCreatedAtDesc(String userId, Pageable pageable);
}
