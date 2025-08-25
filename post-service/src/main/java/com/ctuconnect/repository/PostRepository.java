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

    // Missing methods that are being called in the service layer
    @Query(value = "{ 'author.id': ?0 }", sort = "{ 'createdAt': -1 }")
    Page<PostEntity> findByAuthorIdOrderByCreatedAtDesc(String authorId, Pageable pageable);

    @Query("{ 'author.id': ?0 }")
    long countByAuthorId(String authorId);

    // Enhanced search methods for advanced filtering
    @Query("{ 'tags': { $in: ?0 } }")
    Page<PostEntity> findByTagsContaining(List<String> tags, Pageable pageable);

    @Query("{ 'createdAt': { $gte: ?0, $lte: ?1 } }")
    Page<PostEntity> findByCreatedAtBetween(java.time.LocalDateTime start, java.time.LocalDateTime end, Pageable pageable);

    @Query("{ 'postType': ?0 }")
    Page<PostEntity> findByPostType(String postType, Pageable pageable);

    @Query("{ 'stats.likes': { $gte: ?0 } }")
    Page<PostEntity> findByStatsLikesGreaterThanEqual(int minLikes, Pageable pageable);

    @Query("{ 'stats.views': { $gte: ?0 } }")
    Page<PostEntity> findByStatsViewsGreaterThanEqual(int minViews, Pageable pageable);

    @Query("{ 'stats.comments': { $gte: ?0 } }")
    Page<PostEntity> findByStatsCommentsGreaterThanEqual(int minComments, Pageable pageable);

    // Complex search queries
    @Query("{ '$and': [ " +
           "{ '$or': [ " +
           "  { 'title': { $regex: ?0, $options: 'i' } }, " +
           "  { 'content': { $regex: ?0, $options: 'i' } }, " +
           "  { 'tags': { $in: [?0] } } " +
           "] }, " +
           "{ 'category': ?1 } " +
           "] }")
    Page<PostEntity> findByTextAndCategory(String searchText, String category, Pageable pageable);

    @Query("{ '$and': [ " +
           "{ '$or': [ " +
           "  { 'title': { $regex: ?0, $options: 'i' } }, " +
           "  { 'content': { $regex: ?0, $options: 'i' } } " +
           "] }, " +
           "{ 'author.id': ?1 } " +
           "] }")
    Page<PostEntity> findByTextAndAuthor(String searchText, String authorId, Pageable pageable);

    // Full-text search with multiple filters
    @Query("{ '$and': [ " +
           "{ '$or': [ " +
           "  { 'title': { $regex: ?0, $options: 'i' } }, " +
           "  { 'content': { $regex: ?0, $options: 'i' } } " +
           "] }, " +
           "{ 'tags': { $in: ?1 } }, " +
           "{ 'category': ?2 }, " +
           "{ 'createdAt': { $gte: ?3, $lte: ?4 } } " +
           "] }")
    Page<PostEntity> findByTextAndTagsAndCategoryAndDateRange(
        String searchText,
        List<String> tags,
        String category,
        java.time.LocalDateTime startDate,
        java.time.LocalDateTime endDate,
        Pageable pageable
    );

    // Aggregation queries for suggestions
    @Query("{ 'category': { $regex: ?0, $options: 'i' } }")
    List<PostEntity> findCategoriesByPartialName(String partialCategory);

    @Query("{ 'tags': { $elemMatch: { $regex: ?0, $options: 'i' } } }")
    List<PostEntity> findTagsByPartialName(String partialTag);

    @Query("{ 'author.name': { $regex: ?0, $options: 'i' } }")
    List<PostEntity> findAuthorsByPartialName(String partialName);

    // Popular posts for trending
    List<PostEntity> findTop20ByOrderByStatsLikesDescStatsViewsDesc();

    List<PostEntity> findTop20ByOrderByStatsViewsDescCreatedAtDesc();

    // Recent popular posts (last 7 days)
    @Query("{ 'createdAt': { $gte: ?0 } }")
    List<PostEntity> findRecentPosts(java.time.LocalDateTime since, Pageable pageable);



}
