package vn.ctu.edu.postservice.repository;

import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;
import vn.ctu.edu.postservice.entity.PostEntity;

import java.util.List;

@Repository
public interface PostRepository extends MongoRepository<PostEntity, String> {

    Page<PostEntity> findByAuthorId(String authorId, Pageable pageable);

    Page<PostEntity> findByCategory(String category, Pageable pageable);

    Page<PostEntity> findByTagsIn(List<String> tags, Pageable pageable);

    @Query("{ 'title': { $regex: ?0, $options: 'i' } }")
    Page<PostEntity> findByTitleContainingIgnoreCase(String title, Pageable pageable);

    @Query("{ '$or': [ { 'title': { $regex: ?0, $options: 'i' } }, { 'content': { $regex: ?0, $options: 'i' } } ] }")
    Page<PostEntity> findByTitleOrContentContaining(String searchTerm, Pageable pageable);

    List<PostEntity> findTop10ByOrderByStatsViewsDesc();

    List<PostEntity> findTop10ByOrderByStatsLikesDesc();

    long countByAuthorId(String authorId);
}
