package vn.ctu.edu.recommend.kafka.consumer;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;
import vn.ctu.edu.recommend.kafka.event.PostEvent;
import vn.ctu.edu.recommend.model.entity.postgres.PostEmbedding;
import vn.ctu.edu.recommend.nlp.EmbeddingService;
import vn.ctu.edu.recommend.repository.postgres.PostEmbeddingRepository;
import vn.ctu.edu.recommend.repository.postgres.UserFeedbackRepository;
import vn.ctu.edu.recommend.repository.redis.RedisCacheService;

import java.time.LocalDateTime;
import java.util.Optional;

/**
 * Kafka consumer for post events
 * Automatically generates embeddings for new/updated posts
 */
@Component
@Slf4j
@RequiredArgsConstructor
public class PostEventConsumer {

    private final PostEmbeddingRepository postEmbeddingRepository;
    private final EmbeddingService embeddingService;
    private final RedisCacheService redisCacheService;
    private final UserFeedbackRepository userFeedbackRepository;

    @KafkaListener(topics = "post_created", groupId = "recommendation-service-group", containerFactory = "kafkaListenerContainerFactory")
    public void handlePostCreated(PostEvent event) {
        log.info("📥 Received post_created event: postId={}", event.getPostId());

        try {
            String postId = event.getPostId();
            String content = extractContent(event);
            String authorId = event.getAuthorId();
            
            if (postId == null || content == null) {
                log.warn("⚠️  Invalid post event - missing postId or content");
                return;
            }
            
            // Check if post already exists
            if (postEmbeddingRepository.existsByPostId(postId)) {
                log.info("Post {} already exists, skipping", postId);
                return;
            }

            log.info("🔄 Generating embedding for post: {}", postId);
            
            // Generate embedding
            float[] embedding = embeddingService.generateEmbedding(content, postId);

            // Simple category detection (Python model will provide better classification)
            String category = detectSimpleCategory(content);
            float academicScore = calculateBasicAcademicScore(content);
            
            log.info("📊 Basic classification: category={}, score={}", category, academicScore);

            // Extract tags from event
            String[] tags = extractTags(event);

            // Create post embedding entity
            PostEmbedding postEmbedding = PostEmbedding.builder()
                .postId(postId)
                .authorId(authorId)
                .content(content)
                .academicScore(academicScore)
                .academicCategory(category)
                .popularityScore(0.0f)
                .contentSimilarityScore(0.0f)
                .graphRelationScore(0.0f)
                .likeCount(0)
                .commentCount(0)
                .shareCount(0)
                .viewCount(0)
                .tags(tags)
                .embeddingUpdatedAt(LocalDateTime.now())
                .build();

            postEmbedding.setEmbeddingVectorFromArray(embedding);
            postEmbeddingRepository.save(postEmbedding);

            log.info("✅ Successfully processed post_created event for: {}", postId);

            // Invalidate related caches
            redisCacheService.invalidateAllRecommendations();

        } catch (Exception e) {
            log.error("❌ Error processing post_created event: {}", e.getMessage(), e);
        }
    }
    
    /**
     * Simple category detection based on keywords
     * Python model will provide better ML-based classification
     */
    private String detectSimpleCategory(String content) {
        if (content == null || content.isEmpty()) {
            return "GENERAL";
        }
        
        String lower = content.toLowerCase();
        
        // Quick keyword matching
        if (lower.contains("học bổng") || lower.contains("scholarship")) return "SCHOLARSHIP";
        if (lower.contains("sự kiện") || lower.contains("event")) return "EVENT";
        if (lower.contains("nghiên cứu") || lower.contains("research")) return "RESEARCH";
        if (lower.contains("thông báo") || lower.contains("announcement")) return "ANNOUNCEMENT";
        if (lower.contains("hỏi") || lower.contains("câu hỏi") || lower.contains("question")) return "QA";
        if (lower.contains("khóa học") || lower.contains("course")) return "COURSE";
        
        return "GENERAL";
    }
    
    /**
     * Calculate basic academic score based on content characteristics
     */
    private float calculateBasicAcademicScore(String content) {
        if (content == null || content.isEmpty()) {
            return 0.3f;
        }
        
        float score = 0.3f; // Base score
        
        // Academic keywords boost
        String lower = content.toLowerCase();
        if (lower.matches(".*(nghiên cứu|research|học thuật|academic).*")) score += 0.2f;
        if (lower.matches(".*(phương pháp|method|phân tích|analysis).*")) score += 0.15f;
        if (lower.matches(".*(kết quả|result|dữ liệu|data).*")) score += 0.1f;
        
        // Length indicates more structured content
        if (content.length() > 200) score += 0.1f;
        if (content.length() > 500) score += 0.1f;
        
        return Math.min(1.0f, score);
    }
    
    /**
     * Extract content from event, handling both nested and flat structures
     */
    private String extractContent(PostEvent event) {
        if (event.getData() != null && event.getData().getContent() != null) {
            return event.getData().getContent();
        }
        return event.getContent();
    }
    
    /**
     * Extract tags from event
     */
    private String[] extractTags(PostEvent event) {
        if (event.getData() != null && event.getData().getTags() != null) {
            return event.getData().getTags().toArray(new String[0]);
        }
        return event.getTags();
    }

    @KafkaListener(topics = "post_updated", groupId = "recommendation-service-group", containerFactory = "kafkaListenerContainerFactory")
    public void handlePostUpdated(PostEvent event) {
        log.info("Received post_updated event: {}", event.getPostId());

        try {
            PostEmbedding postEmbedding = postEmbeddingRepository
                .findByPostId(event.getPostId())
                .orElse(null);

            if (postEmbedding == null) {
                log.warn("Post {} not found, creating new entry", event.getPostId());
                handlePostCreated(event);
                return;
            }

            // Regenerate embedding if content changed
            if (!event.getContent().equals(postEmbedding.getContent())) {
                float[] embedding = embeddingService.generateEmbedding(
                    event.getContent(), event.getPostId());
                
                String category = detectSimpleCategory(event.getContent());
                float academicScore = calculateBasicAcademicScore(event.getContent());

                postEmbedding.setContent(event.getContent());
                postEmbedding.setEmbeddingVectorFromArray(embedding);
                postEmbedding.setAcademicScore(academicScore);
                postEmbedding.setAcademicCategory(category);
                postEmbedding.setEmbeddingUpdatedAt(LocalDateTime.now());
                postEmbedding.setTags(event.getTags());

                postEmbeddingRepository.save(postEmbedding);

                log.info("Successfully updated post embedding: {}", event.getPostId());

                // Invalidate caches
                redisCacheService.invalidateEmbedding(event.getPostId());
                redisCacheService.invalidateAllRecommendations();
            }

        } catch (Exception e) {
            log.error("Error processing post_updated event: {}", e.getMessage(), e);
        }
    }

    @KafkaListener(topics = "post_deleted", groupId = "recommendation-service-group", containerFactory = "kafkaListenerContainerFactory")
    public void handlePostDeleted(PostEvent event) {
        log.info("📥 Received post_deleted event: postId={}", event.getPostId());

        try {
            String postId = event.getPostId();
            
            // Delete from post_embeddings
            Optional<PostEmbedding> embeddingOpt = postEmbeddingRepository.findByPostId(postId);
            if (embeddingOpt.isPresent()) {
                postEmbeddingRepository.deleteByPostId(postId);
                log.info("✅ Deleted post embedding for: {}", postId);
            } else {
                log.warn("⚠️  Post embedding not found for deletion: {}", postId);
            }
            
            // Delete from user_feedback
            int deletedFeedback = userFeedbackRepository.deleteByPostId(postId);
            log.info("🗑️  Deleted {} user feedback records for post: {}", deletedFeedback, postId);
            
            // Invalidate caches
            redisCacheService.invalidateEmbedding(postId);
            redisCacheService.invalidateAllRecommendations();
            
            log.info("✅ Successfully processed post_deleted event for: {}", postId);

        } catch (Exception e) {
            log.error("❌ Error processing post_deleted event: {}", e.getMessage(), e);
        }
    }
}
