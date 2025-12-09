package vn.ctu.edu.recommend.kafka.consumer;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;
import vn.ctu.edu.recommend.kafka.event.PostEvent;
import vn.ctu.edu.recommend.model.dto.ClassificationResponse;
import vn.ctu.edu.recommend.model.entity.postgres.PostEmbedding;
import vn.ctu.edu.recommend.nlp.AcademicClassifier;
import vn.ctu.edu.recommend.nlp.EmbeddingService;
import vn.ctu.edu.recommend.repository.postgres.PostEmbeddingRepository;
import vn.ctu.edu.recommend.repository.redis.RedisCacheService;

import java.time.LocalDateTime;

/**
 * Kafka consumer for post events
 * Automatically generates embeddings and classifications for new/updated posts
 */
@Component
@Slf4j
@RequiredArgsConstructor
public class PostEventConsumer {

    private final PostEmbeddingRepository postEmbeddingRepository;
    private final EmbeddingService embeddingService;
    private final AcademicClassifier academicClassifier;
    private final RedisCacheService redisCacheService;

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

            // Classify content
            ClassificationResponse classification = academicClassifier.classify(content);
            
            log.info("📊 Classification result: category={}, confidence={}", 
                classification.getCategory(), classification.getConfidence());

            // Extract tags from event
            String[] tags = extractTags(event);

            // Create post embedding entity
            PostEmbedding postEmbedding = PostEmbedding.builder()
                .postId(postId)
                .authorId(authorId)
                .content(content)
                .academicScore(classification.getConfidence())
                .academicCategory(classification.getCategory())
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
                
                ClassificationResponse classification = 
                    academicClassifier.classify(event.getContent());

                postEmbedding.setContent(event.getContent());
                postEmbedding.setEmbeddingVectorFromArray(embedding);
                postEmbedding.setAcademicScore(classification.getConfidence());
                postEmbedding.setAcademicCategory(classification.getCategory());
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
        log.info("Received post_deleted event: {}", event.getPostId());

        try {
            postEmbeddingRepository.deleteByPostId(event.getPostId());
            redisCacheService.invalidateEmbedding(event.getPostId());
            redisCacheService.invalidateAllRecommendations();
            
            log.info("Successfully deleted post embedding: {}", event.getPostId());

        } catch (Exception e) {
            log.error("Error processing post_deleted event: {}", e.getMessage(), e);
        }
    }
}
