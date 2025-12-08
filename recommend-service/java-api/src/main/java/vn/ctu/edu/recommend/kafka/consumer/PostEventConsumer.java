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
        log.info("Received post_created event: {}", event.getPostId());

        try {
            // Check if post already exists
            if (postEmbeddingRepository.existsByPostId(event.getPostId())) {
                log.info("Post {} already exists, skipping", event.getPostId());
                return;
            }

            // Generate embedding
            float[] embedding = embeddingService.generateEmbedding(
                event.getContent(), event.getPostId());

            // Classify content
            ClassificationResponse classification = 
                academicClassifier.classify(event.getContent());

            // Create post embedding entity
            PostEmbedding postEmbedding = PostEmbedding.builder()
                .postId(event.getPostId())
                .authorId(event.getAuthorId())
                .content(event.getContent())
                .academicScore(classification.getConfidence())
                .academicCategory(classification.getCategory())
                .popularityScore(0.0f)
                .likeCount(0)
                .commentCount(0)
                .shareCount(0)
                .viewCount(0)
                .tags(event.getTags())
                .embeddingUpdatedAt(LocalDateTime.now())
                .build();

            postEmbedding.setEmbeddingVectorFromArray(embedding);
            postEmbeddingRepository.save(postEmbedding);

            log.info("Successfully processed post_created event for: {}", event.getPostId());

            // Invalidate related caches
            redisCacheService.invalidateAllRecommendations();

        } catch (Exception e) {
            log.error("Error processing post_created event: {}", e.getMessage(), e);
        }
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
