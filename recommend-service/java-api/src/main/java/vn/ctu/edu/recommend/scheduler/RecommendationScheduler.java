package vn.ctu.edu.recommend.scheduler;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import vn.ctu.edu.recommend.service.RecommendationService;

/**
 * Scheduled tasks for recommendation service
 */
@Component
@Slf4j
@RequiredArgsConstructor
public class RecommendationScheduler {

    private final RecommendationService recommendationService;

    /**
     * Rebuild embeddings every 5 minutes (as specified in config)
     */
    @Scheduled(cron = "${recommendation.batch.rebuild-cron}")
    public void rebuildEmbeddingsTask() {
        log.info("Starting scheduled embedding rebuild task");
        
        try {
            recommendationService.rebuildEmbeddings();
            log.info("Completed scheduled embedding rebuild task");
        } catch (Exception e) {
            log.error("Error in scheduled embedding rebuild task", e);
        }
    }

    /**
     * Clean up expired recommendation cache every hour
     */
    @Scheduled(fixedRate = 3600000) // 1 hour
    public void cleanupExpiredCache() {
        log.info("Starting scheduled cache cleanup task");
        
        try {
            // Cache cleanup is handled by Redis TTL
            // This is a placeholder for additional cleanup logic if needed
            log.info("Completed scheduled cache cleanup task");
        } catch (Exception e) {
            log.error("Error in scheduled cache cleanup task", e);
        }
    }
}
