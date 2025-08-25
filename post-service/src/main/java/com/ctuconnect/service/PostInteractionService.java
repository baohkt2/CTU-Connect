package com.ctuconnect.service;

import com.ctuconnect.entity.PostEntity;
import com.ctuconnect.repository.PostRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.TimeUnit;

@Service
@RequiredArgsConstructor
@Slf4j
public class PostInteractionService {

    private final PostRepository postRepository;
    private final RedisTemplate<String, Object> redisTemplate;
    private final RecommendationService recommendationService;

    private static final String USER_INTERACTIONS_PREFIX = "user_interactions:";
    private static final String POST_INTERACTIONS_PREFIX = "post_interactions:";
    private static final String USER_PREFERENCES_PREFIX = "user_preferences:";
    private static final int INTERACTION_TTL_DAYS = 90; // 3 tháng

    /**
     * Ghi lại tương tác xem bài viết
     */
    public void recordPostView(String userId, String postId) {
        try {
            // Cập nhật số view của bài viết
            updatePostStats(postId, "views", 1);

            // Ghi lại tương tác của người dùng
            recordUserInteraction(userId, postId, InteractionType.VIEW, 1.0);

            // Cập nhật preferences dựa trên bài viết đã xem
            updateUserPreferencesFromPost(userId, postId, 0.1); // Trọng số thấp cho view

            log.debug("Recorded view interaction: user={}, post={}", userId, postId);

        } catch (Exception e) {
            log.error("Error recording post view: user={}, post={}, error={}", userId, postId, e.getMessage());
        }
    }

    /**
     * Ghi lại tương tác like bài viết
     */
    public void recordPostLike(String userId, String postId, boolean isLike) {
        try {
            // Cập nhật số like của bài viết
            int delta = isLike ? 1 : -1;
            updatePostStats(postId, "likes", delta);

            // Ghi lại tương tác của người dùng
            double interactionScore = isLike ? 3.0 : -3.0; // Like có trọng số cao
            recordUserInteraction(userId, postId, InteractionType.LIKE, interactionScore);

            // Cập nhật preferences mạnh mẽ hơn khi like
            if (isLike) {
                updateUserPreferencesFromPost(userId, postId, 0.5);
            }

            // Invalidate recommendation cache để cập nhật gợi ý
            recommendationService.invalidateRecommendationCache(userId);

            log.debug("Recorded like interaction: user={}, post={}, isLike={}", userId, postId, isLike);

        } catch (Exception e) {
            log.error("Error recording post like: user={}, post={}, error={}", userId, postId, e.getMessage());
        }
    }

    /**
     * Ghi lại tương tác comment bài viết
     */
    public void recordPostComment(String userId, String postId) {
        try {
            // Cập nhật số comment của bài viết
            updatePostStats(postId, "comments", 1);

            // Comment có trọng số cao vì thể hiện sự quan tâm sâu sắc
            recordUserInteraction(userId, postId, InteractionType.COMMENT, 4.0);

            // Cập nhật preferences mạnh
            updateUserPreferencesFromPost(userId, postId, 0.7);

            // Invalidate cache
            recommendationService.invalidateRecommendationCache(userId);

            log.debug("Recorded comment interaction: user={}, post={}", userId, postId);

        } catch (Exception e) {
            log.error("Error recording post comment: user={}, post={}, error={}", userId, postId, e.getMessage());
        }
    }

    /**
     * Ghi lại tương tác share bài viết
     */
    public void recordPostShare(String userId, String postId) {
        try {
            // Cập nhật số share của bài viết
            updatePostStats(postId, "shares", 1);

            // Share có trọng số cao nhất vì cho thấy người dùng thực sự thích nội dung
            recordUserInteraction(userId, postId, InteractionType.SHARE, 5.0);

            // Cập nhật preferences rất mạnh
            updateUserPreferencesFromPost(userId, postId, 1.0);

            // Invalidate cache
            recommendationService.invalidateRecommendationCache(userId);

            log.debug("Recorded share interaction: user={}, post={}", userId, postId);

        } catch (Exception e) {
            log.error("Error recording post share: user={}, post={}, error={}", userId, postId, e.getMessage());
        }
    }

    /**
     * Ghi lại thời gian đọc bài viết (dwell time)
     */
    public void recordReadingTime(String userId, String postId, long readingTimeSeconds) {
        try {
            // Tính điểm dựa trên thời gian đọc (tối đa 5 điểm cho 5 phút)
            double score = Math.min(5.0, readingTimeSeconds / 60.0);

            recordUserInteraction(userId, postId, InteractionType.READ_TIME, score);

            // Nếu đọc lâu (>30s), cập nhật preferences
            if (readingTimeSeconds > 30) {
                double preferenceWeight = Math.min(0.3, readingTimeSeconds / 300.0); // Max 0.3 cho 5 phút
                updateUserPreferencesFromPost(userId, postId, preferenceWeight);
            }

            log.debug("Recorded reading time: user={}, post={}, time={}s, score={}",
                    userId, postId, readingTimeSeconds, score);

        } catch (Exception e) {
            log.error("Error recording reading time: user={}, post={}, error={}", userId, postId, e.getMessage());
        }
    }

    /**
     * Lấy lịch sử tương tác của người dùng
     */
    public Map<String, Double> getUserInteractionHistory(String userId, int days) {
        try {
            String cacheKey = USER_INTERACTIONS_PREFIX + userId;
            Map<String, Double> interactions = (Map<String, Double>) redisTemplate.opsForValue().get(cacheKey);

            return interactions != null ? interactions : new HashMap<>();

        } catch (Exception e) {
            log.error("Error getting user interaction history: user={}, error={}", userId, e.getMessage());
            return new HashMap<>();
        }
    }

    /**
     * Lấy preferences của người dùng (categories, tags)
     */
    public UserContentPreferences getUserContentPreferences(String userId) {
        try {
            String cacheKey = USER_PREFERENCES_PREFIX + userId;
            UserContentPreferences preferences = (UserContentPreferences) redisTemplate.opsForValue().get(cacheKey);

            return preferences != null ? preferences : new UserContentPreferences();

        } catch (Exception e) {
            log.error("Error getting user content preferences: user={}, error={}", userId, e.getMessage());
            return new UserContentPreferences();
        }
    }

    /**
     * Tính toán điểm tương tác tổng thể của người dùng với bài viết
     */
    public double calculateUserPostAffinityScore(String userId, String postId) {
        try {
            Map<String, Double> interactions = getUserInteractionHistory(userId, 30);
            return interactions.getOrDefault(postId, 0.0);
        } catch (Exception e) {
            log.error("Error calculating user-post affinity: user={}, post={}, error={}", userId, postId, e.getMessage());
            return 0.0;
        }
    }

    // Private helper methods

    private void updatePostStats(String postId, String statType, int delta) {
        try {
            Optional<PostEntity> postOpt = postRepository.findById(postId);
            if (postOpt.isPresent()) {
                PostEntity post = postOpt.get();
                if (post.getStats() != null) {
                    switch (statType) {
                        case "views":
                            post.getStats().setViews(Math.max(0, post.getStats().getViews() + delta));
                            break;
                        case "likes":
                            post.getStats().setLikes(Math.max(0, post.getStats().getLikes() + delta));
                            break;
                        case "comments":
                            post.getStats().setComments(Math.max(0, post.getStats().getComments() + delta));
                            break;
                        case "shares":
                            post.getStats().setShares(Math.max(0, post.getStats().getShares() + delta));
                            break;
                    }
                    postRepository.save(post);
                }
            }
        } catch (Exception e) {
            log.error("Error updating post stats: postId={}, statType={}, delta={}, error={}",
                    postId, statType, delta, e.getMessage());
        }
    }

    @SuppressWarnings("unchecked")
    private void recordUserInteraction(String userId, String postId, InteractionType type, double score) {
        try {
            String cacheKey = USER_INTERACTIONS_PREFIX + userId;

            // Lấy interactions hiện tại
            Map<String, Double> interactions = (Map<String, Double>) redisTemplate.opsForValue().get(cacheKey);
            if (interactions == null) {
                interactions = new HashMap<>();
            }

            // Cập nhật điểm tương tác (cộng dồn)
            String interactionKey = postId + ":" + type.name();
            interactions.put(interactionKey, interactions.getOrDefault(interactionKey, 0.0) + score);

            // Tính tổng điểm cho bài viết này
            double totalScore = interactions.entrySet().stream()
                    .filter(entry -> entry.getKey().startsWith(postId + ":"))
                    .mapToDouble(Map.Entry::getValue)
                    .sum();

            interactions.put(postId, totalScore);

            // Giới hạn số lượng interactions (chỉ giữ 1000 gần nhất)
            if (interactions.size() > 1000) {
                // Loại bỏ các tương tác có điểm thấp nhất
                interactions.entrySet().removeIf(entry ->
                    !entry.getKey().contains(":") && entry.getValue() < 1.0
                );
            }

            // Lưu vào cache
            redisTemplate.opsForValue().set(cacheKey, interactions, INTERACTION_TTL_DAYS, TimeUnit.DAYS);

        } catch (Exception e) {
            log.error("Error recording user interaction: user={}, post={}, type={}, error={}",
                    userId, postId, type, e.getMessage());
        }
    }

    @SuppressWarnings("unchecked")
    private void updateUserPreferencesFromPost(String userId, String postId, double weight) {
        try {
            // Lấy thông tin bài viết
            Optional<PostEntity> postOpt = postRepository.findById(postId);
            if (!postOpt.isPresent()) return;

            PostEntity post = postOpt.get();
            String cacheKey = USER_PREFERENCES_PREFIX + userId;

            // Lấy preferences hiện tại
            UserContentPreferences preferences = (UserContentPreferences) redisTemplate.opsForValue().get(cacheKey);
            if (preferences == null) {
                preferences = new UserContentPreferences();
            }

            // Cập nhật category preferences
            if (post.getCategory() != null) {
                preferences.getCategoryScores().put(post.getCategory(),
                        preferences.getCategoryScores().getOrDefault(post.getCategory(), 0.0) + weight);
            }

            // Cập nhật tag preferences
            if (post.getTags() != null) {
                for (String tag : post.getTags()) {
                    preferences.getTagScores().put(tag,
                            preferences.getTagScores().getOrDefault(tag, 0.0) + weight);
                }
            }

            // Cập nhật author preferences
            if (post.getAuthor() != null && post.getAuthor().getId() != null) {
                preferences.getAuthorScores().put(post.getAuthor().getId(),
                        preferences.getAuthorScores().getOrDefault(post.getAuthor().getId(), 0.0) + weight);
            }

            // Lưu vào cache
            redisTemplate.opsForValue().set(cacheKey, preferences, INTERACTION_TTL_DAYS, TimeUnit.DAYS);

        } catch (Exception e) {
            log.error("Error updating user preferences: user={}, post={}, error={}", userId, postId, e.getMessage());
        }
    }

    // Enum cho các loại tương tác
    public enum InteractionType {
        VIEW, LIKE, COMMENT, SHARE, READ_TIME
    }

    // Class để lưu preferences của người dùng
    @lombok.Data
    @lombok.NoArgsConstructor
    public static class UserContentPreferences implements java.io.Serializable {
        private Map<String, Double> categoryScores = new HashMap<>();
        private Map<String, Double> tagScores = new HashMap<>();
        private Map<String, Double> authorScores = new HashMap<>();
        private LocalDateTime lastUpdated = LocalDateTime.now();
    }
}
