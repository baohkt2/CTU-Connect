package vn.ctu.edu.recommend.repository.redis;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.time.Duration;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;

/**
 * Redis cache service for embeddings and recommendations
 * 
 * Cache Strategy:
 * - Recommendations are cached with 6h TTL
 * - When returned to client, items are marked as "viewed"
 * - Viewed items are excluded from cache for a configurable period (default 24h)
 * - This prevents showing the same suggestions repeatedly
 */
@Service
@Slf4j
@RequiredArgsConstructor
public class RedisCacheService {

    private final RedisTemplate<String, Object> redisTemplate;
    private final ObjectMapper objectMapper;

    private static final String EMBEDDING_KEY_PREFIX = "embedding:";
    private static final String RECOMMENDATION_KEY_PREFIX = "recommend:";
    private static final String USER_PROFILE_KEY_PREFIX = "user:";
    
    // New: Viewed items tracking
    private static final String VIEWED_FRIENDS_KEY_PREFIX = "viewed:friends:";
    private static final String VIEWED_POSTS_KEY_PREFIX = "viewed:posts:";
    private static final int DEFAULT_VIEWED_TTL_HOURS = 24; // Items reappear after 24 hours

    // Embedding Cache
    public void cacheEmbedding(String postId, float[] embedding, Duration ttl) {
        try {
            String key = EMBEDDING_KEY_PREFIX + postId;
            redisTemplate.opsForValue().set(key, embedding, ttl.toSeconds(), TimeUnit.SECONDS);
            log.debug("Cached embedding for post: {}", postId);
        } catch (Exception e) {
            log.error("Failed to cache embedding for post: {}", postId, e);
        }
    }

    public float[] getEmbedding(String postId) {
        try {
            String key = EMBEDDING_KEY_PREFIX + postId;
            Object value = redisTemplate.opsForValue().get(key);
            if (value instanceof float[]) {
                return (float[]) value;
            }
            return null;
        } catch (Exception e) {
            log.error("Failed to get embedding for post: {}", postId, e);
            return null;
        }
    }

    // Recommendation Cache
    public <T> void cacheRecommendations(String userId, List<T> recommendations, Duration ttl) {
        try {
            String key = RECOMMENDATION_KEY_PREFIX + userId;
            String json = objectMapper.writeValueAsString(recommendations);
            redisTemplate.opsForValue().set(key, json, ttl.toSeconds(), TimeUnit.SECONDS);
            log.debug("Cached recommendations for user: {}", userId);
        } catch (JsonProcessingException e) {
            log.error("Failed to cache recommendations for user: {}", userId, e);
        }
    }

    /**
     * Cache recommendations with explicit cache key (for friend suggestions, etc.)
     */
    public <T> void cacheRecommendations(String cacheKey, List<T> recommendations, Duration ttl, boolean useRawKey) {
        try {
            String key = useRawKey ? cacheKey : RECOMMENDATION_KEY_PREFIX + cacheKey;
            String json = objectMapper.writeValueAsString(recommendations);
            redisTemplate.opsForValue().set(key, json, ttl.toSeconds(), TimeUnit.SECONDS);
            log.debug("Cached recommendations with key: {}", key);
        } catch (JsonProcessingException e) {
            log.error("Failed to cache recommendations with key: {}", cacheKey, e);
        }
    }

    @SuppressWarnings("unchecked")
    public <T> List<T> getRecommendations(String userId, Class<T> clazz) {
        try {
            String key = RECOMMENDATION_KEY_PREFIX + userId;
            Object value = redisTemplate.opsForValue().get(key);
            if (value instanceof String) {
                return objectMapper.readValue(
                    (String) value,
                    objectMapper.getTypeFactory().constructCollectionType(List.class, clazz)
                );
            }
            return null;
        } catch (Exception e) {
            log.error("Failed to get recommendations for user: {}", userId, e);
            return null;
        }
    }

    /**
     * Get recommendations with explicit cache key
     */
    @SuppressWarnings("unchecked")
    public <T> List<T> getRecommendations(String cacheKey, Class<T> clazz, boolean useRawKey) {
        try {
            String key = useRawKey ? cacheKey : RECOMMENDATION_KEY_PREFIX + cacheKey;
            Object value = redisTemplate.opsForValue().get(key);
            if (value instanceof String) {
                return objectMapper.readValue(
                    (String) value,
                    objectMapper.getTypeFactory().constructCollectionType(List.class, clazz)
                );
            }
            return null;
        } catch (Exception e) {
            log.error("Failed to get recommendations with key: {}", cacheKey, e);
            return null;
        }
    }

    // User Profile Cache
    public <T> void cacheUserProfile(String userId, T profile, Duration ttl) {
        try {
            String key = USER_PROFILE_KEY_PREFIX + userId;
            redisTemplate.opsForValue().set(key, profile, ttl.toSeconds(), TimeUnit.SECONDS);
            log.debug("Cached user profile: {}", userId);
        } catch (Exception e) {
            log.error("Failed to cache user profile: {}", userId, e);
        }
    }

    @SuppressWarnings("unchecked")
    public <T> T getUserProfile(String userId, Class<T> clazz) {
        try {
            String key = USER_PROFILE_KEY_PREFIX + userId;
            Object value = redisTemplate.opsForValue().get(key);
            if (value != null && clazz.isInstance(value)) {
                return (T) value;
            }
            return null;
        } catch (Exception e) {
            log.error("Failed to get user profile: {}", userId, e);
            return null;
        }
    }

    // Cache Invalidation
    public void invalidateEmbedding(String postId) {
        String key = EMBEDDING_KEY_PREFIX + postId;
        redisTemplate.delete(key);
        log.debug("Invalidated embedding cache for post: {}", postId);
    }

    public void invalidateRecommendations(String userId) {
        String key = RECOMMENDATION_KEY_PREFIX + userId;
        redisTemplate.delete(key);
        log.debug("Invalidated recommendations cache for user: {}", userId);
    }

    public void invalidateUserProfile(String userId) {
        String key = USER_PROFILE_KEY_PREFIX + userId;
        redisTemplate.delete(key);
        log.debug("Invalidated user profile cache: {}", userId);
    }

    public void invalidateAllRecommendations() {
        Set<String> keys = redisTemplate.keys(RECOMMENDATION_KEY_PREFIX + "*");
        if (keys != null && !keys.isEmpty()) {
            redisTemplate.delete(keys);
            log.info("Invalidated {} recommendation caches", keys.size());
        }
    }

    public void invalidateAllEmbeddings() {
        Set<String> keys = redisTemplate.keys(EMBEDDING_KEY_PREFIX + "*");
        if (keys != null && !keys.isEmpty()) {
            redisTemplate.delete(keys);
            log.info("Invalidated {} embedding caches", keys.size());
        }
    }

    // Cache Statistics
    public boolean exists(String key) {
        return Boolean.TRUE.equals(redisTemplate.hasKey(key));
    }

    public Long getTtl(String key) {
        return redisTemplate.getExpire(key, TimeUnit.SECONDS);
    }

    /* ==================== VIEWED ITEMS TRACKING (DISABLED) ====================
     * These methods were part of a viewed-items tracking feature that has been disabled.
     * Keeping them commented for potential future use.
     
    public void markFriendsAsViewed(String userId, List<String> viewedUserIds) {
        markFriendsAsViewed(userId, viewedUserIds, DEFAULT_VIEWED_TTL_HOURS);
    }
    
    public void markFriendsAsViewed(String userId, List<String> viewedUserIds, int ttlHours) {
        try {
            String key = VIEWED_FRIENDS_KEY_PREFIX + userId;
            for (String viewedId : viewedUserIds) {
                redisTemplate.opsForSet().add(key, viewedId);
            }
            redisTemplate.expire(key, ttlHours, TimeUnit.HOURS);
            log.debug("Marked {} friends as viewed for user: {}", viewedUserIds.size(), userId);
        } catch (Exception e) {
            log.warn("Failed to mark friends as viewed: {}", e.getMessage());
        }
    }
    
    public Set<String> getViewedFriends(String userId) {
        try {
            String key = VIEWED_FRIENDS_KEY_PREFIX + userId;
            Set<Object> members = redisTemplate.opsForSet().members(key);
            if (members != null) {
                return (Set<String>) (Set<?>) members;
            }
        } catch (Exception e) {
            log.warn("Failed to get viewed friends: {}", e.getMessage());
        }
        return Set.of();
    }
    
    public void markPostsAsViewed(String userId, List<String> viewedPostIds) {
        markPostsAsViewed(userId, viewedPostIds, DEFAULT_VIEWED_TTL_HOURS);
    }
    
    public void markPostsAsViewed(String userId, List<String> viewedPostIds, int ttlHours) {
        try {
            String key = VIEWED_POSTS_KEY_PREFIX + userId;
            for (String postId : viewedPostIds) {
                redisTemplate.opsForSet().add(key, postId);
            }
            redisTemplate.expire(key, ttlHours, TimeUnit.HOURS);
            log.debug("Marked {} posts as viewed for user: {}", viewedPostIds.size(), userId);
        } catch (Exception e) {
            log.warn("Failed to mark posts as viewed: {}", e.getMessage());
        }
    }
    
    public Set<String> getViewedPosts(String userId) {
        try {
            String key = VIEWED_POSTS_KEY_PREFIX + userId;
            Set<Object> members = redisTemplate.opsForSet().members(key);
            if (members != null) {
                return (Set<String>) (Set<?>) members;
            }
        } catch (Exception e) {
            log.warn("Failed to get viewed posts: {}", e.getMessage());
        }
        return Set.of();
    }
    
    public void clearViewedFriends(String userId) {
        try {
            String key = VIEWED_FRIENDS_KEY_PREFIX + userId;
            redisTemplate.delete(key);
            log.debug("Cleared viewed friends for user: {}", userId);
        } catch (Exception e) {
            log.warn("Failed to clear viewed friends: {}", e.getMessage());
        }
    }
    
    public void clearViewedPosts(String userId) {
        try {
            String key = VIEWED_POSTS_KEY_PREFIX + userId;
            redisTemplate.delete(key);
            log.debug("Cleared viewed posts for user: {}", userId);
        } catch (Exception e) {
            log.warn("Failed to clear viewed posts: {}", e.getMessage());
        }
    }
    
    public void removeFromFriendSuggestionsCache(String userId, List<String> returnedUserIds) {
        invalidateRecommendations(RECOMMENDATION_KEY_PREFIX + "friend:" + userId);
        markFriendsAsViewed(userId, returnedUserIds);
    }
    
    public void removeFromPostRecommendationsCache(String userId, List<String> returnedPostIds) {
        invalidateRecommendations(userId);
        markPostsAsViewed(userId, returnedPostIds);
    }
    */
}
