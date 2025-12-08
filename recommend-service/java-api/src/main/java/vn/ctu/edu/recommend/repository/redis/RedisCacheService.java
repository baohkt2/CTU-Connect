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
}
