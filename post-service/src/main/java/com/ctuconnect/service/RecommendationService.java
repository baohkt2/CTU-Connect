package com.ctuconnect.service;

import com.ctuconnect.dto.response.PostResponse;
import com.ctuconnect.entity.PostEntity;
import com.ctuconnect.repository.PostRepository;
import com.ctuconnect.client.UserServiceClient;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.aggregation.Aggregation;
import org.springframework.data.mongodb.core.aggregation.AggregationResults;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class RecommendationService {

    private final PostRepository postRepository;
    private final MongoTemplate mongoTemplate;
    private final UserServiceClient userServiceClient;
    private final RedisTemplate<String, Object> redisTemplate;

    private static final String RECOMMENDATION_CACHE_PREFIX = "recommendations:";
    private static final String USER_INTERESTS_CACHE_PREFIX = "user_interests:";
    private static final String TRENDING_CACHE_KEY = "trending_posts";
    private static final int CACHE_TTL_HOURS = 2;
    private static final int MAX_RECOMMENDATIONS = 20;

    /**
     * Lấy danh sách bài viết được gợi ý cho người dùng
     */
    public List<PostResponse> getPersonalizedRecommendations(String userId, int limit) {
        String cacheKey = RECOMMENDATION_CACHE_PREFIX + userId;

        // Kiểm tra cache trước
        List<PostResponse> cachedRecommendations = getCachedRecommendations(cacheKey);
        if (cachedRecommendations != null && !cachedRecommendations.isEmpty()) {
            return cachedRecommendations.stream().limit(limit).collect(Collectors.toList());
        }

        try {
            // Lấy thông tin người dùng
            UserInterestProfile userProfile = getUserInterestProfile(userId);

            // Tạo danh sách gợi ý dựa trên nhiều yếu tố
            List<PostEntity> recommendations = generateRecommendations(userProfile);

            // Chuyển đổi sang DTO và cache kết quả
            List<PostResponse> recommendationResponses = recommendations.stream()
                    .limit(MAX_RECOMMENDATIONS)
                    .map(this::convertToResponse)
                    .collect(Collectors.toList());

            cacheRecommendations(cacheKey, recommendationResponses);

            return recommendationResponses.stream().limit(limit).collect(Collectors.toList());

        } catch (Exception e) {
            log.error("Error generating personalized recommendations for user {}: {}", userId, e.getMessage());
            // Fallback to trending posts
            return getTrendingPosts(limit);
        }
    }

    /**
     * Gợi ý bài viết tương tự dựa trên bài viết hiện tại
     */
    public List<PostResponse> getSimilarPosts(String postId, int limit) {
        try {
            PostEntity currentPost = mongoTemplate.findById(postId, PostEntity.class);
            if (currentPost == null) {
                return Collections.emptyList();
            }

            Query query = new Query();
            List<Criteria> similarityCriteria = new ArrayList<>();

            // Tìm theo category giống nhau
            if (currentPost.getCategory() != null) {
                similarityCriteria.add(Criteria.where("category").is(currentPost.getCategory()));
            }

            // Tìm theo tags tương tự
            if (currentPost.getTags() != null && !currentPost.getTags().isEmpty()) {
                similarityCriteria.add(Criteria.where("tags").in(currentPost.getTags()));
            }

            // Tìm theo tác giả (có thể quan tâm đến các bài viết khác của cùng tác giả)
            if (currentPost.getAuthor() != null) {
                similarityCriteria.add(Criteria.where("author.id").is(currentPost.getAuthor().getId()));
            }

            // Loại trừ bài viết hiện tại
            query.addCriteria(Criteria.where("id").ne(postId));

            // Chỉ hiển thị bài viết công khai
            query.addCriteria(Criteria.where("privacy").is("PUBLIC"));

            // Áp dụng các tiêu chí tương tự (OR logic)
            if (!similarityCriteria.isEmpty()) {
                query.addCriteria(new Criteria().orOperator(
                    similarityCriteria.toArray(new Criteria[0])
                ));
            }

            // Sắp xếp theo độ phổ biến và thời gian
            query.with(Sort.by(Sort.Direction.DESC, "stats.likes", "stats.views", "createdAt"));
            query.limit(limit);

            List<PostEntity> similarPosts = mongoTemplate.find(query, PostEntity.class);

            return similarPosts.stream()
                    .map(this::convertToResponse)
                    .collect(Collectors.toList());

        } catch (Exception e) {
            log.error("Error finding similar posts for {}: {}", postId, e.getMessage());
            return Collections.emptyList();
        }
    }

    /**
     * Lấy bài viết trending
     */
    public List<PostResponse> getTrendingPosts(int limit) {
        String cacheKey = TRENDING_CACHE_KEY;

        // Kiểm tra cache
        List<PostResponse> cachedTrending = getCachedRecommendations(cacheKey);
        if (cachedTrending != null && !cachedTrending.isEmpty()) {
            return cachedTrending.stream().limit(limit).collect(Collectors.toList());
        }

        try {
            // Lấy bài viết trong 7 ngày gần nhất
            LocalDateTime weekAgo = LocalDateTime.now().minusDays(7);

            Query query = new Query();
            query.addCriteria(Criteria.where("createdAt").gte(weekAgo));
            query.addCriteria(Criteria.where("privacy").is("PUBLIC"));

            List<PostEntity> recentPosts = mongoTemplate.find(query, PostEntity.class);

            // Sắp xếp theo engagement score
            List<PostEntity> trendingPosts = recentPosts.stream()
                    .sorted((a, b) -> Double.compare(
                        calculateEngagementScore(b),
                        calculateEngagementScore(a)
                    ))
                    .limit(MAX_RECOMMENDATIONS)
                    .collect(Collectors.toList());

            List<PostResponse> trendingResponses = trendingPosts.stream()
                    .map(this::convertToResponse)
                    .collect(Collectors.toList());

            // Cache kết quả
            cacheRecommendations(cacheKey, trendingResponses);

            return trendingResponses.stream().limit(limit).collect(Collectors.toList());

        } catch (Exception e) {
            log.error("Error getting trending posts: {}", e.getMessage());
            return Collections.emptyList();
        }
    }

    /**
     * Gợi ý bài viết theo danh mục
     */
    public List<PostResponse> getRecommendationsByCategory(String category, String excludePostId, int limit) {
        try {
            Query query = new Query();
            query.addCriteria(Criteria.where("category").is(category));
            query.addCriteria(Criteria.where("privacy").is("PUBLIC"));

            if (excludePostId != null) {
                query.addCriteria(Criteria.where("id").ne(excludePostId));
            }

            // Sắp xếp theo độ phổ biến trong 30 ngày gần nhất
            LocalDateTime monthAgo = LocalDateTime.now().minusDays(30);
            query.addCriteria(Criteria.where("createdAt").gte(monthAgo));
            query.with(Sort.by(Sort.Direction.DESC, "stats.likes", "stats.views", "createdAt"));
            query.limit(limit);

            List<PostEntity> posts = mongoTemplate.find(query, PostEntity.class);

            return posts.stream()
                    .map(this::convertToResponse)
                    .collect(Collectors.toList());

        } catch (Exception e) {
            log.error("Error getting recommendations by category {}: {}", category, e.getMessage());
            return Collections.emptyList();
        }
    }

    /**
     * Gợi ý bài viết cho người dùng mới (chưa có lịch sử tương tác)
     */
    public List<PostResponse> getNewUserRecommendations(String userId, int limit) {
        try {
            // Lấy thông tin cơ bản của người dùng (khoa, chuyên ngành)
            String userFaculty = userServiceClient.getUserFacultyId(userId);
            String userMajor = userServiceClient.getUserMajorId(userId);

            Query query = new Query();
            List<Criteria> criteria = new ArrayList<>();

            // Ưu tiên bài viết từ cùng khoa/chuyên ngành
            if (userFaculty != null) {
                criteria.add(Criteria.where("audienceSettings.allowedFaculties").in(userFaculty));
            }
            if (userMajor != null) {
                criteria.add(Criteria.where("audienceSettings.allowedMajors").in(userMajor));
            }

            // Hoặc bài viết công khai phổ biến
            criteria.add(Criteria.where("privacy").is("PUBLIC"));

            if (!criteria.isEmpty()) {
                query.addCriteria(new Criteria().orOperator(criteria.toArray(new Criteria[0])));
            }

            // Chỉ lấy bài viết gần đây và phổ biến
            LocalDateTime twoWeeksAgo = LocalDateTime.now().minusDays(14);
            query.addCriteria(Criteria.where("createdAt").gte(twoWeeksAgo));
            query.addCriteria(Criteria.where("stats.likes").gte(1));

            query.with(Sort.by(Sort.Direction.DESC, "stats.likes", "stats.views", "createdAt"));
            query.limit(limit);

            List<PostEntity> posts = mongoTemplate.find(query, PostEntity.class);

            return posts.stream()
                    .map(this::convertToResponse)
                    .collect(Collectors.toList());

        } catch (Exception e) {
            log.error("Error getting new user recommendations for {}: {}", userId, e.getMessage());
            return getTrendingPosts(limit);
        }
    }

    /**
     * Xóa cache gợi ý khi có thay đổi
     */
    public void invalidateRecommendationCache(String userId) {
        try {
            String userCacheKey = RECOMMENDATION_CACHE_PREFIX + userId;
            redisTemplate.delete(userCacheKey);

            // Cũng xóa cache trending nếu cần
            redisTemplate.delete(TRENDING_CACHE_KEY);

            log.debug("Invalidated recommendation cache for user: {}", userId);
        } catch (Exception e) {
            log.warn("Failed to invalidate recommendation cache: {}", e.getMessage());
        }
    }

    // Private helper methods

    private UserInterestProfile getUserInterestProfile(String userId) {
        String cacheKey = USER_INTERESTS_CACHE_PREFIX + userId;

        try {
            // Thử lấy từ cache trước
            UserInterestProfile cached = (UserInterestProfile) redisTemplate.opsForValue().get(cacheKey);
            if (cached != null) {
                return cached;
            }

            // Lấy từ user service
            Set<String> interestTags = userServiceClient.getUserInterestTags(userId);
            Set<String> preferredCategories = userServiceClient.getUserPreferredCategories(userId);
            Set<String> friendIds = userServiceClient.getFriendIds(userId);
            String facultyId = userServiceClient.getUserFacultyId(userId);
            String majorId = userServiceClient.getUserMajorId(userId);

            UserInterestProfile profile = UserInterestProfile.builder()
                    .userId(userId)
                    .interestTags(interestTags != null ? interestTags : new HashSet<>())
                    .preferredCategories(preferredCategories != null ? preferredCategories : new HashSet<>())
                    .friendIds(friendIds != null ? friendIds : new HashSet<>())
                    .facultyId(facultyId)
                    .majorId(majorId)
                    .build();

            // Cache lại
            redisTemplate.opsForValue().set(cacheKey, profile, CACHE_TTL_HOURS, TimeUnit.HOURS);

            return profile;

        } catch (Exception e) {
            log.warn("Error getting user interest profile for {}: {}", userId, e.getMessage());
            return UserInterestProfile.builder()
                    .userId(userId)
                    .interestTags(new HashSet<>())
                    .preferredCategories(new HashSet<>())
                    .friendIds(new HashSet<>())
                    .build();
        }
    }

    private List<PostEntity> generateRecommendations(UserInterestProfile profile) {
        Query query = new Query();
        List<Criteria> recommendationCriteria = new ArrayList<>();

        // 1. Bài viết từ bạn bè
        if (!profile.getFriendIds().isEmpty()) {
            recommendationCriteria.add(
                Criteria.where("author.id").in(profile.getFriendIds())
                        .and("privacy").in("PUBLIC", "FRIENDS")
            );
        }

        // 2. Bài viết theo sở thích (tags)
        if (!profile.getInterestTags().isEmpty()) {
            recommendationCriteria.add(
                Criteria.where("tags").in(profile.getInterestTags())
                        .and("privacy").is("PUBLIC")
            );
        }

        // 3. Bài viết theo danh mục ưa thích
        if (!profile.getPreferredCategories().isEmpty()) {
            recommendationCriteria.add(
                Criteria.where("category").in(profile.getPreferredCategories())
                        .and("privacy").is("PUBLIC")
            );
        }

        // 4. Bài viết từ cùng khoa/chuyên ngành
        List<Criteria> academicCriteria = new ArrayList<>();
        if (profile.getFacultyId() != null) {
            academicCriteria.add(Criteria.where("audienceSettings.allowedFaculties").in(profile.getFacultyId()));
        }
        if (profile.getMajorId() != null) {
            academicCriteria.add(Criteria.where("audienceSettings.allowedMajors").in(profile.getMajorId()));
        }
        if (!academicCriteria.isEmpty()) {
            recommendationCriteria.add(
                new Criteria().orOperator(academicCriteria.toArray(new Criteria[0]))
                        .and("privacy").is("PUBLIC")
            );
        }

        // 5. Fallback: bài viết công khai phổ biến
        recommendationCriteria.add(
            Criteria.where("privacy").is("PUBLIC")
                    .and("stats.likes").gte(5)
        );

        // Kết hợp tất cả criteria với OR
        if (!recommendationCriteria.isEmpty()) {
            query.addCriteria(new Criteria().orOperator(
                recommendationCriteria.toArray(new Criteria[0])
            ));
        }

        // Chỉ lấy bài viết gần đây (trong 30 ngày)
        LocalDateTime monthAgo = LocalDateTime.now().minusDays(30);
        query.addCriteria(Criteria.where("createdAt").gte(monthAgo));

        // Sắp xếp theo độ liên quan và phổ biến
        query.with(Sort.by(Sort.Direction.DESC, "stats.likes", "stats.views", "createdAt"));
        query.limit(MAX_RECOMMENDATIONS);

        return mongoTemplate.find(query, PostEntity.class);
    }

    private double calculateEngagementScore(PostEntity post) {
        if (post.getStats() == null) return 0.0;

        int likes = Math.toIntExact(post.getStats().getLikes());
        int views = Math.toIntExact(post.getStats().getViews());
        int comments = Math.toIntExact(post.getStats().getComments());
        int shares = Math.toIntExact(post.getStats().getShares());

        // Tính điểm engagement với trọng số khác nhau
        double score = (likes * 3.0) + (comments * 2.5) + (shares * 4.0) + (views * 0.1);

        // Điều chỉnh theo thời gian (bài viết mới hơn có điểm cao hơn)
        long hoursOld = java.time.Duration.between(post.getCreatedAt(), LocalDateTime.now()).toHours();
        double recencyFactor = Math.max(0.1, 1.0 - (hoursOld / (24.0 * 7.0))); // Giảm dần trong 7 ngày

        return score * recencyFactor;
    }

    @SuppressWarnings("unchecked")
    private List<PostResponse> getCachedRecommendations(String cacheKey) {
        try {
            return (List<PostResponse>) redisTemplate.opsForValue().get(cacheKey);
        } catch (Exception e) {
            log.warn("Error getting cached recommendations: {}", e.getMessage());
            return null;
        }
    }

    private void cacheRecommendations(String cacheKey, List<PostResponse> recommendations) {
        try {
            redisTemplate.opsForValue().set(cacheKey, recommendations, CACHE_TTL_HOURS, TimeUnit.HOURS);
        } catch (Exception e) {
            log.warn("Error caching recommendations: {}", e.getMessage());
        }
    }

    private PostResponse convertToResponse(PostEntity post) {
        return PostResponse.builder()
                .id(post.getId())
                .title(post.getTitle())
                .content(post.getContent())
                .author(post.getAuthor())
                .images(post.getImages())
                .videos(post.getVideos())
                .tags(post.getTags())
                .category(post.getCategory())
                .privacy(post.getPrivacy())
                .postType(post.getPostType() != null ? post.getPostType().name() : null)
                .location(post.getLocation())
                .stats(post.getStats())
                .engagement(post.getEngagement())
                .createdAt(post.getCreatedAt())
                .updatedAt(post.getUpdatedAt())
                .build();
    }

    // Inner class cho user profile
    @lombok.Data
    @lombok.Builder
    @lombok.NoArgsConstructor
    @lombok.AllArgsConstructor
    private static class UserInterestProfile implements java.io.Serializable {
        private String userId;
        private Set<String> interestTags;
        private Set<String> preferredCategories;
        private Set<String> friendIds;
        private String facultyId;
        private String majorId;
    }
}
