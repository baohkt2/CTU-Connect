# Recommendation Service Integration - Changes Summary

## 📋 Overview
Implemented complete integration flow for personalized feed using AI-powered recommendation service.

## 🔄 Data Flow
```
client-frontend (PostFeed.tsx)
    ↓ GET /api/posts/feed
post-service (PostController)
    ↓ GET /api/recommendations/feed
recommend-service (RecommendationController)
    ↓ Returns List<RecommendedPost> with scores
post-service (PostService)
    ↓ Fetches full post details + enriches with scores
    ↓ Returns List<PostResponse>
client-frontend
    ↓ Displays personalized feed
```

## 📝 Changes Made

### 1. **post-service** - New Feign Client for Recommendation Service

#### Created Files:
- `com.ctuconnect.client.RecommendationServiceClient.java`
  - Feign client to communicate with recommendation-service
  - Method: `getRecommendationFeed(userId, page, size)`
  - Uses Eureka service discovery

- `com.ctuconnect.client.RecommendationServiceClientFallback.java`
  - Fallback when recommendation-service is unavailable
  - Returns empty response to trigger fallback logic

- `com.ctuconnect.dto.response.RecommendationFeedResponse.java`
  - DTO mirroring recommend-service RecommendationResponse
  - Contains list of RecommendedPost with scores

#### Modified Files:
- `com.ctuconnect.controller.PostController.java`
  - Added `@Slf4j` for logging
  - Injected `RecommendationServiceClient`
  - **Enhanced `/api/posts/feed` endpoint**:
    1. Calls recommendation-service for AI recommendations
    2. Extracts postIds from recommendations
    3. Fetches full post details via `postService.getPostsByIds()`
    4. Maps recommendation scores to posts
    5. Returns ordered list maintaining recommendation order
    6. Falls back to NewsFeedService or regular posts if needed
  - Added comprehensive debug logging with visual markers (📥, 📤, ✅, ❌)

- `com.ctuconnect.service.PostService.java`
  - **Added `getPostsByIds(List<String> postIds, String currentUserId)` method**:
    - Fetches multiple posts by IDs
    - Maintains the order of input list (preserves recommendation order)
    - Records view interactions
    - Recalculates post stats
    - Returns enriched PostResponse list

### 2. **client-frontend** - Updated to Use Personalized Feed

#### Modified Files:
- `src/components/post/PostFeed.tsx`
  - Updated `loadPosts()` function for 'latest' tab
  - Now calls `postService.getPersonalizedFeed(page, size)` for main feed
  - Added console logging for debugging
  - Error handling with fallback to regular posts

- `src/services/postService.ts`
  - **Added `getPersonalizedFeed(page, size)` method**:
    - Calls `/api/posts/feed` endpoint
    - Returns array of Post objects
    - Handles errors with fallback to regular posts
    - Includes console logging for debugging

## 🔧 Configuration

### API Gateway Routes (Already Configured)
```yaml
/api/recommendations/** → recommendation-service
/api/recommend/** → recommendation-service  
/api/posts/** → post-service
```

### Feign Configuration (Already Configured)
- Connection timeout: 10 seconds
- Read timeout: 60 seconds
- Retry: 3 attempts with exponential backoff
- Logging: BASIC level
- Circuit breaker with fallback

## 🧪 Testing Flow

### 1. Test Recommendation Service Directly
```bash
# Get recommendations for a user
curl -X GET "http://localhost:8095/api/recommendations/feed?userId=USER_ID&page=0&size=20"
```

### 2. Test Post Service Feed Endpoint
```bash
# Get personalized feed (requires authentication)
curl -X GET "http://localhost:8090/api/posts/feed?page=0&size=10" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### 3. Test Frontend
```bash
# Navigate to http://localhost:3000
# Log in with a user account
# Check the "Mới nhất" (Latest) tab - should load personalized feed
# Check browser console for debug logs:
#   📥 Loading personalized feed from recommendation service...
#   📤 Received X posts from feed
```

## 📊 Debug Logging

### Post-Service Logs
```
========================================
📥 GET /api/posts/feed - User: userId, Page: 0, Size: 10
========================================
🔄 Calling recommendation-service for user: userId
📤 Received X recommendations from recommendation-service
📋 Fetching full details for X posts: [postId1, postId2, ...]
✅ Post postId1: content... (score: 0.95)
✅ Post postId2: content... (score: 0.87)
========================================
✅ Returning X personalized posts (XXXms)
========================================
```

### Client-Frontend Console
```
📥 Loading personalized feed from recommendation service...
📤 Received 10 posts from feed
```

## 🔄 Fallback Strategy

1. **Primary**: Recommendation-service AI recommendations
2. **Fallback 1**: NewsFeedService (if available)
3. **Fallback 2**: Regular posts ordered by createdAt DESC

## ✅ Benefits

1. **Personalized Content**: Users see posts tailored to their interests and academic profile
2. **AI-Powered**: Uses PhoBERT embeddings and ML ranking for Vietnamese content
3. **Graceful Degradation**: Falls back to simpler strategies if recommendation-service is down
4. **Performance**: Redis caching at recommendation layer (30-120s TTL)
5. **Scalability**: Service-oriented architecture with clear separation of concerns
6. **Maintainability**: Comprehensive logging for debugging
7. **User Experience**: Seamless integration - users don't notice backend complexity

## 🚀 Next Steps

1. **Monitor Performance**: Track response times and recommendation quality
2. **A/B Testing**: Compare engagement metrics between personalized and non-personalized feeds
3. **Feedback Loop**: Implement user feedback mechanism to improve recommendations
4. **Analytics**: Track which posts are recommended and user interactions
5. **Optimization**: Fine-tune recommendation algorithms based on user behavior

## 📚 Related Documentation

- `README-RECOMMENDATION-SERVICE.md` - Recommendation service setup and architecture
- `recommend-service/ARCHITECTURE.md` - Detailed recommendation system architecture
- `recommend-service/API-FLOW-DOCUMENTATION.md` - API flow documentation

## 🆘 Troubleshooting

### Issue: No posts returned
- Check if recommendation-service is running: `http://localhost:8095/actuator/health`
- Check if post-service can reach recommendation-service via Eureka
- Check logs for error messages

### Issue: Recommendation service unavailable
- System will automatically fall back to regular posts
- Check recommendation-service logs: `docker-compose logs recommendation-service`

### Issue: Posts returned but not personalized
- Check if user has interaction history in recommendation-service
- Verify user profile is available in user-service
- Check Python ML service is running: `docker-compose logs python-model`

---

**Date**: 2024-12-09  
**Status**: ✅ Implementation Complete - Ready for Testing  
**Version**: 1.0.0
