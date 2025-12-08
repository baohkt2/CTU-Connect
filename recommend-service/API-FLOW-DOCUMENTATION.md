# API Flow Documentation - Recommendation Service

## Architecture Overview

```
Client Frontend → API Gateway → Services (Post-Service / Recommendation-Service)
                                          ↓
                                  Recommendation-Service (Java API)
                                          ↓
                                  Python ML Model Service
```

## Complete Flow for Post Recommendations

### 1. Client → API Gateway → Recommendation Service

#### Option A: Get AI-Powered Recommendations
```
Client: GET /api/recommend/posts?userId={userId}&page=0&size=20
   ↓
API Gateway: Routes to recommendation-service
   ↓
Recommendation Service: /api/recommend/posts
   ↓
Returns: AI-powered personalized recommendations
```

**Frontend Code:**
```typescript
// Using postService.ts
const recommendations = await postService.getRecommendedPosts(userId, 0, 20);
```

**API Endpoint:** `GET http://localhost:8090/api/recommend/posts`
**Query Parameters:**
- `userId` (required): User ID to get recommendations for
- `page` (optional, default: 0): Page number
- `size` (optional, default: 20): Items per page
- `includeExplanations` (optional, default: false): Include explanation for each recommendation

**Response:**
```json
{
  "userId": "user123",
  "recommendations": [
    {
      "postId": "post1",
      "score": 0.95,
      "reason": "Based on your interests in AI",
      "post": { /* post details */ }
    }
  ],
  "algorithm": "HYBRID_ML",
  "generatedAt": "2024-12-08T14:00:00",
  "total": 20
}
```

#### Option B: Get Personalized Feed
```
Client: GET /api/recommendation/feed?userId={userId}&page=0&size=20
   ↓
API Gateway: Routes to recommendation-service
   ↓
Recommendation Service: /api/recommendation/feed
   ↓
Returns: Hybrid feed (ML + Graph + Collaborative filtering)
```

**Frontend Code:**
```typescript
const feed = await postService.getPersonalizedFeed(userId, 0, 20);
```

**API Endpoint:** `GET http://localhost:8090/api/recommendation/feed`

### 2. Recording User Interactions (for ML Training)

When a user interacts with a post, record it to improve recommendations:

```
Client: POST /api/recommendation/interaction
Body: {
  "userId": "user123",
  "postId": "post456",
  "type": "VIEW|LIKE|COMMENT|SHARE",
  "viewDuration": 30.5
}
   ↓
API Gateway → Recommendation Service
   ↓
Records interaction → Sends to Kafka → Python ML Model updates
```

**Frontend Code:**
```typescript
// When user views a post
await postService.recordRecommendationInteraction(userId, postId, 'VIEW', 30.5);

// When user likes a post
await postService.recordRecommendationInteraction(userId, postId, 'LIKE');
```

### 3. Sending Feedback

```
Client: POST /api/recommend/feedback
Body: {
  "userId": "user123",
  "postId": "post456",
  "feedbackType": "POSITIVE|NEGATIVE|NEUTRAL"
}
```

**Frontend Code:**
```typescript
await postService.sendRecommendationFeedback(userId, postId, 'POSITIVE');
```

## API Gateway Routing Configuration

The API Gateway routes recommendation requests as follows:

```yaml
/api/recommend/**           → recommendation-service  # New ML-based endpoints
/api/recommendation/**      → recommendation-service  # Feed and interaction endpoints
/api/recommendations/**     → post-service           # Legacy simple recommendations
/api/posts/feed            → post-service           # Traditional news feed
/api/feed/**               → recommendation-service  # AI-powered feed
```

## Service Endpoints

### Recommendation Service (Java API)

**Base URL (Docker):** `http://recommendation-service:8095`
**Base URL (Local Dev):** `http://localhost:8095`

#### Endpoints:

1. **GET /api/recommend/posts** - Get AI recommendations
2. **POST /api/recommend/posts** - Get recommendations with advanced options
3. **POST /api/recommend/feedback** - Record user feedback
4. **GET /api/recommendation/feed** - Get personalized feed
5. **POST /api/recommendation/interaction** - Record interaction
6. **DELETE /api/recommend/cache/{userId}** - Invalidate cache
7. **GET /api/recommend/health** - Health check

### Post Service (Legacy Recommendations)

**Base URL:** `http://post-service:8084`

#### Endpoints:

1. **GET /api/recommendations/personalized/{userId}** - Simple personalized recommendations
2. **GET /api/recommendations/trending** - Trending posts
3. **GET /api/recommendations/similar/{postId}** - Similar posts
4. **GET /api/posts/feed** - Traditional news feed

## Integration Example

### Homepage Feed Component

```typescript
import { useEffect, useState } from 'react';
import { postService } from '@/services/postService';
import { useAuth } from '@/contexts/AuthContext';

export const FeedPage = () => {
  const { user } = useAuth();
  const [posts, setPosts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadFeed = async () => {
      try {
        setLoading(true);
        
        // Try AI-powered recommendations first
        const recommendations = await postService.getPersonalizedFeed(
          user.id, 
          0, 
          20
        );
        
        setPosts(recommendations);
      } catch (error) {
        console.error('Failed to load feed:', error);
        
        // Fallback to traditional feed
        const fallbackPosts = await postService.getPosts(0, 20);
        setPosts(fallbackPosts.content);
      } finally {
        setLoading(false);
      }
    };

    if (user?.id) {
      loadFeed();
    }
  }, [user?.id]);

  // Record view interaction when user views a post
  const handlePostView = async (postId: string) => {
    if (user?.id) {
      await postService.recordRecommendationInteraction(
        user.id,
        postId,
        'VIEW',
        Date.now() // Track view start time
      );
    }
  };

  return (
    <div>
      {loading ? (
        <div>Loading recommendations...</div>
      ) : (
        posts.map(post => (
          <PostCard 
            key={post.id} 
            post={post} 
            onView={() => handlePostView(post.id)}
          />
        ))
      )}
    </div>
  );
};
```

## Debugging

### Check Service Registration

```bash
# Check if recommendation-service is registered with Eureka
curl http://localhost:8761/eureka/apps/RECOMMENDATION-SERVICE
```

### Test Endpoints Directly

```bash
# Test recommendation service directly (without gateway)
curl "http://localhost:8095/api/recommend/posts?userId=test-user&page=0&size=5"

# Test through API Gateway
curl "http://localhost:8090/api/recommend/posts?userId=test-user&page=0&size=5" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### View Logs

The recommendation service logs all API calls with detailed information:

```
========================================
📥 API REQUEST: GET /api/recommend/posts
   User ID: user123
   Page: 0, Size: 20
   Include Explanations: false
========================================
🔄 Processing recommendation request for user: user123
========================================
📤 API RESPONSE: GET /api/recommend/posts
   Total Recommendations: 20
   User ID: user123
   Algorithm: HYBRID_ML
   Generated At: 2024-12-08T14:00:00
========================================
```

### Common Issues

1. **Service not registered with Eureka**
   - Check `eureka.client.enabled=true` in `application-dev.yml`
   - Verify Eureka server is running on port 8761

2. **404 Not Found**
   - Verify API Gateway routing in `RouteConfig.java`
   - Check service name in Eureka matches routing configuration

3. **No recommendations returned**
   - Check Neo4j connection for graph data
   - Verify PostgreSQL has post data
   - Check Python model service is running (optional)

## Testing Checklist

- [ ] Eureka server running on port 8761
- [ ] Recommendation service registered with Eureka
- [ ] API Gateway routes configured correctly
- [ ] Client calls correct endpoint `/api/recommend/posts` or `/api/recommendation/feed`
- [ ] JWT token included in request headers
- [ ] User ID is valid and exists in system
- [ ] Neo4j and PostgreSQL databases accessible
- [ ] Redis cache accessible (port 6380 in dev)

## Performance Considerations

- **Caching**: Recommendations are cached in Redis for 2 hours
- **Batch Processing**: Embeddings rebuilt every 5 minutes
- **Fallback**: Always have a fallback to simple recommendations if ML service fails
- **Async**: Record interactions asynchronously to not block user experience

## Next Steps

1. Implement the feed component in client-frontend
2. Add recommendation tracking for analytics
3. Test the complete flow from client to ML model
4. Monitor recommendation quality and user engagement
5. A/B test different recommendation strategies
