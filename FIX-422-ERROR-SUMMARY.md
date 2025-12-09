# Fix 422 Error - Data Type Mismatch Between Java and Python Services

## Problem Summary
The Python model service was returning 422 Unprocessable Entity error when called by the Java recommend-service. This was due to data type mismatches in the API contract between Java DTOs and Python Pydantic models.

## Root Causes Identified

### 1. Timestamp Field Type Mismatch
**Issue:** `UserInteractionHistory.timestamp`
- **Java side:** Was sending `LocalDateTime` object
- **Python side:** Expected `Optional[int]` (Unix timestamp in milliseconds)
- **Error:** FastAPI validation failed when receiving LocalDateTime format

### 2. CreatedAt Field Type Mismatch  
**Issue:** `CandidatePost.createdAt`
- **Java side:** Was sending `LocalDateTime` object  
- **Python side:** Expected `Optional[str]` (ISO format string)
- **Error:** FastAPI validation failed when receiving LocalDateTime format

## Solutions Implemented

### Java Side Changes

#### 1. Updated `UserInteractionHistory.java`
```java
// Changed from LocalDateTime to Long (Unix timestamp)
private Long timestamp;  // Unix timestamp in milliseconds

// Convert LocalDateTime to Unix timestamp
public void setTimestampFromDateTime(LocalDateTime dateTime) {
    if (dateTime != null) {
        this.timestamp = dateTime.atZone(ZoneId.systemDefault())
            .toInstant().toEpochMilli();
    }
}
```

#### 2. Updated `CandidatePost.java`
```java
// Changed from LocalDateTime to String (ISO format)
private String createdAt;  // ISO format string for Python

// Keep internal LocalDateTime for Java use
@JsonIgnore
private LocalDateTime createdAtDateTime;

// Convert LocalDateTime to ISO string
public void setCreatedAtFromDateTime(LocalDateTime dateTime) {
    if (dateTime != null) {
        this.createdAtDateTime = dateTime;
        this.createdAt = dateTime.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME);
    }
}
```

#### 3. Updated `HybridRecommendationService.java`
- Convert timestamps when building UserInteractionHistory:
  ```java
  Long timestamp = fb.getTimestamp() != null ? 
      fb.getTimestamp().atZone(ZoneId.systemDefault()).toInstant().toEpochMilli() : null;
  ```

- Convert createdAt when building CandidatePost:
  ```java
  if (post.getCreatedAt() != null) {
      candidate.setCreatedAtFromDateTime(post.getCreatedAt());
  }
  ```

- Use `getCreatedAtDateTime()` when converting back to RecommendedPost for response

### Python Side Changes

#### 1. Updated `models/schemas.py` - UserInteractionHistory
```python
from pydantic import BaseModel, Field, field_validator
from typing import Union

class UserInteractionHistory(BaseModel):
    timestamp: Optional[Union[int, str]] = None
    
    @field_validator('timestamp', mode='before')
    @classmethod
    def normalize_timestamp(cls, v):
        """Convert various timestamp formats to Unix timestamp"""
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            try:
                dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
                return int(dt.timestamp() * 1000)
            except:
                return None
        return None
```

#### 2. Updated `models/schemas.py` - CandidatePost
```python
class CandidatePost(BaseModel):
    createdAt: Optional[Union[str, int]] = None
    
    @field_validator('createdAt', mode='before')
    @classmethod
    def normalize_created_at(cls, v):
        """Accept both timestamp and ISO string format"""
        if v is None:
            return None
        if isinstance(v, (int, str)):
            return v
        return None
```

#### 3. Enhanced Debug Logging in `api/routes.py`
```python
# Log first history item if present
if request.userHistory:
    first_history = request.userHistory[0]
    logger.debug(f"   Sample history: postId={first_history.postId}, "
                f"liked={first_history.liked}, timestamp={first_history.timestamp}")
```

## Files Modified

### Java Service
1. `recommend-service/java-api/src/main/java/vn/ctu/edu/recommend/model/dto/UserInteractionHistory.java`
2. `recommend-service/java-api/src/main/java/vn/ctu/edu/recommend/model/dto/CandidatePost.java`
3. `recommend-service/java-api/src/main/java/vn/ctu/edu/recommend/service/HybridRecommendationService.java`

### Python Service
1. `recommend-service/python-model/models/schemas.py`
2. `recommend-service/python-model/api/routes.py`

## Testing Steps

1. **Build Java Service:**
   ```bash
   cd recommend-service/java-api
   mvn clean install -DskipTests
   ```

2. **Restart Services:**
   ```bash
   # Stop all services
   .\stop-all-services.ps1
   
   # Start all services
   .\start-all-services.ps1
   ```

3. **Test Feed Endpoint:**
   - Login as a user
   - Refresh the feed
   - Check logs for successful Python model predictions
   - Verify no 422 errors

4. **Test User Interactions:**
   - Like/comment/share posts
   - Verify interactions are recorded in user_feedback table
   - Check that recommendations update based on interactions

## Expected Behavior After Fix

### Success Indicators:
✅ No more 422 errors from Python service  
✅ Feed loads with personalized recommendations  
✅ User interactions are properly recorded  
✅ Scores are calculated correctly (not all 0.0)  
✅ Python logs show successful predictions with ranked posts

### Log Examples:
```
# Java Service
✅ Received 5 ranked posts from Python service

# Python Service  
INFO: Prediction request for user: xxx, candidates: 5
INFO: Prediction completed: 5 posts ranked in 150.23ms
INFO: 127.0.0.1 - "POST /api/model/predict HTTP/1.1" 200 OK
```

## Impact
- ✅ Fixes 422 validation errors
- ✅ Enables ML-based recommendation ranking
- ✅ Improves feed personalization
- ✅ Maintains backward compatibility with fallback ranking

## Date Fixed
December 9, 2025
