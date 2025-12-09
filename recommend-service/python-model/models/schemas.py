"""
Pydantic models for API request/response
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class UserAcademicProfile(BaseModel):
    """User academic profile"""
    userId: Optional[str] = None
    major: Optional[str] = None
    faculty: Optional[str] = None
    degree: Optional[str] = None
    batch: Optional[str] = None


class UserInteractionHistory(BaseModel):
    """User interaction history"""
    postId: str
    liked: int = 0
    commented: int = 0
    shared: int = 0
    viewDuration: float = 0.0
    timestamp: Optional[int] = None


class CandidatePost(BaseModel):
    """Candidate post for ranking"""
    postId: str
    content: str
    hashtags: List[str] = Field(default_factory=list)
    mediaDescription: Optional[str] = None
    authorId: Optional[str] = None
    authorMajor: Optional[str] = None
    authorFaculty: Optional[str] = None
    authorBatch: Optional[str] = None
    createdAt: Optional[str] = None
    # Support both naming conventions
    likeCount: int = 0
    likesCount: Optional[int] = None
    commentCount: int = 0
    commentsCount: Optional[int] = None
    shareCount: int = 0
    sharesCount: Optional[int] = None
    viewCount: int = 0
    
    def model_post_init(self, __context):
        """Normalize field names after initialization"""
        # Use whichever is provided
        if self.likesCount is not None:
            self.likeCount = self.likesCount
        if self.commentsCount is not None:
            self.commentCount = self.commentsCount
        if self.sharesCount is not None:
            self.shareCount = self.sharesCount


class PredictionRequest(BaseModel):
    """Request for prediction endpoint"""
    userAcademic: UserAcademicProfile
    userHistory: List[UserInteractionHistory] = Field(default_factory=list)
    candidatePosts: List[CandidatePost]
    topK: int = Field(default=20, ge=1, le=100)


class RankedPost(BaseModel):
    """Ranked post result"""
    postId: str
    score: float = Field(ge=0.0, le=1.0)
    contentSimilarity: Optional[float] = None
    implicitFeedback: Optional[float] = None
    academicScore: Optional[float] = None
    popularityScore: Optional[float] = None
    category: Optional[str] = None
    rank: Optional[int] = None


class PredictionResponse(BaseModel):
    """Response from prediction endpoint"""
    rankedPosts: List[RankedPost]
    modelVersion: str
    processingTimeMs: int
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None
