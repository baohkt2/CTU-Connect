import pytest
import asyncio
import json
from httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch
import numpy as np
from datetime import datetime

from main import app
from core.recommendation_engine import RecommendationEngine
from db.models import UserProfile, PostFeatures, UserInteraction
from config.settings import config

# Test client
client = TestClient(app)

@pytest.fixture
def mock_recommendation_engine():
    """Mock recommendation engine for testing"""
    engine = Mock(spec=RecommendationEngine)
    engine.is_initialized = True
    engine.get_recommendations = AsyncMock()
    engine.record_feedback = AsyncMock()
    return engine

@pytest.fixture
def sample_user_profile():
    """Sample user profile for testing"""
    return UserProfile(
        user_id="test_user_123",
        interests={"Khoa Hoc": 5, "Cong Nghe": 3},
        engagement_score=0.7,
        activity_pattern={"14": 0.8, "15": 0.6}
    )

@pytest.fixture
def sample_post_features():
    """Sample post features for testing"""
    return PostFeatures(
        post_id="test_post_456",
        title="Test Post Title",
        content="This is a test post content",
        author_id="author_123",
        category="Khoa Hoc",
        tags=["test", "science"],
        likes_count=10,
        comments_count=5,
        shares_count=2,
        views_count=100,
        engagement_rate=0.17
    )

@pytest.fixture
def sample_recommendations():
    """Sample recommendations response"""
    return {
        "recommendations": [
            {
                "post_id": "post_1",
                "title": "Test Post 1",
                "content": "Content 1",
                "author_id": "author_1",
                "category": "Khoa Hoc",
                "tags": ["science"],
                "engagement_rate": 0.15,
                "relevance_score": 0.95,
                "rank": 1,
                "reason": "Recommended because you're interested in Khoa Hoc"
            },
            {
                "post_id": "post_2",
                "title": "Test Post 2",
                "content": "Content 2",
                "author_id": "author_2",
                "category": "Cong Nghe",
                "tags": ["technology"],
                "engagement_rate": 0.12,
                "relevance_score": 0.87,
                "rank": 2,
                "reason": "Trending post with high engagement"
            }
        ],
        "ab_variant": "personalized_v1",
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": "test_user_123"
    }

class TestAPI:
    """Test API endpoints"""

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "services" in data

    @patch('main.recommendation_engine')
    def test_get_recommendations_success(self, mock_engine, sample_recommendations):
        """Test successful recommendation request"""
        mock_engine.is_initialized = True
        mock_engine.get_recommendations = AsyncMock(return_value=sample_recommendations)

        request_data = {
            "user_id": "test_user_123",
            "context": {"device_type": "mobile"},
            "k": 5
        }

        response = client.post(
            "/recommendations",
            json=request_data,
            headers={"Authorization": f"Bearer {config.SECRET_KEY}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "recommendations" in data
        assert "ab_variant" in data
        assert "user_id" in data
        assert data["user_id"] == "test_user_123"
        assert len(data["recommendations"]) <= 5

    @patch('main.recommendation_engine')
    def test_get_recommendations_engine_not_initialized(self, mock_engine):
        """Test recommendation request when engine not initialized"""
        mock_engine.is_initialized = False

        request_data = {
            "user_id": "test_user_123",
            "context": {}
        }

        response = client.post(
            "/recommendations",
            json=request_data,
            headers={"Authorization": f"Bearer {config.SECRET_KEY}"}
        )

        assert response.status_code == 503

    def test_get_recommendations_unauthorized(self):
        """Test recommendation request without authorization"""
        request_data = {
            "user_id": "test_user_123",
            "context": {}
        }

        response = client.post("/recommendations", json=request_data)
        assert response.status_code == 403

    @patch('main.recommendation_engine')
    def test_record_feedback_success(self, mock_engine):
        """Test successful feedback recording"""
        mock_engine.is_initialized = True
        mock_engine.record_feedback = AsyncMock()

        request_data = {
            "user_id": "test_user_123",
            "post_id": "test_post_456",
            "feedback_type": "like",
            "context": {"device_type": "desktop"}
        }

        response = client.post(
            "/feedback",
            json=request_data,
            headers={"Authorization": f"Bearer {config.SECRET_KEY}"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "message" in data
        assert "timestamp" in data

    def test_record_feedback_invalid_type(self):
        """Test feedback recording with invalid feedback type"""
        request_data = {
            "user_id": "test_user_123",
            "post_id": "test_post_456",
            "feedback_type": "invalid_type",
            "context": {}
        }

        response = client.post(
            "/feedback",
            json=request_data,
            headers={"Authorization": f"Bearer {config.SECRET_KEY}"}
        )

        # Should still accept but with different processing
        assert response.status_code == 200

class TestRecommendationEngine:
    """Test recommendation engine logic"""

    @pytest.mark.asyncio
    async def test_initialize_engine(self):
        """Test engine initialization"""
        engine = RecommendationEngine()

        # Mock dependencies
        with patch('core.recommendation_engine.aioredis.from_url') as mock_redis, \
             patch.object(engine, 'load_model') as mock_load_model:

            mock_redis.return_value = AsyncMock()
            mock_load_model.return_value = None

            await engine.initialize()

            assert engine.is_initialized is True
            assert engine.redis_client is not None

    @pytest.mark.asyncio
    async def test_get_user_profile(self, sample_user_profile):
        """Test user profile retrieval"""
        engine = RecommendationEngine()

        # Mock database session
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_user_profile
        mock_session.execute.return_value = mock_result

        profile = await engine.get_user_profile("test_user_123", mock_session)

        assert profile is not None
        assert profile.user_id == "test_user_123"
        assert profile.engagement_score == 0.7

    @pytest.mark.asyncio
    async def test_prepare_user_features(self, sample_user_profile):
        """Test user feature preparation"""
        engine = RecommendationEngine()
        engine.data_processor = Mock()
        engine.data_processor.extract_user_context.return_value = np.zeros(32)

        context = {"device_type": "mobile"}
        features = engine.prepare_user_features(sample_user_profile, context)

        assert "profile_features" in features
        assert "context_features" in features
        assert len(features["profile_features"]) == 64
        assert len(features["context_features"]) == 32

    def test_rank_posts(self, sample_post_features):
        """Test post ranking"""
        engine = RecommendationEngine()

        posts = [sample_post_features] * 3
        scores = np.array([0.9, 0.7, 0.8])

        ranked = engine.rank_posts(posts, scores)

        assert len(ranked) == 3
        assert ranked[0][1] == 0.9  # Highest score first
        assert ranked[1][1] == 0.8
        assert ranked[2][1] == 0.7

    def test_apply_diversity_filter(self, sample_post_features):
        """Test diversity filtering"""
        engine = RecommendationEngine()

        # Create posts with different categories
        posts = []
        for i, category in enumerate(["Khoa Hoc", "Cong Nghe", "Van Hoc", "Khoa Hoc"]):
            post = PostFeatures(
                post_id=f"post_{i}",
                title=f"Title {i}",
                content=f"Content {i}",
                author_id=f"author_{i}",
                category=category,
                engagement_rate=0.1
            )
            posts.append(post)

        scores = [0.9, 0.8, 0.7, 0.6]
        ranked_posts = list(zip(posts, scores))

        diverse_posts = engine.apply_diversity_filter(ranked_posts, k=3)

        assert len(diverse_posts) <= 3
        # Should prefer diverse categories
        categories = [post[0].category for post in diverse_posts]
        assert len(set(categories)) >= 2

class TestDataProcessor:
    """Test data processing functions"""

    def test_extract_post_features(self):
        """Test post feature extraction"""
        from data.processor import DataProcessor

        processor = DataProcessor()

        post_data = {
            "id": "test_post_123",
            "title": "Test Title",
            "content": "Test content",
            "authorId": "author_123",
            "category": "Khoa Hoc",
            "tags": ["science", "test"],
            "stats": {
                "likes": 10,
                "comments": 5,
                "shares": 2,
                "views": 100
            },
            "images": ["image1.jpg"],
            "videos": [],
            "postType": "TEXT"
        }

        features = processor.extract_post_features(post_data)

        assert features["post_id"] == "test_post_123"
        assert features["category"] == "Khoa Hoc"
        assert features["likes_count"] == 10
        assert features["has_images"] is True
        assert features["has_videos"] is False
        assert features["engagement_rate"] == 0.17  # (10+5+2)/100

    def test_extract_user_context(self):
        """Test user context extraction"""
        from data.processor import DataProcessor

        processor = DataProcessor()

        user_data = {
            "engagement_score": 0.8,
            "interests": {"Khoa Hoc": 5, "Cong Nghe": 3},
            "activity_pattern": {"14": 0.9}
        }

        request_context = {
            "device_type": "mobile"
        }

        context_vector = processor.extract_user_context(user_data, request_context)

        assert len(context_vector) == 32
        assert context_vector[3] == 0.8  # engagement_score
        assert context_vector[4] == 2    # number of interests
        assert context_vector[5] == 1.0  # mobile device

class TestModelComponents:
    """Test deep learning model components"""

    def test_user_embedding_forward(self):
        """Test user embedding forward pass"""
        from core.models import UserEmbedding
        import torch

        model = UserEmbedding(num_users=1000, embedding_dim=256)

        user_ids = torch.tensor([1, 2, 3])
        profile_features = torch.randn(3, 64)

        output = model(user_ids, profile_features)

        assert output.shape == (3, 256)
        assert torch.all(torch.abs(output) <= 1)  # tanh output

    def test_post_embedding_forward(self):
        """Test post embedding forward pass"""
        from core.models import PostEmbedding
        import torch

        model = PostEmbedding(num_posts=5000, embedding_dim=256)

        post_ids = torch.tensor([1, 2, 3])
        content_features = torch.randn(3, 768)
        categories = torch.tensor([0, 1, 2])

        output = model(post_ids, content_features, categories)

        assert output.shape == (3, 256)
        assert torch.all(torch.abs(output) <= 1)  # tanh output

    def test_attention_mechanism(self):
        """Test multi-head attention"""
        from core.models import MultiHeadAttention
        import torch

        attention = MultiHeadAttention(embedding_dim=256, num_heads=8)

        user_emb = torch.randn(3, 256)
        post_emb = torch.randn(3, 256)

        output = attention(user_emb, post_emb)

        assert output.shape == (3, 256)

# Load testing
class TestPerformance:
    """Performance and load testing"""

    @pytest.mark.asyncio
    async def test_concurrent_recommendations(self):
        """Test concurrent recommendation requests"""
        import asyncio
        import time

        async def make_request():
            async with AsyncClient(app=app, base_url="http://test") as ac:
                response = await ac.post(
                    "/recommendations",
                    json={
                        "user_id": f"user_{np.random.randint(1000)}",
                        "context": {"device_type": "mobile"}
                    },
                    headers={"Authorization": f"Bearer {config.SECRET_KEY}"}
                )
                return response.status_code

        # Skip if recommendation engine not available
        if not hasattr(app.state, 'recommendation_engine'):
            pytest.skip("Recommendation engine not available")

        start_time = time.time()

        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        duration = end_time - start_time

        # Most requests should succeed or return 503 (service unavailable)
        success_count = sum(1 for r in results if isinstance(r, int) and r in [200, 503])
        assert success_count >= 8  # Allow some failures

        # Should complete within reasonable time
        assert duration < 10.0  # 10 seconds max for 10 requests

# Integration tests
class TestIntegration:
    """Integration tests"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_recommendation_flow(self, sample_user_profile, sample_post_features):
        """Test full recommendation flow"""
        # This would require a test database and full setup
        # Skip for unit tests
        pytest.skip("Integration test - requires full setup")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_feedback_processing(self):
        """Test feedback processing integration"""
        # This would test the full feedback pipeline
        pytest.skip("Integration test - requires full setup")

# Fixtures for database testing
@pytest.fixture
async def test_db_session():
    """Test database session"""
    # This would set up a test database
    # For now, return a mock
    return AsyncMock()

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
