from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

Base = declarative_base()

class UserProfile(Base):
    """User profile with preferences and behavior patterns"""
    __tablename__ = "user_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, unique=True, nullable=False, index=True)
    interests = Column(JSON, default=dict)  # Category preferences
    engagement_score = Column(Float, default=0.0)
    activity_pattern = Column(JSON, default=dict)  # Hourly activity patterns
    demographics = Column(JSON, default=dict)  # Age, faculty, major, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    interactions = relationship("UserInteraction", back_populates="user_profile")

class PostFeatures(Base):
    """Post features for recommendation model"""
    __tablename__ = "post_features"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    post_id = Column(String, unique=True, nullable=False, index=True)
    title = Column(Text)
    content = Column(Text)
    author_id = Column(String, nullable=False)
    category = Column(String)
    tags = Column(JSON, default=list)
    post_type = Column(String)  # TEXT, IMAGE, VIDEO

    # Engagement metrics
    likes_count = Column(Integer, default=0)
    comments_count = Column(Integer, default=0)
    shares_count = Column(Integer, default=0)
    views_count = Column(Integer, default=0)
    engagement_rate = Column(Float, default=0.0)

    # Content features
    has_images = Column(Boolean, default=False)
    has_videos = Column(Boolean, default=False)
    content_length = Column(Integer, default=0)

    # Embeddings (stored as JSON arrays)
    content_embedding = Column(JSON)  # PhoBERT embeddings

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    interactions = relationship("UserInteraction", back_populates="post_features")

class UserInteraction(Base):
    """User-Post interaction records"""
    __tablename__ = "user_interactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False, index=True)
    post_id = Column(String, nullable=False, index=True)
    interaction_type = Column(String, nullable=False)  # view, like, comment, share

    # Context information
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    context_data = Column(JSON, default=dict)  # Device, location, etc.

    # Feedback for reinforcement learning
    reward = Column(Float, default=0.0)
    session_id = Column(String)

    # Relationships
    user_profile = relationship("UserProfile", back_populates="interactions")
    post_features = relationship("PostFeatures", back_populates="interactions")

class RecommendationLog(Base):
    """Log of recommendations served to users"""
    __tablename__ = "recommendation_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False, index=True)
    post_ids = Column(JSON, nullable=False)  # Array of recommended post IDs
    model_version = Column(String, nullable=False)
    ab_test_variant = Column(String)

    # Context
    request_context = Column(JSON, default=dict)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Performance metrics
    served_count = Column(Integer, default=0)
    clicked_count = Column(Integer, default=0)
    ctr = Column(Float, default=0.0)

class ModelMetrics(Base):
    """Model performance metrics"""
    __tablename__ = "model_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_version = Column(String, nullable=False)
    metric_name = Column(String, nullable=False)  # precision, recall, f1, diversity
    metric_value = Column(Float, nullable=False)

    # Evaluation context
    evaluation_date = Column(DateTime, default=datetime.utcnow)
    dataset_size = Column(Integer)
    evaluation_config = Column(JSON, default=dict)

class ABTestExperiment(Base):
    """A/B testing experiments"""
    __tablename__ = "ab_test_experiments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_name = Column(String, unique=True, nullable=False)
    variants = Column(JSON, nullable=False)  # Dict of variant_name: traffic_percentage

    # Experiment status
    is_active = Column(Boolean, default=True)
    start_date = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime)

    # Results
    results = Column(JSON, default=dict)  # Statistical results per variant
    winner = Column(String)  # Winning variant

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ReplayBuffer(Base):
    """Reinforcement learning replay buffer"""
    __tablename__ = "replay_buffer"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False, index=True)

    # RL components (state, action, reward, next_state)
    state = Column(JSON, nullable=False)  # User and context features
    action = Column(JSON, nullable=False)  # Recommended posts
    reward = Column(Float, nullable=False)
    next_state = Column(JSON)
    done = Column(Boolean, default=False)

    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

# Database utility functions
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from config.settings import config

# Create async engine
engine = create_async_engine(
    config.DATABASE_URL,
    echo=config.DEBUG,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

# Create session factory
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def get_db_session() -> AsyncSession:
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def create_tables():
    """Create all database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
