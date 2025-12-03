import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, insert
import aioredis
from core.models import PersonalizedRecommendationModel
from db.models import UserInteraction, PostFeatures, UserProfile
from config.settings import config

logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles data ingestion, processing, and feature engineering"""

    def __init__(self):
        self.kafka_consumer = None
        self.kafka_producer = None
        self.redis_client = None

    async def initialize(self):
        """Initialize Kafka and Redis connections"""
        self.kafka_consumer = AIOKafkaConsumer(
            config.KAFKA_TOPIC_USER_INTERACTIONS,
            config.KAFKA_TOPIC_POST_UPDATES,
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            group_id=config.KAFKA_GROUP_ID,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )

        self.kafka_producer = AIOKafkaProducer(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )

        self.redis_client = aioredis.from_url(config.REDIS_URL)

        await self.kafka_consumer.start()
        await self.kafka_producer.start()

    async def process_user_interaction(self, interaction_data: Dict[str, Any],
                                     db_session: AsyncSession):
        """Process real-time user interaction data"""
        try:
            # Extract interaction features
            user_id = interaction_data.get('userId')
            post_id = interaction_data.get('postId')
            interaction_type = interaction_data.get('type')  # like, comment, share, view
            timestamp = datetime.fromisoformat(interaction_data.get('timestamp'))

            # Store in database
            interaction = UserInteraction(
                user_id=user_id,
                post_id=post_id,
                interaction_type=interaction_type,
                timestamp=timestamp,
                context_data=interaction_data.get('context', {})
            )

            db_session.add(interaction)
            await db_session.commit()

            # Update user profile in real-time
            await self.update_user_profile(user_id, interaction_data, db_session)

            # Invalidate cached recommendations
            cache_key = f"recommendations:{user_id}"
            await self.redis_client.delete(cache_key)

            logger.info(f"Processed interaction: {user_id} -> {post_id} ({interaction_type})")

        except Exception as e:
            logger.error(f"Error processing interaction: {e}")
            await db_session.rollback()

    async def update_user_profile(self, user_id: str, interaction_data: Dict[str, Any],
                                db_session: AsyncSession):
        """Update user profile based on interactions"""
        # Get current profile
        result = await db_session.execute(
            select(UserProfile).where(UserProfile.user_id == user_id)
        )
        profile = result.scalar_one_or_none()

        if not profile:
            # Create new profile
            profile = UserProfile(
                user_id=user_id,
                interests={},
                engagement_score=0.0,
                activity_pattern={}
            )
            db_session.add(profile)

        # Update interests based on post category
        post_category = interaction_data.get('postCategory')
        if post_category:
            current_interests = profile.interests or {}
            current_interests[post_category] = current_interests.get(post_category, 0) + 1
            profile.interests = current_interests

        # Update engagement score
        interaction_weights = {
            'view': 0.1,
            'like': 0.5,
            'comment': 0.8,
            'share': 1.0
        }

        weight = interaction_weights.get(interaction_data.get('type'), 0.1)
        profile.engagement_score = (profile.engagement_score * 0.9) + (weight * 0.1)

        await db_session.commit()

    def extract_post_features(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive features from post data"""
        features = {
            'post_id': post_data.get('id'),
            'title': post_data.get('title', ''),
            'content': post_data.get('content', ''),
            'author_id': post_data.get('authorId'),
            'category': post_data.get('category', ''),
            'tags': post_data.get('tags', []),
            'created_at': post_data.get('createdAt'),
            'stats': post_data.get('stats', {}),
            'has_images': len(post_data.get('images', [])) > 0,
            'has_videos': len(post_data.get('videos', [])) > 0,
            'post_type': post_data.get('postType', 'TEXT')
        }

        # Extract engagement metrics
        stats = post_data.get('stats', {})
        features.update({
            'likes_count': stats.get('likes', 0),
            'comments_count': stats.get('comments', 0),
            'shares_count': stats.get('shares', 0),
            'views_count': stats.get('views', 0)
        })

        # Calculate engagement rate
        total_interactions = sum([
            features['likes_count'],
            features['comments_count'],
            features['shares_count']
        ])
        features['engagement_rate'] = total_interactions / max(features['views_count'], 1)

        return features

    def extract_user_context(self, user_data: Dict[str, Any],
                           request_context: Dict[str, Any]) -> np.ndarray:
        """Extract contextual features for recommendations"""
        now = datetime.now()

        # Temporal features
        hour_of_day = now.hour / 24.0
        day_of_week = now.weekday() / 7.0

        # User activity pattern
        user_activity = user_data.get('activity_pattern', {})
        current_hour_activity = user_activity.get(str(now.hour), 0.0)

        # Device and location context
        device_type = request_context.get('device_type', 'desktop')
        device_features = {
            'mobile': [1.0, 0.0, 0.0],
            'tablet': [0.0, 1.0, 0.0],
            'desktop': [0.0, 0.0, 1.0]
        }
        device_vec = device_features.get(device_type, [0.0, 0.0, 1.0])

        # Combine all context features
        context_vector = np.array([
            hour_of_day,
            day_of_week,
            current_hour_activity,
            user_data.get('engagement_score', 0.0),
            len(user_data.get('interests', {})),
            *device_vec,
            # Padding to reach 32 dimensions
            *[0.0] * 23
        ])

        return context_vector[:32]  # Ensure exactly 32 dimensions

class FeatureStore:
    """Manages feature storage and retrieval"""

    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client

    async def store_user_features(self, user_id: str, features: Dict[str, Any]):
        """Store user features in Redis"""
        key = f"user_features:{user_id}"
        await self.redis.hset(key, mapping={
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in features.items()
        })
        await self.redis.expire(key, config.REDIS_TTL)

    async def get_user_features(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user features from Redis"""
        key = f"user_features:{user_id}"
        features = await self.redis.hgetall(key)

        if not features:
            return None

        # Deserialize JSON fields
        result = {}
        for k, v in features.items():
            k = k.decode('utf-8')
            v = v.decode('utf-8')
            try:
                result[k] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                result[k] = v

        return result

    async def store_post_features(self, post_id: str, features: Dict[str, Any]):
        """Store post features in Redis"""
        key = f"post_features:{post_id}"
        await self.redis.hset(key, mapping={
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in features.items()
        })
        await self.redis.expire(key, config.REDIS_TTL)

    async def get_post_features(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve post features from Redis"""
        key = f"post_features:{post_id}"
        features = await self.redis.hgetall(key)

        if not features:
            return None

        result = {}
        for k, v in features.items():
            k = k.decode('utf-8')
            v = v.decode('utf-8')
            try:
                result[k] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                result[k] = v

        return result

class BatchProcessor:
    """Handles batch processing for model training"""

    def __init__(self, db_session_factory):
        self.db_session_factory = db_session_factory

    async def prepare_training_data(self, start_date: datetime,
                                  end_date: datetime) -> Dict[str, Any]:
        """Prepare training data from user interactions"""
        async with self.db_session_factory() as session:
            # Get interactions in date range
            result = await session.execute(
                select(UserInteraction)
                .where(UserInteraction.timestamp.between(start_date, end_date))
                .order_by(UserInteraction.timestamp)
            )
            interactions = result.scalars().all()

            # Convert to training format
            training_data = {
                'user_ids': [],
                'post_ids': [],
                'labels': [],
                'timestamps': []
            }

            # Create positive and negative samples
            for interaction in interactions:
                training_data['user_ids'].append(interaction.user_id)
                training_data['post_ids'].append(interaction.post_id)

                # Label based on interaction type
                label_map = {
                    'view': 0.1,
                    'like': 0.7,
                    'comment': 0.9,
                    'share': 1.0
                }
                training_data['labels'].append(
                    label_map.get(interaction.interaction_type, 0.1)
                )
                training_data['timestamps'].append(interaction.timestamp)

            return training_data
