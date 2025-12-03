import asyncio
import random
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import aioredis

from core.models import PersonalizedRecommendationModel, ReinforcementLearningAgent
from data.processor import DataProcessor, FeatureStore
from db.models import UserProfile, PostFeatures, UserInteraction, RecommendationLog, ABTestExperiment
from config.settings import config

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """Core recommendation engine with multiple algorithms"""

    def __init__(self):
        self.model = None
        self.rl_agent = None
        self.feature_store = None
        self.data_processor = None
        self.redis_client = None
        self.is_initialized = False

    async def initialize(self):
        """Initialize the recommendation engine"""
        try:
            # Initialize Redis client
            self.redis_client = aioredis.from_url(config.REDIS_URL)

            # Initialize feature store and data processor
            self.feature_store = FeatureStore(self.redis_client)
            self.data_processor = DataProcessor()
            await self.data_processor.initialize()

            # Load pre-trained model if exists
            await self.load_model()

            self.is_initialized = True
            logger.info("Recommendation engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize recommendation engine: {e}")
            raise

    async def load_model(self):
        """Load the trained recommendation model"""
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Initialize model with default parameters
            # These would be loaded from a configuration or model registry
            num_users = 10000  # This should be dynamic based on actual user count
            num_posts = 50000  # This should be dynamic based on actual post count

            self.model = PersonalizedRecommendationModel(
                num_users=num_users,
                num_posts=num_posts,
                embedding_dim=config.EMBEDDING_DIM,
                num_heads=config.NUM_HEADS
            ).to(device)

            # Load pre-trained weights if available
            try:
                checkpoint = torch.load(config.MODEL_PATH, map_location=device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded pre-trained model from {config.MODEL_PATH}")
            except FileNotFoundError:
                logger.warning("No pre-trained model found, using randomly initialized model")

            # Initialize RL agent
            state_dim = config.EMBEDDING_DIM + 32  # User embedding + context features
            action_dim = config.TOP_K_RECOMMENDATIONS

            self.rl_agent = ReinforcementLearningAgent(
                state_dim=state_dim,
                action_dim=action_dim
            ).to(device)

            self.model.eval()

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    async def get_recommendations(self, user_id: str, context: Dict[str, Any],
                                db_session: AsyncSession, k: int = None) -> Dict[str, Any]:
        """Get personalized recommendations for a user"""
        if not self.is_initialized:
            raise RuntimeError("Recommendation engine not initialized")

        k = k or config.TOP_K_RECOMMENDATIONS

        # Check cache first
        cache_key = f"recommendations:{user_id}:{hash(str(sorted(context.items())))}"
        cached_result = await self.redis_client.get(cache_key)

        if cached_result:
            import json
            return json.loads(cached_result)

        # Get A/B test variant
        ab_variant = await self.get_ab_test_variant(user_id, db_session)

        # Generate recommendations based on variant
        if ab_variant == "popularity_based":
            recommendations = await self.get_popularity_based_recommendations(
                user_id, context, db_session, k
            )
        else:
            recommendations = await self.get_personalized_recommendations(
                user_id, context, db_session, k, ab_variant
            )

        # Add A/B test info
        result = {
            "recommendations": recommendations,
            "ab_variant": ab_variant,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id
        }

        # Cache result
        await self.redis_client.setex(
            cache_key,
            config.REDIS_TTL,
            json.dumps(result, default=str)
        )

        # Log recommendation
        await self.log_recommendation(user_id, recommendations, ab_variant, context, db_session)

        return result

    async def get_personalized_recommendations(self, user_id: str, context: Dict[str, Any],
                                             db_session: AsyncSession, k: int,
                                             variant: str) -> List[Dict[str, Any]]:
        """Generate personalized recommendations using deep learning model"""
        try:
            # Get user profile and features
            user_profile = await self.get_user_profile(user_id, db_session)
            if not user_profile:
                # Fallback to popularity-based for new users
                return await self.get_popularity_based_recommendations(
                    user_id, context, db_session, k
                )

            # Get candidate posts (exclude recently interacted posts)
            candidate_posts = await self.get_candidate_posts(user_id, db_session, k * 5)

            if not candidate_posts:
                return []

            # Prepare model inputs
            user_features = self.prepare_user_features(user_profile, context)
            post_features_list = []

            for post in candidate_posts:
                post_features = await self.prepare_post_features(post, db_session)
                post_features_list.append(post_features)

            # Get model predictions
            scores = await self.predict_scores(user_features, post_features_list)

            # Apply reinforcement learning adjustment if available
            if self.rl_agent and variant == "personalized_v2":
                scores = await self.apply_rl_adjustment(user_id, candidate_posts, scores, context)

            # Rank and select top-k
            ranked_posts = self.rank_posts(candidate_posts, scores)

            # Apply diversity constraints
            diverse_posts = self.apply_diversity_filter(ranked_posts, k)

            # Format results
            recommendations = []
            for i, (post, score) in enumerate(diverse_posts[:k]):
                recommendations.append({
                    "post_id": post.post_id,
                    "title": post.title,
                    "content": post.content[:200] + "..." if len(post.content) > 200 else post.content,
                    "author_id": post.author_id,
                    "category": post.category,
                    "tags": post.tags,
                    "engagement_rate": post.engagement_rate,
                    "relevance_score": float(score),
                    "rank": i + 1,
                    "reason": self.generate_explanation(user_profile, post)
                })

            return recommendations

        except Exception as e:
            logger.error(f"Error generating personalized recommendations: {e}")
            # Fallback to popularity-based
            return await self.get_popularity_based_recommendations(
                user_id, context, db_session, k
            )

    async def get_popularity_based_recommendations(self, user_id: str, context: Dict[str, Any],
                                                 db_session: AsyncSession, k: int) -> List[Dict[str, Any]]:
        """Generate popularity-based recommendations as fallback"""
        try:
            # Get top posts by engagement in the last 7 days
            week_ago = datetime.utcnow() - timedelta(days=7)

            result = await db_session.execute(
                select(PostFeatures)
                .where(PostFeatures.updated_at >= week_ago)
                .order_by(PostFeatures.engagement_rate.desc())
                .limit(k * 2)
            )
            posts = result.scalars().all()

            # Filter out posts user has already interacted with
            user_interactions = await db_session.execute(
                select(UserInteraction.post_id)
                .where(UserInteraction.user_id == user_id)
                .where(UserInteraction.timestamp >= week_ago)
            )
            interacted_post_ids = {row[0] for row in user_interactions.fetchall()}

            filtered_posts = [post for post in posts if post.post_id not in interacted_post_ids]

            recommendations = []
            for i, post in enumerate(filtered_posts[:k]):
                recommendations.append({
                    "post_id": post.post_id,
                    "title": post.title,
                    "content": post.content[:200] + "..." if len(post.content) > 200 else post.content,
                    "author_id": post.author_id,
                    "category": post.category,
                    "tags": post.tags,
                    "engagement_rate": post.engagement_rate,
                    "relevance_score": post.engagement_rate,
                    "rank": i + 1,
                    "reason": "Trending post"
                })

            return recommendations

        except Exception as e:
            logger.error(f"Error generating popularity-based recommendations: {e}")
            return []

    async def get_user_profile(self, user_id: str, db_session: AsyncSession) -> Optional[UserProfile]:
        """Get user profile from database"""
        result = await db_session.execute(
            select(UserProfile).where(UserProfile.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_candidate_posts(self, user_id: str, db_session: AsyncSession,
                                limit: int = 100) -> List[PostFeatures]:
        """Get candidate posts for recommendation"""
        # Get posts from last 30 days, excluding user's own posts and already interacted posts
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)

        # Get user's recent interactions
        user_interactions = await db_session.execute(
            select(UserInteraction.post_id)
            .where(UserInteraction.user_id == user_id)
            .where(UserInteraction.timestamp >= thirty_days_ago)
        )
        interacted_post_ids = {row[0] for row in user_interactions.fetchall()}

        # Get candidate posts
        query = select(PostFeatures).where(
            PostFeatures.updated_at >= thirty_days_ago,
            PostFeatures.author_id != user_id
        )

        if interacted_post_ids:
            query = query.where(~PostFeatures.post_id.in_(interacted_post_ids))

        result = await db_session.execute(
            query.order_by(PostFeatures.updated_at.desc()).limit(limit)
        )

        return result.scalars().all()

    def prepare_user_features(self, user_profile: UserProfile, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare user features for model input"""
        # Extract user profile features
        interests = user_profile.interests or {}
        engagement_score = user_profile.engagement_score or 0.0
        activity_pattern = user_profile.activity_pattern or {}

        # Create user profile vector
        profile_features = np.zeros(64)  # 64-dimensional profile vector

        # Encode interests (top categories)
        top_interests = sorted(interests.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (category, score) in enumerate(top_interests):
            if i < 10:
                profile_features[i] = min(score / 10.0, 1.0)  # Normalize

        # Add engagement score
        profile_features[10] = min(engagement_score, 1.0)

        # Add activity pattern features
        current_hour = datetime.now().hour
        profile_features[11] = activity_pattern.get(str(current_hour), 0.0)

        # Context features
        context_features = self.data_processor.extract_user_context(
            user_profile.__dict__, context
        )

        return {
            'profile_features': profile_features,
            'context_features': context_features
        }

    async def prepare_post_features(self, post: PostFeatures, db_session: AsyncSession) -> Dict[str, Any]:
        """Prepare post features for model input"""
        # Content embedding (would be pre-computed and stored)
        content_embedding = post.content_embedding or np.zeros(768).tolist()

        # Category encoding
        categories = [
            "Van Hoc", "Khoa Hoc", "Cong Nghe", "The Thao", "Giai Tri",
            "Kinh Te", "Chinh Tri", "Xa Hoi", "Giao Duc", "Y Te"
        ]
        category_id = categories.index(post.category) if post.category in categories else 0

        return {
            'content_embedding': np.array(content_embedding),
            'category_id': category_id,
            'engagement_metrics': np.array([
                post.likes_count / 100.0,  # Normalize
                post.comments_count / 50.0,
                post.shares_count / 20.0,
                post.views_count / 1000.0,
                post.engagement_rate
            ])
        }

    async def predict_scores(self, user_features: Dict[str, Any],
                           post_features_list: List[Dict[str, Any]]) -> np.ndarray:
        """Predict relevance scores using the deep learning model"""
        if not self.model:
            # Return random scores if model not available
            return np.random.random(len(post_features_list))

        try:
            self.model.eval()
            with torch.no_grad():
                batch_size = len(post_features_list)

                # Prepare batch tensors (simplified for this example)
                user_ids = torch.zeros(batch_size, dtype=torch.long)  # Would be actual user ID mapping
                post_ids = torch.zeros(batch_size, dtype=torch.long)  # Would be actual post ID mapping

                user_profiles = torch.tensor([user_features['profile_features']] * batch_size).float()
                context_features = torch.tensor([user_features['context_features']] * batch_size).float()

                post_contents = torch.stack([
                    torch.tensor(pf['content_embedding']).float()
                    for pf in post_features_list
                ])

                post_categories = torch.tensor([
                    pf['category_id'] for pf in post_features_list
                ], dtype=torch.long)

                # Get predictions
                scores = self.model(
                    user_ids, post_ids, user_profiles,
                    post_contents, post_categories, context_features
                )

                return scores.cpu().numpy()

        except Exception as e:
            logger.error(f"Error in model prediction: {e}")
            return np.random.random(len(post_features_list))

    def rank_posts(self, posts: List[PostFeatures], scores: np.ndarray) -> List[Tuple[PostFeatures, float]]:
        """Rank posts by predicted scores"""
        ranked = list(zip(posts, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def apply_diversity_filter(self, ranked_posts: List[Tuple[PostFeatures, float]],
                             k: int) -> List[Tuple[PostFeatures, float]]:
        """Apply diversity constraints to recommendations"""
        if len(ranked_posts) <= k:
            return ranked_posts

        diverse_posts = []
        seen_categories = set()
        seen_authors = set()

        # First pass: ensure category diversity
        for post, score in ranked_posts:
            if len(diverse_posts) >= k:
                break

            category = post.category or "Unknown"
            author = post.author_id

            # Add if we haven't seen this category or if we have room
            if category not in seen_categories or len(diverse_posts) < k // 2:
                diverse_posts.append((post, score))
                seen_categories.add(category)
                seen_authors.add(author)

        # Second pass: fill remaining slots with high-scoring posts
        for post, score in ranked_posts:
            if len(diverse_posts) >= k:
                break

            if (post, score) not in diverse_posts:
                diverse_posts.append((post, score))

        return diverse_posts[:k]

    def generate_explanation(self, user_profile: UserProfile, post: PostFeatures) -> str:
        """Generate explanation for why this post was recommended"""
        interests = user_profile.interests or {}

        if post.category in interests:
            return f"Recommended because you're interested in {post.category}"
        elif post.engagement_rate > 0.1:
            return "Trending post with high engagement"
        else:
            return "Recommended based on your activity pattern"

    async def get_ab_test_variant(self, user_id: str, db_session: AsyncSession) -> str:
        """Get A/B test variant for user"""
        # Simple hash-based assignment
        user_hash = hash(user_id) % 1000

        cumulative = 0
        for variant, percentage in config.AB_TEST_VARIANTS.items():
            cumulative += int(percentage * 1000)
            if user_hash < cumulative:
                return variant

        return "personalized_v1"  # Default

    async def log_recommendation(self, user_id: str, recommendations: List[Dict[str, Any]],
                               ab_variant: str, context: Dict[str, Any],
                               db_session: AsyncSession):
        """Log recommendation for analytics"""
        try:
            post_ids = [rec["post_id"] for rec in recommendations]

            log_entry = RecommendationLog(
                user_id=user_id,
                post_ids=post_ids,
                model_version="v1.0",
                ab_test_variant=ab_variant,
                request_context=context,
                served_count=len(post_ids)
            )

            db_session.add(log_entry)
            await db_session.commit()

        except Exception as e:
            logger.error(f"Error logging recommendation: {e}")

    async def record_feedback(self, user_id: str, post_id: str, feedback_type: str,
                            context: Dict[str, Any], db_session: AsyncSession):
        """Record user feedback for reinforcement learning"""
        try:
            # Calculate reward based on feedback type
            reward_map = {
                'click': 0.2,
                'like': 0.5,
                'comment': 0.8,
                'share': 1.0,
                'skip': -0.1,
                'dislike': -0.5
            }

            reward = reward_map.get(feedback_type, 0.0)

            # Store interaction
            interaction = UserInteraction(
                user_id=user_id,
                post_id=post_id,
                interaction_type=feedback_type,
                context_data=context,
                reward=reward
            )

            db_session.add(interaction)
            await db_session.commit()

            # Update user profile
            await self.data_processor.update_user_profile(user_id, {
                'type': feedback_type,
                'postId': post_id,
                'timestamp': datetime.utcnow().isoformat(),
                **context
            }, db_session)

            # Invalidate cache
            await self.redis_client.delete(f"recommendations:{user_id}*")

            logger.info(f"Recorded feedback: {user_id} -> {post_id} ({feedback_type}, reward: {reward})")

        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            await db_session.rollback()
