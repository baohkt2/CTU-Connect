import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timedelta
import mlflow
import mlflow.pytorch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio

from core.models import PersonalizedRecommendationModel, ReinforcementLearningAgent
from data.processor import DataProcessor, BatchProcessor
from db.models import UserInteraction, PostFeatures, UserProfile, ModelMetrics
from config.settings import config

logger = logging.getLogger(__name__)

class RecommendationDataset(Dataset):
    """Dataset for training recommendation model"""

    def __init__(self, interactions: List[Dict], user_features: Dict, post_features: Dict):
        self.interactions = interactions
        self.user_features = user_features
        self.post_features = post_features

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        interaction = self.interactions[idx]
        user_id = interaction['user_id']
        post_id = interaction['post_id']
        label = interaction['label']

        # Get features
        user_profile = self.user_features.get(user_id, np.zeros(64))
        post_content = self.post_features.get(post_id, {}).get('content_embedding', np.zeros(768))
        post_category = self.post_features.get(post_id, {}).get('category_id', 0)
        context = interaction.get('context', np.zeros(32))

        return {
            'user_id': torch.tensor(hash(user_id) % 10000, dtype=torch.long),
            'post_id': torch.tensor(hash(post_id) % 50000, dtype=torch.long),
            'user_profile': torch.tensor(user_profile, dtype=torch.float32),
            'post_content': torch.tensor(post_content, dtype=torch.float32),
            'post_category': torch.tensor(post_category, dtype=torch.long),
            'context': torch.tensor(context, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }

class ModelTrainer:
    """Handles model training and evaluation"""

    def __init__(self, db_session_factory):
        self.db_session_factory = db_session_factory
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None

    async def prepare_training_data(self, days_back: int = 30) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data"""
        try:
            # Get data from last N days
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)

            batch_processor = BatchProcessor(self.db_session_factory)
            training_data = await batch_processor.prepare_training_data(start_date, end_date)

            # Load user and post features
            user_features = await self.load_user_features()
            post_features = await self.load_post_features()

            # Create positive and negative samples
            interactions = []
            for i in range(len(training_data['user_ids'])):
                interactions.append({
                    'user_id': training_data['user_ids'][i],
                    'post_id': training_data['post_ids'][i],
                    'label': training_data['labels'][i],
                    'context': np.random.random(32)  # Simplified context
                })

            # Add negative samples
            negative_samples = await self.generate_negative_samples(interactions, user_features, post_features)
            interactions.extend(negative_samples)

            # Split train/validation
            split_idx = int(len(interactions) * 0.8)
            train_interactions = interactions[:split_idx]
            val_interactions = interactions[split_idx:]

            # Create datasets
            train_dataset = RecommendationDataset(train_interactions, user_features, post_features)
            val_dataset = RecommendationDataset(val_interactions, user_features, post_features)

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=2
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=2
            )

            logger.info(f"Prepared {len(train_interactions)} training and {len(val_interactions)} validation samples")

            return train_loader, val_loader

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise

    async def load_user_features(self) -> Dict[str, np.ndarray]:
        """Load user features from database"""
        async with self.db_session_factory() as session:
            from sqlalchemy import select
            result = await session.execute(select(UserProfile))
            profiles = result.scalars().all()

            user_features = {}
            for profile in profiles:
                # Create feature vector from profile
                interests = profile.interests or {}
                feature_vector = np.zeros(64)

                # Encode top interests
                top_interests = sorted(interests.items(), key=lambda x: x[1], reverse=True)[:10]
                for i, (category, score) in enumerate(top_interests):
                    if i < 10:
                        feature_vector[i] = min(score / 10.0, 1.0)

                feature_vector[10] = min(profile.engagement_score or 0.0, 1.0)
                user_features[profile.user_id] = feature_vector

            return user_features

    async def load_post_features(self) -> Dict[str, Dict]:
        """Load post features from database"""
        async with self.db_session_factory() as session:
            from sqlalchemy import select
            result = await session.execute(select(PostFeatures))
            posts = result.scalars().all()

            post_features = {}
            categories = ["Van Hoc", "Khoa Hoc", "Cong Nghe", "The Thao", "Giai Tri"]

            for post in posts:
                category_id = categories.index(post.category) if post.category in categories else 0

                post_features[post.post_id] = {
                    'content_embedding': post.content_embedding or np.zeros(768).tolist(),
                    'category_id': category_id,
                    'engagement_rate': post.engagement_rate or 0.0
                }

            return post_features

    async def generate_negative_samples(self, positive_interactions: List[Dict],
                                      user_features: Dict, post_features: Dict) -> List[Dict]:
        """Generate negative samples for training"""
        negative_samples = []
        user_ids = list(user_features.keys())
        post_ids = list(post_features.keys())

        # Generate negative samples (1:1 ratio with positive samples)
        for _ in range(len(positive_interactions)):
            user_id = np.random.choice(user_ids)
            post_id = np.random.choice(post_ids)

            # Ensure it's not a positive interaction
            is_positive = any(
                int['user_id'] == user_id and int['post_id'] == post_id
                for int in positive_interactions
            )

            if not is_positive:
                negative_samples.append({
                    'user_id': user_id,
                    'post_id': post_id,
                    'label': 0.0,
                    'context': np.random.random(32)
                })

        return negative_samples

    def initialize_model(self):
        """Initialize model, optimizer, and scheduler"""
        # Initialize model
        self.model = PersonalizedRecommendationModel(
            num_users=10000,  # Should be dynamic
            num_posts=50000,   # Should be dynamic
            embedding_dim=config.EMBEDDING_DIM,
            num_heads=config.NUM_HEADS
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )

        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")

    async def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        criterion = nn.BCELoss()

        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            self.optimizer.zero_grad()

            predictions = self.model(
                batch['user_id'],
                batch['post_id'],
                batch['user_profile'],
                batch['post_content'],
                batch['post_category'],
                batch['context']
            )

            loss = criterion(predictions, batch['label'])

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    async def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model performance"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        criterion = nn.BCELoss()

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                predictions = self.model(
                    batch['user_id'],
                    batch['post_id'],
                    batch['user_profile'],
                    batch['post_content'],
                    batch['post_category'],
                    batch['context']
                )

                loss = criterion(predictions, batch['label'])
                total_loss += loss.item()
                num_batches += 1

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())

        # Calculate metrics
        predictions_binary = [1 if p > 0.5 else 0 for p in all_predictions]

        metrics = {
            'loss': total_loss / num_batches,
            'precision': precision_score(all_labels, predictions_binary, zero_division=0),
            'recall': recall_score(all_labels, predictions_binary, zero_division=0),
            'f1': f1_score(all_labels, predictions_binary, zero_division=0),
            'auc': roc_auc_score(all_labels, all_predictions) if len(set(all_labels)) > 1 else 0.0
        }

        return metrics

    async def train(self, epochs: int = 10) -> Dict[str, Any]:
        """Full training pipeline"""
        try:
            # Start MLflow run
            with mlflow.start_run():
                # Log parameters
                mlflow.log_params({
                    'epochs': epochs,
                    'batch_size': config.BATCH_SIZE,
                    'embedding_dim': config.EMBEDDING_DIM,
                    'num_heads': config.NUM_HEADS,
                    'learning_rate': 0.001
                })

                # Prepare data
                train_loader, val_loader = await self.prepare_training_data()

                # Initialize model
                self.initialize_model()

                best_val_loss = float('inf')
                training_history = []

                for epoch in range(epochs):
                    logger.info(f"Training epoch {epoch + 1}/{epochs}")

                    # Train
                    train_loss = await self.train_epoch(train_loader)

                    # Validate
                    val_metrics = await self.validate(val_loader)

                    # Log metrics
                    mlflow.log_metrics({
                        'train_loss': train_loss,
                        'val_loss': val_metrics['loss'],
                        'val_precision': val_metrics['precision'],
                        'val_recall': val_metrics['recall'],
                        'val_f1': val_metrics['f1'],
                        'val_auc': val_metrics['auc']
                    }, step=epoch)

                    # Update learning rate
                    self.scheduler.step(val_metrics['loss'])

                    # Save best model
                    if val_metrics['loss'] < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        await self.save_model(epoch, val_metrics)

                    training_history.append({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_metrics': val_metrics
                    })

                    logger.info(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_metrics['loss']:.4f}")

                # Log model
                mlflow.pytorch.log_model(self.model, "model")

                # Save final metrics to database
                await self.save_metrics_to_db(val_metrics)

                return {
                    'best_val_loss': best_val_loss,
                    'final_metrics': val_metrics,
                    'training_history': training_history
                }

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    async def save_model(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }

        torch.save(checkpoint, config.MODEL_PATH)
        logger.info(f"Model saved at epoch {epoch}")

    async def save_metrics_to_db(self, metrics: Dict[str, float]):
        """Save training metrics to database"""
        async with self.db_session_factory() as session:
            try:
                for metric_name, metric_value in metrics.items():
                    if metric_name != 'loss':  # Skip loss, save others
                        metric_record = ModelMetrics(
                            model_version="v1.0",
                            metric_name=metric_name,
                            metric_value=metric_value,
                            dataset_size=1000,  # Should be actual dataset size
                            evaluation_config={'training': True}
                        )
                        session.add(metric_record)

                await session.commit()
                logger.info("Metrics saved to database")

            except Exception as e:
                logger.error(f"Error saving metrics to database: {e}")
                await session.rollback()

class RLTrainer:
    """Reinforcement learning trainer for recommendation optimization"""

    def __init__(self, db_session_factory):
        self.db_session_factory = db_session_factory
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = None
        self.replay_buffer = []
        self.batch_size = 32
        self.gamma = 0.99

    def initialize_agent(self):
        """Initialize RL agent"""
        state_dim = config.EMBEDDING_DIM + 32
        action_dim = config.TOP_K_RECOMMENDATIONS

        self.agent = ReinforcementLearningAgent(
            state_dim=state_dim,
            action_dim=action_dim
        ).to(self.device)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=0.0001)

    async def collect_experiences(self, days_back: int = 7):
        """Collect experiences from recent interactions"""
        async with self.db_session_factory() as session:
            from sqlalchemy import select

            # Get recent interactions with rewards
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)

            result = await session.execute(
                select(UserInteraction)
                .where(UserInteraction.timestamp.between(start_date, end_date))
                .where(UserInteraction.reward != 0.0)
            )

            interactions = result.scalars().all()

            experiences = []
            for interaction in interactions:
                # Simplified state representation
                state = np.random.random(config.EMBEDDING_DIM + 32)
                action = np.random.randint(0, config.TOP_K_RECOMMENDATIONS)
                reward = interaction.reward
                next_state = np.random.random(config.EMBEDDING_DIM + 32)

                experiences.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': False
                })

            self.replay_buffer.extend(experiences)
            logger.info(f"Collected {len(experiences)} experiences")

    def train_step(self):
        """Single training step for RL agent"""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch from replay buffer
        batch = np.random.choice(self.replay_buffer, self.batch_size, replace=False)

        states = torch.tensor([exp['state'] for exp in batch], dtype=torch.float32).to(self.device)
        actions = torch.tensor([exp['action'] for exp in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32).to(self.device)
        next_states = torch.tensor([exp['next_state'] for exp in batch], dtype=torch.float32).to(self.device)
        dones = torch.tensor([exp['done'] for exp in batch], dtype=torch.bool).to(self.device)

        # Current Q values
        current_q_values = self.agent(states).gather(1, actions.unsqueeze(1))

        # Next Q values
        next_q_values = self.agent.get_target_q_values(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.agent.update_target_network()

        return loss.item()

# Training script
async def main():
    """Main training script"""
    from db.models import AsyncSessionLocal

    # Setup MLflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    # Initialize trainer
    trainer = ModelTrainer(AsyncSessionLocal)

    # Train model
    results = await trainer.train(epochs=20)

    logger.info("Training completed!")
    logger.info(f"Best validation loss: {results['best_val_loss']:.4f}")
    logger.info(f"Final metrics: {results['final_metrics']}")

if __name__ == "__main__":
    asyncio.run(main())
