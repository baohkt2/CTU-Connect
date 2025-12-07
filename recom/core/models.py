import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from transformers import AutoModel, AutoTokenizer

class UserEmbedding(nn.Module):
    """User embedding layer with profile features"""

    def __init__(self, num_users: int, embedding_dim: int, profile_dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.profile_encoder = nn.Sequential(
            nn.Linear(profile_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )
        self.fusion = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, user_ids: torch.Tensor, profile_features: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        profile_emb = self.profile_encoder(profile_features)
        fused = torch.cat([user_emb, profile_emb], dim=-1)
        return torch.tanh(self.fusion(fused))

class PostEmbedding(nn.Module):
    """Post embedding with content and metadata features"""

    def __init__(self, num_posts: int, embedding_dim: int, content_dim: int = 768):
        super().__init__()
        self.post_embedding = nn.Embedding(num_posts, embedding_dim)
        self.content_encoder = nn.Sequential(
            nn.Linear(content_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.category_embedding = nn.Embedding(20, embedding_dim // 4)  # Assume 20 categories
        self.fusion = nn.Linear(embedding_dim + embedding_dim // 4, embedding_dim)

    def forward(self, post_ids: torch.Tensor, content_features: torch.Tensor,
                categories: torch.Tensor) -> torch.Tensor:
        post_emb = self.post_embedding(post_ids)
        content_emb = self.content_encoder(content_features)
        cat_emb = self.category_embedding(categories)

        # Combine embeddings
        combined = torch.cat([post_emb + content_emb, cat_emb], dim=-1)
        return torch.tanh(self.fusion(combined))

class MultiHeadAttention(nn.Module):
    """Multi-head attention for user-post interaction"""

    def __init__(self, embedding_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert embedding_dim % num_heads == 0

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.output = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, user_emb: torch.Tensor, post_emb: torch.Tensor) -> torch.Tensor:
        batch_size = user_emb.size(0)

        # Generate Q, K, V
        Q = self.query(user_emb).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(post_emb).view(batch_size, self.num_heads, self.head_dim)
        V = self.value(post_emb).view(batch_size, self.num_heads, self.head_dim)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.view(batch_size, -1)

        return self.output(attended)

class PersonalizedRecommendationModel(nn.Module):
    """Main recommendation model with deep learning and attention"""

    def __init__(self, num_users: int, num_posts: int, embedding_dim: int = 256,
                 num_heads: int = 8, content_model_name: str = "vinai/phobert-base"):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Content encoder (PhoBERT for Vietnamese)
        self.content_tokenizer = AutoTokenizer.from_pretrained(content_model_name)
        self.content_encoder = AutoModel.from_pretrained(content_model_name)

        # Freeze PhoBERT parameters for efficiency
        for param in self.content_encoder.parameters():
            param.requires_grad = False

        # Embedding layers
        self.user_embedding = UserEmbedding(num_users, embedding_dim)
        self.post_embedding = PostEmbedding(num_posts, embedding_dim)

        # Attention mechanism
        self.attention = MultiHeadAttention(embedding_dim, num_heads)

        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )

        # Context encoder for temporal and situational features
        self.context_encoder = nn.Sequential(
            nn.Linear(32, embedding_dim // 2),  # time, device, location features
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 4)
        )

    def encode_content(self, texts: List[str]) -> torch.Tensor:
        """Encode Vietnamese text content using PhiBERT"""
        if not texts:
            return torch.zeros(1, 768)

        inputs = self.content_tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.content_encoder(**inputs)
            return outputs.last_hidden_state.mean(dim=1)  # Average pooling

    def forward(self, user_ids: torch.Tensor, post_ids: torch.Tensor,
                user_profiles: torch.Tensor, post_contents: torch.Tensor,
                post_categories: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:

        # Generate embeddings
        user_emb = self.user_embedding(user_ids, user_profiles)
        post_emb = self.post_embedding(post_ids, post_contents, post_categories)

        # Apply attention
        attended_post = self.attention(user_emb, post_emb)

        # Encode context
        context_emb = self.context_encoder(context_features)

        # Combine all features
        combined = torch.cat([
            user_emb,
            attended_post,
            context_emb.expand(-1, self.embedding_dim // 4)
        ], dim=-1)

        # Predict relevance score
        relevance_score = self.predictor(combined)

        return relevance_score.squeeze(-1)

class ReinforcementLearningAgent(nn.Module):
    """DQN-based RL agent for recommendation optimization"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super().__init__()

        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        self.target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.q_network(state)

    def get_target_q_values(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.target_network(state)

    def update_target_network(self):
        """Soft update of target network"""
        tau = 0.005
        for target_param, param in zip(self.target_network.parameters(),
                                     self.q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
