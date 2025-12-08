"""
PhoBERT Inference Engine for CTU Connect Recommendation System
Handles embedding generation for posts and users using PhoBERT model
"""

import sys
import os

# Fix encoding for Windows console (only if not already wrapped)
if sys.platform == 'win32':
    import io
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except (AttributeError, ValueError):
            pass  # Already wrapped or can't wrap
    
    if not isinstance(sys.stderr, io.TextIOWrapper) or sys.stderr.encoding != 'utf-8':
        try:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except (AttributeError, ValueError):
            pass  # Already wrapped or can't wrap

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


class PhoBERTInference:
    """
    PhoBERT inference engine for generating embeddings
    """
    
    def __init__(self, model_path: str = "./model/academic_posts_model"):
        """
        Initialize PhoBERT model and tokenizer
        
        Args:
            model_path: Path to the trained PhoBERT model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("PhoBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def encode_text(self, text: str, max_length: int = 256) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to encode
            max_length: Maximum sequence length
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embedding[0]
        
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise
    
    def encode_batch(self, texts: List[str], max_length: int = 256) -> np.ndarray:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            
        Returns:
            Batch of embedding vectors as numpy array
        """
        try:
            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embeddings
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Error encoding batch: {e}")
            raise
    
    def encode_post(self, post_content: str, post_title: str = "") -> np.ndarray:
        """
        Generate embedding for a post
        
        Args:
            post_content: Post content text
            post_title: Post title (optional)
            
        Returns:
            Post embedding vector
        """
        # Combine title and content
        text = f"{post_title} {post_content}".strip()
        return self.encode_text(text)
    
    def encode_user_profile(self, user_data: Dict) -> np.ndarray:
        """
        Generate embedding for user profile
        
        Args:
            user_data: Dictionary containing user information
                - major: User's major
                - faculty: User's faculty
                - courses: List of courses
                - skills: List of skills
                - bio: User bio
                
        Returns:
            User embedding vector
        """
        # Combine user information into text
        components = []
        
        if user_data.get('major'):
            components.append(f"Chuyên ngành: {user_data['major']}")
        
        if user_data.get('faculty'):
            components.append(f"Khoa: {user_data['faculty']}")
        
        if user_data.get('courses'):
            courses = ', '.join(user_data['courses'])
            components.append(f"Môn học: {courses}")
        
        if user_data.get('skills'):
            skills = ', '.join(user_data['skills'])
            components.append(f"Kỹ năng: {skills}")
        
        if user_data.get('bio'):
            components.append(user_data['bio'])
        
        text = '. '.join(components)
        return self.encode_text(text)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def compute_batch_similarity(self, query_embedding: np.ndarray, 
                                candidate_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarity between one query and multiple candidates
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Array of candidate embeddings
            
        Returns:
            Array of similarity scores
        """
        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Normalize candidates
        candidate_norms = candidate_embeddings / np.linalg.norm(
            candidate_embeddings, axis=1, keepdims=True
        )
        
        # Compute similarities
        similarities = np.dot(candidate_norms, query_norm)
        return similarities


# Global inference engine instance
_inference_engine = None


def get_inference_engine(model_path: str = "./model/academic_posts_model") -> PhoBERTInference:
    """
    Get or create global inference engine instance
    
    Args:
        model_path: Path to model
        
    Returns:
        PhoBERTInference instance
    """
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = PhoBERTInference(model_path)
    return _inference_engine
