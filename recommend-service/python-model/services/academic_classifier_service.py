"""
Academic Post Classifier Service
Uses fine-tuned PhoBERT model for binary classification (academic vs non-academic)
"""

import os
import logging
import torch
import numpy as np
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

logger = logging.getLogger(__name__)


class AcademicClassifierService:
    """
    Service for classifying posts as academic or non-academic
    Uses locally fine-tuned PhoBERT model
    """
    
    # Class labels
    LABELS = {
        0: "non_academic",
        1: "academic"
    }
    
    def __init__(self, model_path: str = "./model/academic_posts_model"):
        """
        Initialize the academic classifier
        
        Args:
            model_path: Path to the fine-tuned model directory
        """
        self.model_path = model_path
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_loaded = False
        self.max_length = 256
        
        # Load model on initialization
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned PhoBERT model"""
        try:
            logger.info(f"🔄 Loading academic classifier from: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                logger.error(f"❌ Model path does not exist: {self.model_path}")
                return
            
            # Check required files
            required_files = ["pytorch_model.bin", "config.json", "vocab.txt"]
            for f in required_files:
                if not os.path.exists(os.path.join(self.model_path, f)):
                    logger.error(f"❌ Missing required file: {f}")
                    return
            
            # Load tokenizer (use PhoBERT tokenizer)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=False  # PhoBERT uses slow tokenizer
            )
            logger.info("✅ Tokenizer loaded")
            
            # Load model weights
            model_weights = torch.load(
                os.path.join(self.model_path, "pytorch_model.bin"),
                map_location=self.device,
                weights_only=True
            )
            
            # Check classifier structure
            if "classifier.weight" in model_weights:
                classifier_shape = model_weights["classifier.weight"].shape
                logger.info(f"📊 Classifier shape: {classifier_shape}")
                num_labels = classifier_shape[0]
            else:
                logger.error("❌ No classifier weights found in model")
                return
            
            # Build custom model with simple classifier head
            from transformers import RobertaModel, RobertaConfig
            import torch.nn as nn
            
            # Load config
            config = RobertaConfig.from_pretrained(self.model_path)
            
            # Create custom classification model
            class PhoBERTClassifier(nn.Module):
                def __init__(self, config, num_labels):
                    super().__init__()
                    self.phobert = RobertaModel(config, add_pooling_layer=True)
                    self.classifier = nn.Linear(config.hidden_size, num_labels)
                    self.dropout = nn.Dropout(config.hidden_dropout_prob)
                    
                def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
                    outputs = self.phobert(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                    # Use [CLS] token
                    pooled_output = outputs.last_hidden_state[:, 0, :]
                    pooled_output = self.dropout(pooled_output)
                    logits = self.classifier(pooled_output)
                    return type('Output', (), {'logits': logits})()
            
            self.model = PhoBERTClassifier(config, num_labels)
            
            # Remap keys from saved model (phobert.* -> phobert.*)
            # Filter and load weights
            phobert_weights = {}
            classifier_weights = {}
            
            for key, value in model_weights.items():
                if key.startswith("phobert."):
                    # Remove 'phobert.' prefix for loading into RobertaModel
                    new_key = key.replace("phobert.", "")
                    phobert_weights[new_key] = value
                elif key.startswith("classifier."):
                    classifier_weights[key] = value
            
            # Load PhoBERT weights
            missing, unexpected = self.model.phobert.load_state_dict(phobert_weights, strict=False)
            if missing:
                logger.warning(f"⚠️ Missing keys in PhoBERT: {len(missing)} keys")
            if unexpected:
                logger.warning(f"⚠️ Unexpected keys: {len(unexpected)} keys")
            
            # Load classifier weights
            self.model.classifier.weight.data = classifier_weights["classifier.weight"]
            self.model.classifier.bias.data = classifier_weights["classifier.bias"]
            
            self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"✅ Academic classifier loaded successfully on {self.device}")
            logger.info(f"   - Labels: {self.LABELS}")
            logger.info(f"   - Num labels: {num_labels}")
            logger.info(f"   - Max length: {self.max_length}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load academic classifier: {e}")
            import traceback
            traceback.print_exc()
            self.is_loaded = False
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict if text is academic or non-academic
        
        Args:
            text: Input text to classify
            
        Returns:
            Dict with prediction results:
            - is_academic: bool
            - confidence: float (0-1)
            - label: str ("academic" or "non_academic")
            - probabilities: Dict[str, float]
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            logger.warning("⚠️ Model not loaded, using fallback")
            return self._fallback_predict(text)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                
            # Get prediction
            probs = probabilities[0].cpu().numpy()
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])
            
            is_academic = predicted_class == 1
            label = self.LABELS[predicted_class]
            
            result = {
                "is_academic": is_academic,
                "confidence": round(confidence, 4),
                "label": label,
                "probabilities": {
                    "non_academic": round(float(probs[0]), 4),
                    "academic": round(float(probs[1]), 4)
                }
            }
            
            logger.debug(f"🎯 Classification: '{text[:50]}...' -> {label} ({confidence:.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return self._fallback_predict(text)
    
    def _fallback_predict(self, text: str) -> Dict[str, Any]:
        """
        Fallback heuristic-based prediction when model is unavailable
        """
        academic_keywords = [
            "nghiên cứu", "học thuật", "hội thảo", "seminar", "workshop",
            "luận văn", "luận án", "đề tài", "học bổng", "scholarship",
            "tuyển sinh", "đào tạo", "khóa học", "giảng viên", "giáo sư",
            "báo cáo", "thuyết trình", "đề cương", "tài liệu", "giáo trình",
            "thực tập", "internship", "research", "paper", "thesis",
            "academic", "conference", "journal", "publication"
        ]
        
        text_lower = text.lower()
        matches = sum(1 for kw in academic_keywords if kw in text_lower)
        confidence = min(matches / 5.0, 1.0)
        is_academic = confidence >= 0.4
        
        return {
            "is_academic": is_academic,
            "confidence": round(confidence, 4),
            "label": "academic" if is_academic else "non_academic",
            "probabilities": {
                "non_academic": round(1 - confidence, 4),
                "academic": round(confidence, 4)
            },
            "fallback": True
        }
    
    def batch_predict(self, texts: list) -> list:
        """
        Batch prediction for multiple texts
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of prediction results
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            return [self._fallback_predict(text) for text in texts]
        
        try:
            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict batch
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Process results
            results = []
            probs_np = probabilities.cpu().numpy()
            
            for i, probs in enumerate(probs_np):
                predicted_class = int(np.argmax(probs))
                confidence = float(probs[predicted_class])
                is_academic = predicted_class == 1
                
                results.append({
                    "is_academic": is_academic,
                    "confidence": round(confidence, 4),
                    "label": self.LABELS[predicted_class],
                    "probabilities": {
                        "non_academic": round(float(probs[0]), 4),
                        "academic": round(float(probs[1]), 4)
                    }
                })
            
            logger.info(f"🎯 Batch classified {len(texts)} texts")
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch prediction error: {e}")
            return [self._fallback_predict(text) for text in texts]
    
    def is_ready(self) -> bool:
        """Check if classifier is ready"""
        return self.is_loaded and self.model is not None
    
    def get_info(self) -> Dict[str, Any]:
        """Get classifier information"""
        return {
            "model_path": self.model_path,
            "is_loaded": self.is_loaded,
            "device": str(self.device),
            "labels": self.LABELS,
            "max_length": self.max_length
        }


# Singleton instance
_classifier_instance: Optional[AcademicClassifierService] = None


def get_academic_classifier(model_path: str = "./model/academic_posts_model") -> AcademicClassifierService:
    """
    Get or create singleton instance of AcademicClassifierService
    """
    global _classifier_instance
    
    if _classifier_instance is None:
        _classifier_instance = AcademicClassifierService(model_path)
    
    return _classifier_instance
