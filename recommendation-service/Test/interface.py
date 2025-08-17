import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers library loaded")
except ImportError as e:
    print(f"Transformers import failed: {e}")
    TRANSFORMERS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
    print("✅ Faiss library loaded")
except ImportError:
    print("Faiss not available - similarity search will be limited")
    FAISS_AVAILABLE = False
try:
    from fastapi import FastAPI
    import uvicorn
    FASTAPI_AVAILABLE = True
    print("✅ FastAPI library loaded")
except ImportError:
    print("FastAPI not available - API will not be served")
    FASTAPI_AVAILABLE = False
try:
    from google.colab import drive
    IN_COLAB = True
    import nest_asyncio
    nest_asyncio.apply()  # Allow nested event loops in Colab
    print("✅ Running in Colab, nest_asyncio applied")
except ImportError:
    IN_COLAB = False
import pickle
import asyncio

class SimpleTokenizer:
    def __init__(self, vocab_size=30000, max_length=128):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word_to_id = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.current_id = 4
    def build_vocab(self, texts):
        for text in texts:
            words = str(text).lower().split()
            for word in words:
                if word not in self.word_to_id and self.current_id < self.vocab_size:
                    self.word_to_id[word] = self.current_id
                    self.id_to_word[self.current_id] = word
                    self.current_id += 1
    def __call__(self, text, truncation=True, padding=True, max_length=None, return_tensors='pt'):
        if max_length is None:
            max_length = self.max_length
        words = str(text).lower().split()
        ids = [self.word_to_id['[CLS]']]
        for word in words:
            if len(ids) >= max_length - 1:
                break
            ids.append(self.word_to_id.get(word, self.word_to_id['[UNK]']))
        ids.append(self.word_to_id['[SEP]'])
        if truncation and len(ids) > max_length:
            ids = ids[:max_length]
        attention_mask = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(self.word_to_id['[PAD]'])
            attention_mask.append(0)
        if return_tensors == 'pt':
            return {
                'input_ids': torch.tensor([ids]),
                'attention_mask': torch.tensor([attention_mask])
            }
        return {
            'input_ids': ids,
            'attention_mask': attention_mask
        }

class PersonalizedCTUModel(nn.Module):
    def __init__(self, vocab_size=30000, embedding_dim=128, user_feature_dim=10,
                 post_feature_dim=14, interaction_feature_dim=8, dropout_rate=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        if TRANSFORMERS_AVAILABLE:
            try:
                self.text_encoder = AutoModel.from_pretrained('vinai/phobert-base')
                self.text_embedding_dim = self.text_encoder.config.hidden_size
                self.use_pretrained = True
                print("✅ Using PhoBERT for text encoding")
            except Exception as e:
                print(f"Failed to load PhoBERT: {e}")
                self.use_pretrained = False
                self._init_simple_text_encoder(vocab_size, embedding_dim, dropout_rate)
        else:
            self.use_pretrained = False
            self._init_simple_text_encoder(vocab_size, embedding_dim, dropout_rate)
        self.user_transform = nn.Sequential(
            nn.Linear(user_feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.post_transform = nn.Sequential(
            nn.Linear(post_feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.interaction_transform = nn.Sequential(
            nn.Linear(interaction_feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        total_dim = self.text_embedding_dim + embedding_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, 1)
        )
    def _init_simple_text_encoder(self, vocab_size, embedding_dim, dropout_rate=0.1):
        self.text_embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(512, embedding_dim)
        self.text_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout_rate,
                batch_first=True
            ),
            num_layers=2
        )
        self.text_pooler = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh()
        )
        print("✅ Using simple transformer for text encoding")
    def encode_text(self, input_ids, attention_mask):
        if self.use_pretrained:
            try:
                outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
                return outputs.pooler_output
            except Exception as e:
                print(f"Error in pretrained encoding: {e}")
                return self._encode_text_simple(input_ids, attention_mask)
        return self._encode_text_simple(input_ids, attention_mask)
    def _encode_text_simple(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        word_embs = self.word_embeddings(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        pos_embs = self.position_embeddings(positions)
        embeddings = word_embs + pos_embs
        attention_mask_bool = attention_mask.bool()
        transformer_output = self.text_transformer(
            embeddings,
            src_key_padding_mask=~attention_mask_bool
        )
        pooled_output = self.text_pooler(transformer_output[:, 0, :])
        return pooled_output
    def forward(self, input_ids, attention_mask, user_features, post_features, interaction_features):
        text_embedding = self.encode_text(input_ids, attention_mask)
        user_emb = self.user_transform(user_features)
        post_emb = self.post_transform(post_features)
        interaction_emb = self.interaction_transform(interaction_features)
        combined = torch.cat([text_embedding, user_emb, post_emb, interaction_emb], dim=-1)
        prediction = self.fusion(combined)
        return {
            'prediction': prediction,
            'user_embedding': user_emb,
            'post_embedding': post_emb,
            'content_embedding': text_embedding
        }

def recommend_articles(user_id, df, top_k=10, index_path='embeddings/faiss_index.bin', threshold_prob=0.5, model_path='trained_models/ctu_model.pt'):
    if not FAISS_AVAILABLE:
        print("Faiss not available, returning empty recommendations")
        return []
    try:
        index = faiss.read_index(index_path)
        with open(index_path.replace('.bin', '_metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        post_ids = metadata['post_ids']
        user_row = df[df['user_id'] == user_id]
        if user_row.empty:
            raise ValueError(f"User {user_id} not found!")
        user_emb = np.array(user_row['user_embedding'].iloc[0]).reshape(1, -1).astype(np.float32)
        distances, indices = index.search(user_emb, 2 * top_k)
        candidate_post_ids = [post_ids[i] for i in indices[0]]
        candidates = df[df['post_id'].isin(candidate_post_ids)]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base') if TRANSFORMERS_AVAILABLE else SimpleTokenizer()
        model = PersonalizedCTUModel(
            vocab_size=30000,
            embedding_dim=128,
            user_feature_dim=10,
            post_feature_dim=14,
            interaction_feature_dim=8,
            dropout_rate=0.1
        ).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        probs = []
        for _, row in candidates.iterrows():
            text = f"{row['title']} {row['content']}"
            encoding = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            uf = torch.tensor(row['user_features'], dtype=torch.float32).to(device).unsqueeze(0)
            pf = torch.tensor(row['post_features'], dtype=torch.float32).to(device).unsqueeze(0)
            if_ = torch.tensor(row['interaction_features'], dtype=torch.float32).to(device).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask, uf, pf, if_)
                prob = torch.sigmoid(outputs['prediction'].squeeze()).item()
                probs.append(prob)
        candidates = candidates.assign(prob=probs)
        candidates = candidates[candidates['prob'] > threshold_prob].sort_values('prob', ascending=False).head(top_k)
        return candidates['post_id'].tolist()
    except Exception as e:
        print(f"Error in recommendation: {e}")
        return []

if FASTAPI_AVAILABLE:
    app = FastAPI()
    @app.get("/recommend/{user_id}")
    async def get_recommendations(user_id: str, top_k: int = 10, threshold_prob: float = 0.5):
        try:
            embedding_path = '/content/drive/MyDrive/embeddings/daily_embeddings.parquet' if IN_COLAB else 'embeddings/daily_embeddings.parquet'
            index_path = '/content/drive/MyDrive/embeddings/faiss_index.bin' if IN_COLAB else 'embeddings/faiss_index.bin'
            model_path = '/content/drive/MyDrive/trained_models/ctu_model.pt' if IN_COLAB else 'trained_models/ctu_model.pt'
            df = pd.read_parquet(embedding_path)
            recommendations = recommend_articles(user_id, df, top_k, index_path, threshold_prob, model_path)
            return {"user_id": user_id, "recommended_post_ids": recommendations}
        except Exception as e:
            return {"error": str(e)}

def main():
    try:
        if IN_COLAB:
            drive.mount('/content/drive')
        if FASTAPI_AVAILABLE:
            print("🚀 Starting FastAPI server...")
            if IN_COLAB:
                # Use nest_asyncio for Colab
                loop = asyncio.get_event_loop()
                config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
                server = uvicorn.Server(config)
                loop.run_until_complete(server.serve())
            else:
                # Normal uvicorn run for non-Colab environments
                uvicorn.run(app, host="0.0.0.0", port=8000)
        else:
            print("❌ FastAPI not available. Please install fastapi and uvicorn.")
    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()