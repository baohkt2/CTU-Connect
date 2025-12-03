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
try:
    from pyngrok import ngrok
    NGROK_AVAILABLE = True
    print("✅ Pyngrok library loaded")
except ImportError:
    print("Pyngrok not available - installing...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
        from pyngrok import ngrok
        NGROK_AVAILABLE = True
        print("✅ Pyngrok installed and loaded")
    except Exception as e:
        print(f"Failed to install pyngrok: {e}")
        NGROK_AVAILABLE = False
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

_model_cache = {}
_tokenizer_cache = {}

def load_model_components(model_path, device=None):
    global _model_cache, _tokenizer_cache
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache_key = f"{model_path}_{device}"
    if cache_key not in _model_cache:
        print(f"🔄 Loading model from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if TRANSFORMERS_AVAILABLE:
            try:
                tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
                print("✅ PhoBERT tokenizer loaded")
            except Exception as e:
                print(f"⚠️ PhoBERT tokenizer failed, using simple tokenizer: {e}")
                tokenizer = SimpleTokenizer()
        else:
            tokenizer = SimpleTokenizer()
        model = PersonalizedCTUModel(
            vocab_size=30000,
            embedding_dim=128,
            user_feature_dim=10,
            post_feature_dim=14,
            interaction_feature_dim=8,
            dropout_rate=0.1
        ).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        _model_cache[cache_key] = model
        _tokenizer_cache[cache_key] = tokenizer
        print(f"✅ Model loaded successfully on {device}")
    return _model_cache[cache_key], _tokenizer_cache[cache_key]

def recommend_articles(user_id, df, top_k=10, index_path='embeddings/faiss_index.bin', threshold_prob=0.5,
                      model_path='trained_models/ctu_model.pt'):
    if not FAISS_AVAILABLE:
        print("Faiss not available, returning empty recommendations")
        return []
    try:
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        metadata_path = index_path.replace('.bin', '_metadata.pkl')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        print(f"🔄 Loading FAISS index from {index_path}...")
        index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        post_ids = metadata['post_ids']
        print(f"✅ FAISS index loaded with {len(post_ids)} posts")
        user_row = df[df['user_id'] == user_id]
        if user_row.empty:
            raise ValueError(f"User {user_id} not found in dataframe!")
        user_emb = user_row['user_embedding'].iloc[0]
        if isinstance(user_emb, str):
            user_emb = np.fromstring(user_emb.strip('[]'), sep=',', dtype=np.float32)
        elif isinstance(user_emb, (list, np.ndarray)):
            user_emb = np.array(user_emb, dtype=np.float32)
        else:
            raise ValueError(f"Invalid user_embedding type: {type(user_emb)}")
        user_emb = user_emb.reshape(1, -1)
        print(f"✅ User embedding shape: {user_emb.shape}")
        distances, indices = index.search(user_emb, min(2 * top_k, len(post_ids)))
        candidate_post_ids = [post_ids[i] for i in indices[0] if i < len(post_ids)]
        candidates = df[df['post_id'].isin(candidate_post_ids)]
        print(f"🔍 Found {len(candidates)} candidate posts")
        if candidates.empty:
            print("⚠️ No candidate posts found")
            return []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, tokenizer = load_model_components(model_path, device)
        probs = []
        valid_post_ids = []
        for idx, row in candidates.iterrows():
            try:
                title = str(row.get('title', ''))
                content = str(row.get('content', ''))
                text = f"{title} {content}".strip()
                if not text:
                    continue
                encoding = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                def safe_feature_parse(feature_data, expected_dim, feature_name):
                    try:
                        if isinstance(feature_data, str):
                            if feature_data.strip() == '' or feature_data == 'nan':
                                return np.zeros(expected_dim, dtype=np.float32)
                            feature_data = np.fromstring(feature_data.strip('[]'), sep=',', dtype=np.float32)
                        elif isinstance(feature_data, (list, np.ndarray)):
                            feature_data = np.array(feature_data, dtype=np.float32)
                        else:
                            return np.zeros(expected_dim, dtype=np.float32)
                        if len(feature_data) < expected_dim:
                            feature_data = np.pad(feature_data, (0, expected_dim - len(feature_data)), 'constant')
                        elif len(feature_data) > expected_dim:
                            feature_data = feature_data[:expected_dim]
                        return feature_data
                    except Exception as e:
                        print(f"⚠️ Error parsing {feature_name}: {e}")
                        return np.zeros(expected_dim, dtype=np.float32)
                uf = safe_feature_parse(row.get('user_features', ''), 10, 'user_features')
                pf = safe_feature_parse(row.get('post_features', ''), 14, 'post_features')
                if_ = safe_feature_parse(row.get('interaction_features', ''), 8, 'interaction_features')
                uf = torch.tensor(uf, dtype=torch.float32).to(device).unsqueeze(0)
                pf = torch.tensor(pf, dtype=torch.float32).to(device).unsqueeze(0)
                if_ = torch.tensor(if_, dtype=torch.float32).to(device).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask, uf, pf, if_)
                    prob = torch.sigmoid(outputs['prediction'].squeeze()).item()
                    probs.append(prob)
                    valid_post_ids.append(row['post_id'])
            except Exception as e:
                print(f"⚠️ Error processing candidate {row.get('post_id', 'unknown')}: {e}")
                continue
        if not valid_post_ids:
            print("⚠️ No valid predictions generated")
            return []
        results_df = pd.DataFrame({'post_id': valid_post_ids, 'prob': probs})
        filtered_results = results_df[results_df['prob'] > threshold_prob].sort_values('prob', ascending=False).head(top_k)
        final_recommendations = filtered_results['post_id'].tolist()
        print(f"✅ Generated {len(final_recommendations)} recommendations")
        return final_recommendations
    except Exception as e:
        print(f"❌ Error in recommendation: {e}")
        import traceback
        traceback.print_exc()
        return []

tunnel = None
public_url = None

def setup_ngrok(auth_token, port=8000):
    global tunnel, public_url
    if not NGROK_AVAILABLE:
        print("❌ Ngrok not available. Please install pyngrok.")
        return None
    try:
        ngrok.set_auth_token(auth_token)
        print("✅ Ngrok auth token set")
        ngrok.kill()
        tunnel = ngrok.connect(port, bind_tls=True)
        public_url = tunnel.public_url
        print(f"🌐 Ngrok tunnel created: {public_url}")
        return public_url
    except Exception as e:
        print(f"❌ Error setting up ngrok: {e}")
        return None

def cleanup_ngrok():
    global tunnel, public_url
    try:
        if tunnel and public_url:
            ngrok.disconnect(public_url)
            print(f"🔌 Ngrok tunnel disconnected: {public_url}")
        ngrok.kill()
        tunnel = None
        public_url = None
        print("🧹 Ngrok cleanup completed")
    except Exception as e:
        print(f"⚠️ Error during ngrok cleanup: {e}")

def find_available_port(start_port=8000, max_attempts=10):
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.bind(('127.0.0.1', port))
            sock.close()
            return port
        except OSError:
            continue
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts}")

if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Personalized CTU Recommendation API",
        description="API for personalized article recommendations using CTU model",
        version="1.0.0"
    )
    @app.get("/")
    async def root():
        return {
            "message": "Personalized CTU Recommendation API",
            "status": "running",
            "public_url": public_url,
            "endpoints": {
                "recommendations": "/recommend/{user_id}",
                "health": "/health",
                "docs": "/docs"
            }
        }
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "faiss_available": FAISS_AVAILABLE,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "ngrok_available": NGROK_AVAILABLE,
            "public_url": public_url
        }
    @app.get("/recommend/{user_id}")
    async def get_recommendations(user_id: str, top_k: int = 10, threshold_prob: float = 0.5):
        try:
            embedding_path = '/content/drive/MyDrive/embeddings/daily_embeddings.parquet' if IN_COLAB else 'embeddings/daily_embeddings.parquet'
            index_path = '/content/drive/MyDrive/embeddings/faiss_index.bin' if IN_COLAB else 'embeddings/faiss_index.bin'
            model_path = '/content/drive/MyDrive/trained_models/ctu_model.pt' if IN_COLAB else 'trained_models/ctu_model.pt'
            df = pd.read_parquet(embedding_path, engine='pyarrow')
            recommendations = recommend_articles(user_id, df, top_k, index_path, threshold_prob, model_path)
            return {
                "user_id": user_id,
                "recommended_post_ids": recommendations,
                "count": len(recommendations),
                "threshold_prob": threshold_prob,
                "public_url": public_url
            }
        except Exception as e:
            return {"error": str(e), "user_id": user_id}

def main(auth_token=None, port=8000):
    try:
        if IN_COLAB:
            drive.mount('/content/drive')
        if not FASTAPI_AVAILABLE:
            print("❌ FastAPI not available. Please install fastapi and uvicorn.")
            return
        if not NGROK_AVAILABLE and IN_COLAB:
            print("❌ Ngrok not available in Colab. Please install pyngrok.")
            return
        port = find_available_port(port)
        print(f"🚀 Starting FastAPI server on port {port}...")
        if IN_COLAB and auth_token:
            public_url = setup_ngrok(auth_token, port)
            if public_url:
                print(f"🌐 Public URL: {public_url}")
                print(f"📖 API Documentation: {public_url}/docs")
                print(f"🔗 Recommendation endpoint: {public_url}/recommend/{{user_id}}")
            else:
                print("❌ Failed to create ngrok tunnel. Running locally only.")
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info")
        server = uvicorn.Server(config)
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        cleanup_ngrok()
    except Exception as e:
        print(f"❌ Error in main: {e}")
        cleanup_ngrok()
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Lấy auth token từ biến môi trường hoặc file cấu hình
    NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "YOUR_AUTH_TOKEN_HERE")  # Thay bằng token thật
    main(auth_token=NGROK_AUTH_TOKEN)