import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
import random
from sklearn.preprocessing import LabelEncoder

# Dataset class
class CTUConnectDataset(Dataset):
    """Dataset class cho recommendation system"""
    def __init__(self, data_list, tokenizer, max_length=128):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'user_features': item['user_features'],
            'post_features': item['post_features'],
            'interaction_features': item['interaction_features'],
            'post_id': item['post_id'],
            'timestamp': item['timestamp']
        }

# Model definition
class PersonalizedCTUModel(nn.Module):
    """Mô hình recommendation với PhoBERT"""
    def __init__(self,
                 phobert_model_name="vinai/phobert-base",
                 user_feature_dim=9,
                 post_feature_dim=13,
                 interaction_feature_dim=8,
                 embedding_dim=128,
                 dropout_rate=0.1):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(phobert_model_name)
        self.phobert_dim = self.phobert.config.hidden_size
        self.content_transform = nn.Sequential(
            nn.Linear(self.phobert_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, embedding_dim)
        )
        self.user_transform = nn.Sequential(
            nn.Linear(user_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, embedding_dim)
        )
        self.post_transform = nn.Sequential(
            nn.Linear(post_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, embedding_dim)
        )
        self.interaction_transform = nn.Sequential(
            nn.Linear(interaction_feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 64)
        )
        fusion_dim = embedding_dim * 3 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.final_embedding_dim = embedding_dim

    def forward(self, input_ids, attention_mask, user_features, post_features, interaction_features):
        phobert_output = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        content_pooled = phobert_output.last_hidden_state[:, 0, :]
        content_emb = self.content_transform(content_pooled)
        user_emb = self.user_transform(user_features)
        post_emb = self.post_transform(post_features)
        interaction_emb = self.interaction_transform(interaction_features)
        combined_features = torch.cat([content_emb, user_emb, post_emb, interaction_emb], dim=1)
        output = self.fusion(combined_features)
        return {
            'prediction': output,
            'content_embedding': content_emb,
            'user_embedding': user_emb,
            'post_embedding': post_emb
        }

# Load trained model
def load_trained_model(model_path, config_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Tải mô hình đã huấn luyện và tokenizer"""
    default_config = {
        'phobert_model': 'vinai/phobert-base',
        'max_length': 128,
        'embedding_dim': 128,
        'dropout_rate': 0.1,
        'user_feature_dim': 9,
        'post_feature_dim': 13,
        'interaction_feature_dim': 8
    }

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        config = {
            'phobert_model': 'vinai/phobert-base',
            'max_length': 128,
            'embedding_dim': loaded_config.get('model_config', {}).get('embedding_dim', 128),
            'dropout_rate': 0.1,
            'user_feature_dim': loaded_config.get('model_config', {}).get('user_feature_dim', 9),
            'post_feature_dim': loaded_config.get('model_config', {}).get('post_feature_dim', 13),
            'interaction_feature_dim': loaded_config.get('model_config', {}).get('interaction_feature_dim', 8)
        }
    else:
        print("⚠️ config.json không tồn tại, sử dụng cấu hình mặc định")
        config = default_config

    tokenizer = AutoTokenizer.from_pretrained(config['phobert_model'])

    model = PersonalizedCTUModel(
        phobert_model_name=config['phobert_model'],
        user_feature_dim=config['user_feature_dim'],
        post_feature_dim=config['post_feature_dim'],
        interaction_feature_dim=config['interaction_feature_dim'],
        embedding_dim=config['embedding_dim'],
        dropout_rate=config['dropout_rate']
    )

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"✅ Model loaded from {model_path}")

    return model, tokenizer, config

# Prepare inference data for new posts
def prepare_inference_data(user_id, data_df, posts_df, tokenizer, max_length=128, seen_posts=None):
    """Chuẩn bị dữ liệu cho dự đoán bài viết mới"""
    user_id = str(user_id).strip()
    data_df['user_id'] = data_df['user_id'].astype(str).str.strip()

    # Lấy thông tin người dùng từ data_df
    user_data = data_df[data_df['user_id'] == user_id].copy()
    print(f"🔍 Đã lọc {len(user_data)} bản ghi cho user_id: {user_id}")

    if user_data.empty:
        print(f"⚠️ Không tìm thấy dữ liệu cho user_id: {user_id} trong data_df")
        return []

    # Lấy user_features gần nhất (giả sử user_features không thay đổi theo post_id)
    user_row = user_data.sort_values('timestamp', ascending=False).iloc[0]
    user_features = eval(user_row['user_features']) if isinstance(user_row['user_features'], str) else user_row['user_features']
    if len(user_features) != 9:
        user_features = user_features[:9] + [0.0] * max(0, 9 - len(user_features))

    # Lấy tất cả bài viết từ posts_df
    inference_data = posts_df.copy()
    print(f"🔍 Tổng số bài viết từ posts_df: {len(inference_data)}")

    # Loại bỏ bài viết đã tương tác
    seen_posts = set(data_df[data_df['user_id'] == user_id]['post_id'].unique())
    if seen_posts:
        inference_data = inference_data[~inference_data['id'].isin(seen_posts)]
        print(f"🔍 Sau khi loại bỏ bài viết đã tương tác, còn {len(inference_data)} bài viết")

    if seen_posts:
        inference_data = inference_data[~inference_data['id'].isin(seen_posts)]
        print(f"🔍 Sau khi lọc bài viết đã xem, còn {len(inference_data)} bài viết")

    if inference_data.empty:
        print("⚠️ Không còn bài viết mới để gợi ý sau khi lọc")
        return []

    # Mã hóa post_author_faculty
    post_faculty_encoder = LabelEncoder()
    inference_data['post_author_faculty_encoded'] = post_faculty_encoder.fit_transform(inference_data['author_faculty_name'].fillna('Unknown'))

    data = []
    for _, row in inference_data.iterrows():
        post_text = str(row.get('title', '')) + " " + str(row.get('content', ''))
        if not post_text.strip():
            print(f"⚠️ Bỏ qua bản ghi với post_id {row.get('id', 'N/A')} vì thiếu nội dung")
            continue

        encoding = tokenizer(
            post_text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        # Tạo post_features (giả định post_features trong data_df là mẫu)
        sample_post_features = eval(user_data.iloc[0]['post_features']) if isinstance(user_data.iloc[0]['post_features'], str) else user_data.iloc[0]['post_features']
        post_features = sample_post_features[:12] + [row['post_author_faculty_encoded']]  # Lấy 12 đặc trưng đầu + faculty
        if len(post_features) != 13:
            post_features = post_features[:13] + [0.0] * max(0, 13 - len(post_features))

        # Interaction features mặc định (chưa tương tác)
        interaction_features = [0.0] * 8

        data.append({
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'user_features': torch.tensor(user_features, dtype=torch.float32),
            'post_features': torch.tensor(post_features, dtype=torch.float32),
            'interaction_features': torch.tensor(interaction_features, dtype=torch.float32),
            'post_id': row['id'],
            'timestamp': pd.to_datetime(row['created_at'], errors='coerce')
        })

    return data

# Predict recommendations with batch processing
def predict_recommendations_batch(model, inference_data, device, batch_size=16, freshness_weight=0.3, epsilon=0.1):
    """Dự đoán xác suất tương tác với batch, ưu tiên bài mới và thêm ngẫu nhiên"""
    dataset = CTUConnectDataset(inference_data, None)
    inference_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if os.name == 'nt' else 2,
        pin_memory=True if torch.cuda.is_available() else False,
        collate_fn=lambda x: {
            'input_ids': torch.stack([item['input_ids'] for item in x]),
            'attention_mask': torch.stack([item['attention_mask'] for item in x]),
            'user_features': torch.stack([item['user_features'] for item in x]),
            'post_features': torch.stack([item['post_features'] for item in x]),
            'interaction_features': torch.stack([item['interaction_features'] for item in x]),
            'post_id': [item['post_id'] for item in x],
            'timestamp': [item['timestamp'] for item in x]
        }
    )

    predictions = []
    model.eval()

    max_timestamp = max([d['timestamp'] for d in inference_data], default=pd.Timestamp.now())
    with torch.no_grad():
        for batch in tqdm(inference_loader, desc='Predicting'):
            tensor_batch = {k: v.to(device) for k, v in batch.items() if k not in ['post_id', 'timestamp']}
            post_ids = batch['post_id']
            timestamps = batch['timestamp']

            outputs = model(
                tensor_batch['input_ids'],
                tensor_batch['attention_mask'],
                tensor_batch['user_features'],
                tensor_batch['post_features'],
                tensor_batch['interaction_features']
            )
            probs = torch.sigmoid(outputs['prediction']).cpu().numpy().flatten()

            for post_id, prob, timestamp in zip(post_ids, probs, timestamps):
                freshness_score = 1.0 - ((max_timestamp - timestamp).total_seconds() / (7 * 24 * 3600))
                freshness_score = max(0.0, min(1.0, freshness_score))
                final_score = (1 - freshness_weight) * prob + freshness_weight * freshness_score
                predictions.append({
                    'post_id': post_id,
                    'probability': final_score,
                    'original_prob': prob,
                    'freshness_score': freshness_score
                })

    predictions = sorted(predictions, key=lambda x: (x['post_id'], x['probability']), reverse=True)
    unique_predictions = []
    seen_post_ids = set()
    for pred in predictions:
        if pred['post_id'] not in seen_post_ids:
            unique_predictions.append(pred)
            seen_post_ids.add(pred['post_id'])

    if epsilon > 0:
        top_k = int(len(unique_predictions) * (1 - epsilon))
        top_predictions = unique_predictions[:top_k]
        other_predictions = unique_predictions[top_k:]
        random.shuffle(other_predictions)
        unique_predictions = top_predictions + other_predictions[:max(0, len(unique_predictions) - top_k)]

    unique_predictions = sorted(unique_predictions, key=lambda x: x['probability'], reverse=True)
    return unique_predictions

# Get top N recommendations
def get_top_n_recommendations(predictions, n=5):
    """Lấy top N bài viết được gợi ý"""
    return predictions[:n]

# Main recommendation function
def recommend_articles(user_id, data_df, posts_df, model_path, config_path=None, top_n=5, batch_size=16, seen_posts=None, freshness_weight=0.3, epsilon=0.1):
    """Hàm chính để gợi ý bài viết cho người dùng"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer, config = load_trained_model(model_path, config_path, device)

    inference_data = prepare_inference_data(user_id, data_df, posts_df, tokenizer, config['max_length'], seen_posts)

    if not inference_data:
        print(f"❌ Không có dữ liệu để gợi ý cho user_id: {user_id}")
        return []

    predictions = predict_recommendations_batch(model, inference_data, device, batch_size, freshness_weight, epsilon)

    top_recommendations = get_top_n_recommendations(predictions, top_n)

    return top_recommendations

# Main function
def main():
    """Hàm main để chạy gợi ý bài viết"""
    model_path = 'trained_models/best_ctu_model.pt'
    config_path = 'trained_models/ctu_config.json'
    user_id = '1e2ae772-5902-4ea4-87d7-1a2278e02034'
    top_n = 5
    batch_size = 16
    seen_posts = None  # Sẽ lấy từ data_df
    freshness_weight = 0.3
    epsilon = 0.1

    data_path = 'sample_data/ctu_connect_training.csv'
    posts_path = 'sample_data/ctu_connect_posts.csv'
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file ({data_path}) not found!")
        return
    if not os.path.exists(posts_path):
        print(f"❌ Error: Posts file ({posts_path}) not found!")
        return

    data_df = pd.read_csv(data_path)
    posts_df = pd.read_csv(posts_path)
    print(f"📊 Loaded training dataset with {len(data_df)} records")
    print(f"📊 Loaded posts dataset with {len(posts_df)} records")
    print("Các cột trong data_df:", data_df.columns.tolist())
    print("Các cột trong posts_df:", posts_df.columns.tolist())

    if not os.path.exists(model_path):
        print(f"❌ Error: Model file ({model_path}) not found!")
        return

    print(f"🚀 Generating recommendations for user: {user_id}")
    recommendations = recommend_articles(
        user_id, data_df, posts_df, model_path, config_path, top_n, batch_size, seen_posts, freshness_weight, epsilon
    )

    print(f"\n🏆 Top {top_n} recommended posts:")
    for rec in recommendations:
        print(f"Post ID: {rec['post_id']}, Probability: {rec['probability']:.4f}, Original Prob: {rec['original_prob']:.4f}, Freshness: {rec['freshness_score']:.4f}")

if __name__ == "__main__":
    main()