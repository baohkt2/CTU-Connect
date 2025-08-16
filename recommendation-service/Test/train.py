import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Dataset class
class CTUConnectDataset(Dataset):
    """Dataset class cho recommendation system"""
    def __init__(self, data_df, tokenizer, max_length=128):
        self.data_df = data_df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        post_text = str(row['title']) + " " + str(row['content'])
        encoding = self.tokenizer(
            post_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        user_features = eval(row['user_features']) if isinstance(row['user_features'], str) else row['user_features']
        post_features = eval(row['post_features']) if isinstance(row['post_features'], str) else row['post_features']
        interaction_features = eval(row['interaction_features']) if isinstance(row['interaction_features'], str) else row['interaction_features']

        # Đảm bảo kích thước đặc trưng
        if len(user_features) != 9:
            user_features = user_features[:9] + [0.0] * max(0, 9 - len(user_features))
        if len(post_features) != 13:
            post_features = post_features[:13] + [0.0] * max(0, 13 - len(post_features))
        if len(interaction_features) != 8:
            interaction_features = interaction_features[:8] + [0.0] * max(0, 8 - len(interaction_features))

        # Chuyển timestamp thành Unix timestamp
        timestamp = pd.to_datetime(row['timestamp'], errors='coerce')
        unix_timestamp = timestamp.timestamp() if pd.notna(timestamp) else 0.0

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'user_features': torch.tensor(user_features, dtype=torch.float32),
            'post_features': torch.tensor(post_features, dtype=torch.float32),
            'interaction_features': torch.tensor(interaction_features, dtype=torch.float32),
            'label': torch.tensor(row['label'], dtype=torch.float32),
            'post_id': row['post_id'],
            'timestamp': torch.tensor(unix_timestamp, dtype=torch.float32)
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

# Load and prepare data
def load_and_prepare_data(train_path, posts_path):
    """Tải và tiền xử lý dữ liệu"""
    data_df = pd.read_csv(train_path)
    posts_df = pd.read_csv(posts_path)
    print(f"Số bản ghi trong data_df: {len(data_df)}")
    print(f"Số bản ghi trong posts_df: {len(posts_df)}")

    # Merge dữ liệu
    data_df = data_df.merge(posts_df[['id', 'title', 'content', 'created_at']], left_on='post_id', right_on='id', how='left')
    print(f"Số bản ghi sau merge: {len(data_df)}")

    if data_df.empty:
        raise ValueError("Dữ liệu sau merge rỗng! Kiểm tra post_id hoặc file dữ liệu.")

    required_columns = ['user_id', 'post_id', 'user_features', 'post_features', 'interaction_features', 'label', 'title', 'content', 'user_faculty', 'post_author_faculty']
    missing_columns = [col for col in required_columns if col not in data_df.columns]
    if missing_columns:
        raise ValueError(f"Thiếu các cột: {missing_columns}")

    # Loại bỏ NA
    data_df = data_df.dropna(subset=['title', 'content'])
    print(f"Số bản ghi sau khi loại bỏ NA: {len(data_df)}")

    # Kiểm tra kích thước đặc trưng ban đầu
    for col in ['user_features', 'post_features', 'interaction_features']:
        lengths = data_df[col].apply(lambda x: len(eval(x) if isinstance(x, str) else x)).unique()
        print(f"{col} length before update: {lengths}")

    # Mã hóa user_faculty và post_author_faculty
    user_faculty_encoder = LabelEncoder()
    post_faculty_encoder = LabelEncoder()
    data_df['user_faculty_encoded'] = user_faculty_encoder.fit_transform(data_df['user_faculty'].fillna('Unknown'))
    data_df['post_author_faculty_encoded'] = post_faculty_encoder.fit_transform(data_df['post_author_faculty'].fillna('Unknown'))

    # Thêm đặc trưng học thuật
    def append_faculty_features(row):
        user_features = eval(row['user_features']) if isinstance(row['user_features'], str) else row['user_features']
        post_features = eval(row['post_features']) if isinstance(row['post_features'], str) else row['post_features']
        user_features = list(user_features) + [row['user_faculty_encoded']]
        post_features = list(post_features) + [row['post_author_faculty_encoded']]
        return user_features, post_features

    data_df[['user_features', 'post_features']] = data_df.apply(append_faculty_features, axis=1, result_type='expand')

    # Kiểm tra kích thước đặc trưng sau cập nhật
    for col in ['user_features', 'post_features', 'interaction_features']:
        lengths = data_df[col].apply(lambda x: len(eval(x) if isinstance(x, str) else x)).unique()
        print(f"{col} length after update: {lengths}")

    # Loại bỏ post_id trùng lặp, giữ bản ghi mới nhất
    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'], errors='coerce')
    data_df = data_df.sort_values('timestamp', ascending=False).drop_duplicates('post_id', keep='first')
    print(f"Số bản ghi sau khi loại bỏ trùng lặp: {len(data_df)}")

    return data_df

# Split data
def split_data(data_df, test_size=0.2, random_state=42):
    """Chia dữ liệu thành train/test"""
    if len(data_df) == 0:
        raise ValueError("Dữ liệu đầu vào rỗng! Không thể chia train/test.")

    train_df, test_df = train_test_split(data_df, test_size=test_size, random_state=random_state)
    print(f"Số bản ghi train: {len(train_df)}, test: {len(test_df)}")
    return train_df, test_df

# Train model
def train_model(model, train_loader, optimizer, device, num_epochs=5):
    """Huấn luyện mô hình"""
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            user_features = batch['user_features'].to(device)
            post_features = batch['post_features'].to(device)
            interaction_features = batch['interaction_features'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, user_features, post_features, interaction_features)
            loss = criterion(outputs['prediction'].squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

# Save model
def save_model(model, path):
    """Lưu mô hình"""
    torch.save({'model_state_dict': model.state_dict()}, path)
    print(f"✅ Model saved to {path}")

# Main function
def main():
    """Hàm main để huấn luyện mô hình"""
    train_path = 'sample_data/ctu_connect_training.csv'
    posts_path = 'sample_data/ctu_connect_posts.csv'
    model_path = 'trained_models/best_ctu_model.pt'
    config_path = 'trained_models/ctu_config.json'
    batch_size = 16
    num_epochs = 5
    learning_rate = 2e-5

    # Tải và tiền xử lý dữ liệu
    data_df = load_and_prepare_data(train_path, posts_path)
    train_df, test_df = split_data(data_df)

    # Khởi tạo tokenizer và model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    model = PersonalizedCTUModel(
        phobert_model_name='vinai/phobert-base',
        user_feature_dim=9,
        post_feature_dim=13,
        interaction_feature_dim=8,
        embedding_dim=128,
        dropout_rate=0.1
    ).to(device)

    # Tạo DataLoader
    train_dataset = CTUConnectDataset(train_df, tokenizer, max_length=128)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if os.name == 'nt' else 2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Huấn luyện
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_model(model, train_loader, optimizer, device, num_epochs)

    # Lưu mô hình
    os.makedirs('trained_models', exist_ok=True)
    save_model(model, model_path)

    # Lưu config
    config = {
        'model_config': {
            'phobert_model': 'vinai/phobert-base',
            'user_feature_dim': 9,
            'post_feature_dim': 13,
            'interaction_feature_dim': 8,
            'embedding_dim': 128,
            'dropout_rate': 0.1,
            'max_length': 128
        }
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✅ Config saved to {config_path}")

if __name__ == "__main__":
    main()