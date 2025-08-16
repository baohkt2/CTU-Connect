
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dataset class
class ArticleRecommendationDataset(Dataset):
    """Dataset class cho recommendation system"""
    def __init__(self, data_list, tokenizer, max_length=256):
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
            'article_features': item['article_features'],
            'interaction_features': item['interaction_features'],
            'article_id': item['article_id'],
            'label': item['label']  # Thêm label cho huấn luyện
        }

# Model definition
class PersonalizedArticleModel(nn.Module):
    """Mô hình recommendation với PhoBERT"""
    def __init__(self,
                 phobert_model_name="vinai/phobert-base",
                 user_feature_dim=20,
                 article_feature_dim=16,  # Sử dụng 16 chiều với academic_score
                 interaction_feature_dim=10,
                 embedding_dim=128,
                 dropout_rate=0.1,
                 freeze_phobert=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.phobert = AutoModel.from_pretrained(phobert_model_name)
        self.phobert_dim = self.phobert.config.hidden_size

        if freeze_phobert:
            for param in self.phobert.parameters():
                param.requires_grad = False

        self.content_transform = nn.Sequential(
            nn.Linear(self.phobert_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        self.user_transform = nn.Sequential(
            nn.Linear(user_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        self.article_transform = nn.Sequential(
            nn.Linear(article_feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        self.interaction_transform = nn.Sequential(
            nn.Linear(interaction_feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 64),
            nn.LayerNorm(64)
        )

        fusion_dim = embedding_dim * 3 + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_ids, attention_mask, user_features, article_features, interaction_features):
        batch_size = input_ids.size(0)
        phobert_output = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        content_pooled = phobert_output.last_hidden_state[:, 0, :]
        content_emb = self.content_transform(content_pooled)
        user_emb = self.user_transform(user_features)
        article_emb = self.article_transform(article_features)
        interaction_emb = self.interaction_transform(interaction_features)
        combined_features = torch.cat([content_emb, user_emb, article_emb, interaction_emb], dim=1)
        output = self.fusion(combined_features)
        return {
            'prediction': output,
            'content_embedding': content_emb,
            'user_embedding': user_emb,
            'article_embedding': article_emb
        }

# Chuẩn bị dữ liệu huấn luyện
def prepare_training_data(data_df, tokenizer, max_length=256):
    """Chuẩn bị dữ liệu huấn luyện từ dataset"""
    data = []
    for _, row in data_df.iterrows():
        article_text = str(row['article_content'])
        encoding = tokenizer(
            article_text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        user_features = eval(row['user_features']) if isinstance(row['user_features'], str) else row['user_features']
        article_features = eval(row['article_features']) if isinstance(row['article_features'], str) else row['article_features']
        interaction_features = eval(row['interaction_features']) if isinstance(row['interaction_features'], str) else row['interaction_features']

        # Thêm academic_score vào article_features
        academic_score = 1 if any(keyword in article_text.lower() for keyword in ['nghiên cứu', 'research', 'study', 'journal']) or len(article_text.split()) > 1000 else 0
        if len(article_features) == 15:
            article_features = article_features + [academic_score]
        elif len(article_features) < 16:
            article_features = article_features[:15] + [academic_score]

        if len(user_features) != 20:
            user_features = user_features[:20] + [0.0] * max(0, 20 - len(user_features))
        if len(article_features) != 16:
            article_features = article_features[:16] + [0.0] * max(0, 16 - len(article_features))
        if len(interaction_features) != 10:
            interaction_features = interaction_features[:10] + [0.0] * max(0, 10 - len(interaction_features))

        data.append({
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'user_features': torch.tensor(user_features, dtype=torch.float32),
            'article_features': torch.tensor(article_features, dtype=torch.float32),
            'interaction_features': torch.tensor(interaction_features, dtype=torch.float32),
            'article_id': row['article_id'],
            'label': torch.tensor(float(row['label']), dtype=torch.float32)  # Giả sử có cột label
        })
    return data

# Hàm huấn luyện
def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=2e-5):
    """Huấn luyện mô hình"""
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))  # Tăng trọng số cho label 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            user_features = batch['user_features'].to(device)
            article_features = batch['article_features'].to(device)
            interaction_features = batch['interaction_features'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, user_features, article_features, interaction_features)
            loss = criterion(outputs['prediction'].squeeze(), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                user_features = batch['user_features'].to(device)
                article_features = batch['article_features'].to(device)
                interaction_features = batch['interaction_features'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask, user_features, article_features, interaction_features)
                loss = criterion(outputs['prediction'].squeeze(), labels)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'trained_models/best_model_epoch_final.pt')
            logger.info("✅ Saved best model")

# Main function
def main():
    """Hàm main để huấn luyện lại mô hình từ checkpoint"""
    model_path = 'trained_models/best_model_epoch_3.pt'  # Đường dẫn đến model cũ
    config_path = 'trained_models/config.json'  # Có thể bỏ qua nếu không có
    data_path = 'recommendation_data/training_dataset.csv'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Kiểm tra file
    if not os.path.exists(data_path):
        logger.error(f"❌ Error: Data file ({data_path}) not found!")
        return
    if not os.path.exists(model_path):
        logger.error(f"❌ Error: Model file ({model_path}) not found!")
        return

    # Tải dữ liệu
    data_df = pd.read_csv(data_path)
    logger.info(f"📊 Loaded dataset with {len(data_df)} records")

    # Tách train và validation (giả sử 80-20)
    train_df = data_df.sample(frac=0.8, random_state=42)
    val_df = data_df.drop(train_df.index)

    # Chuẩn bị tokenizer và mô hình
    config = {
        'phobert_model_name': 'vinai/phobert-base',
        'embedding_dim': 128,
        'dropout_rate': 0.1,
        'freeze_phobert': True,
        'user_feature_dim': 20,
        'article_feature_dim': 16,  # Đảm bảo khớp với academic_score
        'interaction_feature_dim': 10
    }
    max_length = 256  # Define max_length separately
    tokenizer = AutoTokenizer.from_pretrained(config['phobert_model_name'])
    model = PersonalizedArticleModel(**config).to(device)

    # Load checkpoint từ model cũ
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✅ Loaded checkpoint from {model_path}")
    else:
        logger.warning("⚠️ No checkpoint found, training from initialized model")

    # Chuẩn bị dữ liệu huấn luyện
    train_data = prepare_training_data(train_df, tokenizer, max_length)
    val_data = prepare_training_data(val_df, tokenizer, max_length)
    train_loader = DataLoader(ArticleRecommendationDataset(train_data, tokenizer), batch_size=16, shuffle=True)
    val_loader = DataLoader(ArticleRecommendationDataset(val_data, tokenizer), batch_size=16)

    # Huấn luyện lại
    train_model(model, train_loader, val_loader, device)

if __name__ == "__main__":
    main()

