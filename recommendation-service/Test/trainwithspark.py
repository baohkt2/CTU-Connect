from google.colab import drive
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when
from pyspark.sql.types import ArrayType, FloatType

# Cài đặt Spark
spark = SparkSession.builder.master("local[*]").appName("CTUConnect").getOrCreate()


# [Giữ nguyên CTUConnectDataset và PersonalizedCTUModel]

def load_and_prepare_data(train_path, posts_path):
    data_df = spark.read.csv(train_path, header=True, inferSchema=True)
    posts_df = spark.read.csv(posts_path, header=True, inferSchema=True)
    data_df = data_df.join(posts_df.select('id', 'title', 'content', 'created_at'), data_df.post_id == posts_df.id,
                           'left')
    data_df = data_df.na.drop(subset=['title', 'content'])

    def encode_column(df, column, new_column):
        pandas_df = df.select(column).distinct().toPandas()
        le = LabelEncoder()
        pandas_df[new_column] = le.fit_transform(pandas_df[column].fillna('Unknown'))
        mapping = spark.createDataFrame(pandas_df)
        return df.join(mapping, column, 'left').select(df['*'], mapping[new_column])

    data_df = encode_column(data_df, 'user_faculty', 'user_faculty_encoded')
    data_df = encode_column(data_df, 'post_author_faculty', 'post_author_faculty_encoded')

    data_df = data_df.withColumn('features', append_features_udf(
        col('user_features'), col('user_faculty_encoded'),
        col('post_features'), col('post_author_faculty_encoded')
    ))
    data_df = data_df.withColumn('user_features', col('features')[0])
    data_df = data_df.withColumn('post_features', col('features')[1]).drop('features')

    data_df = data_df.orderBy(col('timestamp').desc()).dropDuplicates(['post_id'])
    return data_df.toPandas()


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)
    print(f"✅ Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"✅ Loaded checkpoint from {path}, starting from epoch {epoch + 1}")
        return epoch + 1, loss
    return 0, None


def train_model(model, train_loader, optimizer, device, num_epochs=5,
                checkpoint_path='/content/drive/MyDrive/checkpoints/ctu_checkpoint.pt'):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path, device)

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
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
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
        save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)


def precompute_embeddings_spark(model, tokenizer, data_df, device, max_length=128,
                                output_path='/content/drive/MyDrive/embeddings/embeddings.parquet'):
    model.eval()
    dataset = CTUConnectDataset(data_df, tokenizer, max_length=max_length)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0,
                             pin_memory=True if torch.cuda.is_available() else False)

    user_embeddings = []
    post_embeddings = []
    content_embeddings = []
    post_ids = []
    user_ids = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Precomputing embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            user_features = batch['user_features'].to(device)
            post_features = batch['post_features'].to(device)
            outputs = model(input_ids, attention_mask, user_features, post_features,
                            batch['interaction_features'].to(device))
            user_embeddings.append(outputs['user_embedding'].cpu().numpy())
            post_embeddings.append(outputs['post_embedding'].cpu().numpy())
            content_embeddings.append(outputs['content_embedding'].cpu().numpy())
            post_ids.extend(batch['post_id'])
            user_ids.extend(batch['user_id'])

    embedding_df = pd.DataFrame({
        'user_id': user_ids,
        'post_id': post_ids,
        'user_embedding': list(np.vstack(user_embeddings)),
        'post_embedding': list(np.vstack(post_embeddings)),
        'content_embedding': list(np.vstack(content_embeddings))
    })

    spark_df = spark.createDataFrame(embedding_df)
    spark_df.write.mode('overwrite').parquet(output_path)
    print(f"✅ Saved embeddings to {output_path}")
    return spark_df


def main():
    drive.mount('/content/drive')
    train_path = '/content/drive/MyDrive/data/ctu_connect_training.csv'
    posts_path = '/content/drive/MyDrive/data/ctu_connect_posts.csv'
    model_path = '/content/drive/MyDrive/trained_models/best_ctu_model.pt'
    checkpoint_path = '/content/drive/MyDrive/checkpoints/ctu_checkpoint.pt'
    config_path = '/content/drive/MyDrive/trained_models/ctu_config.json'
    embedding_path = '/content/drive/MyDrive/embeddings/embeddings.parquet'
    batch_size = 8
    num_epochs = 5
    learning_rate = 2e-5

    # Tải và tiền xử lý dữ liệu
    data_df = load_and_prepare_data(train_path, posts_path)
    train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42)

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
    train_model(model, train_loader, optimizer, device, num_epochs, checkpoint_path)

    # Tiền tính toán embedding
    precompute_embeddings_spark(model, tokenizer, data_df, device, output_path=embedding_path)

    # Lưu mô hình
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
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