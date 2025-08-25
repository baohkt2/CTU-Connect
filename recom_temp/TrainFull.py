import os
import sys
from google.colab import drive
import nest_asyncio
nest_asyncio.apply()
try:
    import findspark
    findspark.init()
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, pandas_udf
    from pyspark.sql.types import ArrayType, FloatType, StructType, StructField
    spark = SparkSession.builder.master("local[2]").appName("CTUConnect").config("spark.driver.memory", "4g").getOrCreate()
    SPARK_AVAILABLE = True
    print("✅ Spark initialized successfully")
except Exception as e:
    print(f"Spark initialization failed: {e}")
    SPARK_AVAILABLE = False
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from transformers import AutoModel, AutoTokenizer
import faiss
import schedule
import time
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from pyngrok import ngrok
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
            return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.tensor([attention_mask])}
        return {'input_ids': ids, 'attention_mask': attention_mask}

class CTUConnectDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        if hasattr(tokenizer, 'build_vocab'):
            texts = []
            for _, row in df.iterrows():
                text = f"{str(row.get('title', ''))} {str(row.get('content', ''))}"
                if 'comment_content' in row and pd.notnull(row['comment_content']):
                    text += f" [SEP] {str(row['comment_content'])}"
                texts.append(text)
            tokenizer.build_vocab(texts)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            # Kết hợp title, content và comment_content (nếu có)
            text = f"{str(row.get('title', ''))} {str(row.get('content', ''))}"
            if 'comment_content' in row and pd.notnull(row['comment_content']):
                text += f" [SEP] {str(row['comment_content'])}"
            encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            def safe_parse_features(feature_str, default_size):
                try:
                    if isinstance(feature_str, str):
                        if feature_str.strip() == '' or feature_str == '[]':
                            return np.zeros(default_size, dtype=np.float32)
                        return np.array(eval(feature_str), dtype=np.float32)
                    return np.array(feature_str, dtype=np.float32) if isinstance(feature_str, (list, np.ndarray)) else np.zeros(default_size, dtype=np.float32)
                except:
                    return np.zeros(default_size, dtype=np.float32)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'user_features': safe_parse_features(row.get('user_features', []), 10),
                'post_features': safe_parse_features(row.get('post_features', []), 14),
                'interaction_features': safe_parse_features(row.get('interaction_features', []), 8),
                'post_id': int(row.get('post_id', 0)),
                'user_id': str(row.get('user_id', '0')),
                'comment_id': str(row.get('comment_id', '0')),
                'label': float(row.get('label', 0))
            }
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'user_features': np.zeros(10, dtype=np.float32),
                'post_features': np.zeros(14, dtype=np.float32),
                'interaction_features': np.zeros(8, dtype=np.float32),
                'post_id': 0,
                'user_id': '0',
                'comment_id': '0',
                'label': 0.0
            }

class PersonalizedCTUModel(nn.Module):
    def __init__(self, vocab_size=30000, embedding_dim=128, user_feature_dim=10, post_feature_dim=14, interaction_feature_dim=8, dropout_rate=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        try:
            self.text_encoder = AutoModel.from_pretrained('vinai/phobert-base')
            self.text_embedding_dim = self.text_encoder.config.hidden_size
            self.use_pretrained = True
            print("✅ Using PhoBERT for text encoding")
        except:
            self.use_pretrained = False
            self._init_simple_text_encoder(vocab_size, embedding_dim, dropout_rate)
        self.user_transform = nn.Sequential(nn.Linear(user_feature_dim, embedding_dim), nn.BatchNorm1d(embedding_dim), nn.ReLU(), nn.Dropout(dropout_rate))
        self.post_transform = nn.Sequential(nn.Linear(post_feature_dim, embedding_dim), nn.BatchNorm1d(embedding_dim), nn.ReLU(), nn.Dropout(dropout_rate))
        self.interaction_transform = nn.Sequential(nn.Linear(interaction_feature_dim, embedding_dim), nn.BatchNorm1d(embedding_dim), nn.ReLU(), nn.Dropout(dropout_rate))
        total_dim = self.text_embedding_dim + embedding_dim * 3
        self.fusion = nn.Sequential(nn.Linear(total_dim, embedding_dim * 2), nn.BatchNorm1d(embedding_dim * 2), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(embedding_dim * 2, embedding_dim), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(embedding_dim, 1))
    def _init_simple_text_encoder(self, vocab_size, embedding_dim, dropout_rate=0.1):
        self.text_embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_embeddings = nn.Embedding(512, embedding_dim)
        self.text_transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, dim_feedforward=embedding_dim * 4, dropout=dropout_rate, batch_first=True), num_layers=2)
        self.text_pooler = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.Tanh())
        print("✅ Using simple transformer for text encoding")
    def encode_text(self, input_ids, attention_mask):
        if self.use_pretrained:
            try:
                outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
                return outputs.pooler_output
            except:
                return self._encode_text_simple(input_ids, attention_mask)
        return self._encode_text_simple(input_ids, attention_mask)
    def _encode_text_simple(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        word_embs = self.word_embeddings(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        pos_embs = self.position_embeddings(positions)
        embeddings = word_embs + pos_embs
        transformer_output = self.text_transformer(embeddings, src_key_padding_mask=~attention_mask.bool())
        pooled_output = self.text_pooler(transformer_output[:, 0, :])
        return pooled_output
    def forward(self, input_ids, attention_mask, user_features, post_features, interaction_features):
        text_embedding = self.encode_text(input_ids, attention_mask)
        user_emb = self.user_transform(user_features)
        post_emb = self.post_transform(post_features)
        interaction_emb = self.interaction_transform(interaction_features)
        combined = torch.cat([text_embedding, user_emb, post_emb, interaction_emb], dim=-1)
        prediction = self.fusion(combined)
        return {'prediction': prediction, 'user_embedding': user_emb, 'post_embedding': post_emb, 'content_embedding': text_embedding}

if SPARK_AVAILABLE:
    def append_faculty_features_spark(user_features: pd.Series, user_faculty_encoded: pd.Series, post_features: pd.Series, post_author_faculty_encoded: pd.Series) -> pd.DataFrame:
        result = []
        for uf, ufe, pf, pafe in zip(user_features, user_faculty_encoded, post_features, post_author_faculty_encoded):
            try:
                uf = eval(uf) if isinstance(uf, str) else uf
                pf = eval(pf) if isinstance(pf, str) else pf
                ufe = float(ufe) if pd.notnull(ufe) else 0.0
                pafe = float(pafe) if pd.notnull(pafe) else 0.0
                uf = list(uf) + [ufe]
                pf = list(pf) + [pafe]
            except:
                uf = [0.0] * 10
                pf = [0.0] * 14
            result.append({'user_features': uf, 'post_features': pf})
        return pd.DataFrame(result)
    append_features_udf = pandas_udf(append_faculty_features_spark, StructType([StructField('user_features', ArrayType(FloatType())), StructField('post_features', ArrayType(FloatType()))]))

def load_and_prepare_data_spark(train_path, posts_path, comments_path):
    try:
        data_df = spark.read.csv(train_path, header=True, inferSchema=True)
        posts_df = spark.read.csv(posts_path, header=True, inferSchema=True)
        comments_df = spark.read.csv(comments_path, header=True, inferSchema=True)

        # Join posts_df với data_df
        data_df = data_df.join(posts_df.select('id', 'title', 'content', 'created_at', 'quality'),
                               data_df.post_id == posts_df.id, 'left')

        # Join comments_df với data_df (bao gồm cả trường hợp không có comment)
        data_df = data_df.join(comments_df.select('id', 'post_id', 'content').withColumnRenamed('id', 'comment_id').withColumnRenamed('content', 'comment_content'),
                               data_df.post_id == comments_df.post_id, 'left_outer')

        # Fill missing values
        data_df = data_df.fillna({
            'title': 'No Title',
            'content': 'No Content',
            'comment_content': 'No Comment',
            'user_features': '[]',
            'post_features': '[]',
            'interaction_features': '[]',
            'user_faculty': 'Unknown',
            'post_author_faculty': 'Unknown'
        })

        # Encode categorical columns
        def encode_column(df, column, new_column):
            if new_column in df.columns:
                df = df.drop(new_column)
            pandas_df = df.select(column).distinct().toPandas()
            le = LabelEncoder()
            pandas_df[new_column] = le.fit_transform(pandas_df[column].fillna('Unknown'))
            mapping = spark.createDataFrame(pandas_df[[column, new_column]])
            temp_column = f"{column}_temp_{new_column}"
            mapping = mapping.withColumnRenamed(column, temp_column).withColumnRenamed(new_column, f"{new_column}_temp")
            result = df.join(mapping, df[column] == mapping[temp_column], 'left')
            result = result.select([col(c) for c in df.columns] + [col(f"{new_column}_temp").alias(new_column)]).drop(temp_column)
            return result

        data_df = encode_column(data_df, 'user_faculty', 'user_faculty_encoded')
        data_df = encode_column(data_df, 'post_author_faculty', 'post_author_faculty_encoded')

        # Append faculty features
        data_df = data_df.withColumn('features', append_features_udf(col('user_features'), col('user_faculty_encoded'), col('post_features'), col('post_author_faculty_encoded')))
        data_df = data_df.withColumn('user_features', col('features.user_features')).withColumn('post_features', col('features.post_features')).drop('features')

        # Order by timestamp and remove duplicates
        data_df = data_df.orderBy(col('timestamp').desc()).dropDuplicates(['post_id', 'comment_id'])

        return data_df.toPandas().head(5000)
    except Exception as e:
        print(f"Error in Spark data loading: {e}")
        raise

def load_and_prepare_data_pandas(train_path, posts_path, comments_path):
    try:
        data_df = pd.read_csv(train_path)
        posts_df = pd.read_csv(posts_path)
        comments_df = pd.read_csv(comments_path)

        # Merge posts_df
        merged_df = data_df.merge(posts_df[['id', 'title', 'content', 'created_at', 'quality']],
                                  left_on='post_id', right_on='id', how='left')

        # Merge comments_df (left outer để giữ các hàng không có comment)
        merged_df = merged_df.merge(comments_df[['id', 'post_id', 'content']].rename(columns={'id': 'comment_id', 'content': 'comment_content'}),
                                    left_on='post_id', right_on='post_id', how='left_outer')

        # Fill missing values
        merged_df['title'] = merged_df['title'].fillna('No Title')
        merged_df['content'] = merged_df['content'].fillna('No Content')
        merged_df['comment_content'] = merged_df['comment_content'].fillna('No Comment')
        merged_df['user_features'] = merged_df['user_features'].fillna('[]')
        merged_df['post_features'] = merged_df['post_features'].fillna('[]')
        merged_df['interaction_features'] = merged_df['interaction_features'].fillna('[]')

        # Encode categorical columns
        le_user = LabelEncoder()
        merged_df['user_faculty_encoded'] = le_user.fit_transform(merged_df['user_faculty'].fillna('Unknown'))
        le_post = LabelEncoder()
        merged_df['post_author_faculty_encoded'] = le_post.fit_transform(merged_df['post_author_faculty'].fillna('Unknown'))

        def append_features(row):
            try:
                uf = eval(row['user_features']) if isinstance(row['user_features'], str) else row['user_features']
                pf = eval(row['post_features']) if isinstance(row['post_features'], str) else row['post_features']
                uf = list(uf) + [float(row['user_faculty_encoded'])] if pd.notnull(row['user_faculty_encoded']) else list(uf) + [0.0]
                pf = list(pf) + [float(row['post_author_faculty_encoded'])] if pd.notnull(row['post_author_faculty_encoded']) else list(pf) + [0.0]
                return pd.Series({'user_features': uf, 'post_features': pf})
            except:
                return pd.Series({'user_features': [0.0] * 10, 'post_features': [0.0] * 14})

        features_df = merged_df.apply(append_features, axis=1)
        merged_df['user_features'] = features_df['user_features']
        merged_df['post_features'] = features_df['post_features']

        # Sort by timestamp and remove duplicates
        merged_df = merged_df.sort_values('timestamp', ascending=False).drop_duplicates(['post_id', 'comment_id'])

        return merged_df.head(5000)
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def load_and_prepare_data(train_path, posts_path, comments_path):
    if SPARK_AVAILABLE:
        try:
            return load_and_prepare_data_spark(train_path, posts_path, comments_path)
        except:
            print("Falling back to pandas...")
            return load_and_prepare_data_pandas(train_path, posts_path, comments_path)
    return load_and_prepare_data_pandas(train_path, posts_path, comments_path)

def custom_collate_fn(batch):
    max_len = max(item['input_ids'].size(0) for item in batch)
    padded_batch = []
    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        if input_ids.size(0) < max_len:
            padding_length = max_len - input_ids.size(0)
            input_ids = torch.cat([input_ids, torch.zeros(padding_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
        elif input_ids.size(0) > max_len:
            input_ids = input_ids[:max_len]
            attention_mask = attention_mask[:max_len]
        padded_item = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'user_features': torch.tensor(item['user_features'], dtype=torch.float32),
            'post_features': torch.tensor(item['post_features'], dtype=torch.float32),
            'interaction_features': torch.tensor(item['interaction_features'], dtype=torch.float32),
            'post_id': item['post_id'],
            'user_id': item['user_id'],
            'comment_id': item['comment_id'],
            'label': torch.tensor(item['label'], dtype=torch.float32)
        }
        padded_batch.append(padded_item)
    return {
        'input_ids': torch.stack([item['input_ids'] for item in padded_batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in padded_batch]),
        'user_features': torch.stack([item['user_features'] for item in padded_batch]),
        'post_features': torch.stack([item['post_features'] for item in padded_batch]),
        'interaction_features': torch.stack([item['interaction_features'] for item in padded_batch]),
        'post_id': [item['post_id'] for item in padded_batch],
        'user_id': [item['user_id'] for item in padded_batch],
        'comment_id': [item['comment_id'] for item in padded_batch],
        'label': torch.stack([item['label'] for item in padded_batch])
    }

def train_model(model, train_loader, optimizer, device, num_epochs=3, checkpoint_path='/content/drive/MyDrive/ctu_connect/datasetTemp/checkpoints/'):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    os.makedirs(checkpoint_path, exist_ok=True)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                user_features = batch['user_features'].to(device)
                post_features = batch['post_features'].to(device)
                interaction_features = batch['interaction_features'].to(device)
                labels = batch['label'].to(device)
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask, user_features, post_features, interaction_features)
                loss = criterion(outputs['prediction'].squeeze(), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # Checkpoint mỗi 100 batch
                if batch_idx % 100 == 0 and batch_idx > 0:
                    checkpoint = {
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item()
                    }
                    torch.save(checkpoint, f"{checkpoint_path}/checkpoint_epoch_{epoch}_batch_{batch_idx}.pt")
                    print(f"✅ Saved checkpoint at epoch {epoch}, batch {batch_idx}")

            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss}")

        # Lưu checkpoint cuối mỗi epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }
        torch.save(checkpoint, f"{checkpoint_path}/checkpoint_epoch_{epoch}.pt")
        print(f"✅ Saved checkpoint at epoch {epoch}")

        # Lưu checkpoint tốt nhất
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, f"{checkpoint_path}/best_checkpoint.pt")
            print(f"✅ Saved best checkpoint with loss {best_loss}")

    return best_loss

def compute_embeddings(model, tokenizer, df, device, checkpoint_path='/content/drive/MyDrive/ctu_connect/datasetTemp/checkpoints/', batch_size=4):
    model.eval()
    dataset = CTUConnectDataset(df, tokenizer, max_length=128)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
    user_embeddings = []
    post_embeddings = []
    content_embeddings = []
    post_ids = []
    user_ids = []
    comment_ids = []

    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_file = f"{checkpoint_path}/embeddings_checkpoint.pkl"
    start_idx = 0

    # Kiểm tra checkpoint hiện có
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        user_embeddings = checkpoint_data['user_embeddings']
        post_embeddings = checkpoint_data['post_embeddings']
        content_embeddings = checkpoint_data['content_embeddings']
        post_ids = checkpoint_data['post_ids']
        user_ids = checkpoint_data['user_ids']
        comment_ids = checkpoint_data['comment_ids']
        start_idx = checkpoint_data['last_idx']
        print(f"✅ Resumed from checkpoint at index {start_idx}")

    for batch_idx, batch in enumerate(tqdm(loader, desc="Computing embeddings", initial=start_idx//batch_size)):
        if batch_idx * batch_size < start_idx:
            continue  # Bỏ qua các batch đã xử lý
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            user_features = batch['user_features'].to(device)
            post_features = batch['post_features'].to(device)
            interaction_features = batch['interaction_features'].to(device)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask, user_features, post_features, interaction_features)
            user_embeddings.extend(outputs['user_embedding'].cpu().numpy())
            post_embeddings.extend(outputs['post_embedding'].cpu().numpy())
            content_embeddings.extend(outputs['content_embedding'].cpu().numpy())
            post_ids.extend(batch['post_id'])
            user_ids.extend(batch['user_id'])
            comment_ids.extend(batch['comment_id'])

            # Lưu checkpoint mỗi 1000 bản ghi
            if (batch_idx + 1) * batch_size % 1000 == 0:
                checkpoint_data = {
                    'user_embeddings': user_embeddings,
                    'post_embeddings': post_embeddings,
                    'content_embeddings': content_embeddings,
                    'post_ids': post_ids,
                    'user_ids': user_ids,
                    'comment_ids': comment_ids,
                    'last_idx': (batch_idx + 1) * batch_size
                }
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                print(f"✅ Saved embeddings checkpoint at index {(batch_idx + 1) * batch_size}")

        except Exception as e:
            print(f"Error computing embeddings for batch {batch_idx}: {e}")
            continue

    df['user_embedding'] = user_embeddings
    df['post_embedding'] = post_embeddings
    df['content_embedding'] = content_embeddings
    df['post_id'] = post_ids
    df['user_id'] = user_ids
    df['comment_id'] = comment_ids

    # Lưu embeddings cuối cùng
    df.to_parquet(f"{checkpoint_path}/../daily_embeddings.parquet")
    print(f"✅ Saved final embeddings to {checkpoint_path}/../daily_embeddings.parquet")
    return df

def build_faiss_index(df, index_path):
    try:
        post_embeddings = np.vstack(df['post_embedding'].apply(np.array).values)
        post_ids = df['post_id'].values
        dimension = post_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(post_embeddings)
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)
        with open(index_path.replace('.bin', '_metadata.pkl'), 'wb') as f:
            pickle.dump({'post_ids': post_ids}, f)
        print(f"✅ Saved Faiss index to {index_path}")
        return index, post_ids
    except Exception as e:
        print(f"Error building Faiss index: {e}")
        return None, None

def daily_precompute():
    try:
        print(f"Running daily precomputation at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        train_path = '/content/drive/MyDrive/ctu_connect/datasetTempTemp/ctu_connect_training.csv'
        posts_path = '/content/drive/MyDrive/ctu_connect/datasetTempTemp/ctu_connect_posts.csv'
        comments_path = '/content/drive/MyDrive/ctu_connect/datasetTempTemp/ctu_connect_comments.csv'
        embedding_path = '/content/drive/MyDrive/ctu_connect/datasetTempTemp/daily_embeddings.parquet'
        index_path = '/content/drive/MyDrive/ctu_connect/datasetTempTemp/faiss_index.bin'
        model_path = '/content/drive/MyDrive/ctu_connect/datasetTempTemp/checkpoints/ctu_model.pt'
        checkpoint_path = '/content/drive/MyDrive/ctu_connect/datasetTempTemp/checkpoints/'

        # Tạo dataset big data nếu chưa tồn tại
        if not os.path.exists(train_path) or not os.path.exists(posts_path) or not os.path.exists(comments_path):
            print("Generating big data dataset...")
            generator = CTUConnectDataGenerator(seed=42)
            dataset = generator.generate_full_dataset(
                n_users=10000,
                n_posts=100000,
                n_interactions=1000000,
                save_path='/content/drive/MyDrive/ctu_connect/datasetTemp/'
            )

        data_df = load_and_prepare_data(train_path, posts_path, comments_path)
        if len(data_df) == 0:
            print("No data loaded, skipping precomputation")
            return

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
        model = PersonalizedCTUModel(vocab_size=30000, embedding_dim=128, user_feature_dim=10, post_feature_dim=14, interaction_feature_dim=8).to(device)

        # Kiểm tra và tải checkpoint mô hình
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded pre-trained model from {model_path}")
        else:
            train_dataset = CTUConnectDataset(data_df, tokenizer, max_length=128)
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            train_model(model, train_loader, optimizer, device, num_epochs=3, checkpoint_path=checkpoint_path)
            torch.save({'model_state_dict': model.state_dict()}, model_path)
            print(f"✅ Saved model to {model_path}")

        data_df = compute_embeddings(model, tokenizer, data_df, device, checkpoint_path=checkpoint_path)
        build_faiss_index(data_df, index_path)
        print("Daily precomputation completed!")
    except Exception as e:
        print(f"Error in daily precomputation: {e}")

class TrainingData(BaseModel):
    train_data: list
    posts_data: list
    comments_data: list

app = FastAPI(title="Training API", description="API for training and embedding generation", version="1.0.0")
tunnel = None
public_url = None

def setup_ngrok(auth_token, port=8000):
    global tunnel, public_url
    try:
        ngrok.set_auth_token(auth_token)
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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "public_url": public_url}

# @app.post("/train")
# async def receive_training_data(data: TrainingData):
#     try:
#         train_df = pd.DataFrame(data.train_data)
#         posts_df = pd.DataFrame(data.posts_data)
#         comments_df = pd.DataFrame(data.comments_data)
#         train_path = '/content/drive/MyDrive/ctu_connect/datasetTemp/ctu_connect_training.csv'
#         posts_path = '/content/drive/MyDrive/ctu_connect/datasetTemp/ctu_connect_posts.csv'
#         comments_path = '/content/drive/MyDrive/ctu_connect/datasetTemp/ctu_connect_comments.csv'
#         os.makedirs(os.path.dirname(train_path), exist_ok=True)
#         train_df.to_csv(train_path, index=False)
#         posts_df.to_csv(posts_path, index=False)
#         comments_df.to_csv(comments_path, index=False)
#         print(f"✅ Received and saved new training data: {len(train_df)} records, {len(posts_df)} posts, {len(comments_df)} comments")
#         daily_precompute()
#         return {"status": "success", "message": "Training completed", "public_url": public_url}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing training data: {e}")

@app.get("/train")
async def receive_training_data():
    try:
        train_path = '/content/drive/MyDrive/ctu_connect/datasetTemp/ctu_connect_training.csv'
        posts_path = '/content/drive/MyDrive/ctu_connect/datasetTemp/ctu_connect_posts.csv'
        comments_path = '/content/drive/MyDrive/ctu_connect/datasetTemp/ctu_connect_comments.csv'

        # Kiểm tra xem các file có tồn tại không
        if not all(os.path.exists(path) for path in [train_path, posts_path, comments_path]):
            raise HTTPException(status_code=404, detail="One or more data files not found in /content/drive/MyDrive/ctu_connect/datasetTemp/")

        # Đọc dữ liệu từ các file CSV
        train_df = pd.read_csv(train_path)
        posts_df = pd.read_csv(posts_path)
        comments_df = pd.read_csv(comments_path)

        print(f"✅ Loaded training data: {len(train_df)} records, {len(posts_df)} posts, {len(comments_df)} comments")

        # Gọi hàm daily_precompute để xử lý và train
        daily_precompute()

        return {"status": "success", "message": "Training completed", "public_url": public_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing training data: {e}")

def main():
    try:
        drive.mount('/content/drive')
        NGROK_AUTH_TOKEN = "31Pe7hDuOjwRujTA2wgi3tDZysh_3a4fEhmqhSw1QxfEK55ga"  # Đã sử dụng token cung cấp
        port = 8000
        setup_ngrok(NGROK_AUTH_TOKEN, port)
        schedule.every().day.at("09:00").do(daily_precompute)
        async def run_schedule():
            while True:
                schedule.run_pending()
                await asyncio.sleep(60)
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info")
        server = uvicorn.Server(config)
        loop = asyncio.get_event_loop()
        loop.create_task(run_schedule())
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        print("🛑 Server stopped by user")
        cleanup_ngrok()
    except Exception as e:
        print(f"❌ Error in main: {e}")
        cleanup_ngrok()

if __name__ == "__main__":
    main()
