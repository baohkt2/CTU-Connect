import os
import sys
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
try:
    import findspark
    findspark.init()
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, pandas_udf, PandasUDFType
    from pyspark.sql.types import ArrayType, FloatType, StructType, StructField
    spark = SparkSession.builder \
        .master("local[2]") \
        .appName("CTUConnect") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .getOrCreate()
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
import json
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
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
import schedule
import time
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
class CTUConnectDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        if hasattr(tokenizer, 'build_vocab'):
            texts = [f"{row.get('title', '')} {row.get('content', '')}"
                    for _, row in df.iterrows()]
            tokenizer.build_vocab(texts)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            text = f"{str(row.get('title', ''))} {str(row.get('content', ''))}"
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            if input_ids.size(0) != self.max_length:
                if input_ids.size(0) > self.max_length:
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
                else:
                    padding_length = self.max_length - input_ids.size(0)
                    input_ids = torch.cat([input_ids, torch.zeros(padding_length, dtype=torch.long)])
                    attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
            def safe_parse_features(feature_str, default_size):
                try:
                    if isinstance(feature_str, str):
                        if feature_str.strip() == '' or feature_str == '[]':
                            features = [0.0] * default_size
                        else:
                            features = eval(feature_str)
                    elif isinstance(feature_str, (list, np.ndarray)):
                        features = list(feature_str)
                    elif pd.isna(feature_str):
                        features = [0.0] * default_size
                    else:
                        features = [float(feature_str)] if default_size == 1 else [0.0] * default_size
                    if len(features) != default_size:
                        if len(features) > default_size:
                            features = features[:default_size]
                        else:
                            features = features + [0.0] * (default_size - len(features))
                    return np.array(features, dtype=np.float32)
                except Exception as e:
                    print(f"Error parsing features: {e}, using zeros")
                    return np.zeros(default_size, dtype=np.float32)
            def safe_parse_id(id_value, default=0):
                try:
                    if isinstance(id_value, str):
                        if '-' in id_value and len(id_value) > 10:
                            return hash(id_value) % (2**31)
                        return int(id_value)
                    return int(id_value) if id_value is not None else default
                except (ValueError, TypeError):
                    return default
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'user_features': safe_parse_features(row.get('user_features', []), 9),
                'post_features': safe_parse_features(row.get('post_features', []), 13),
                'interaction_features': safe_parse_features(row.get('interaction_features', []), 8),
                'post_id': safe_parse_id(row.get('post_id', 0)),
                'user_id': safe_parse_id(row.get('user_id', 0)),
                'label': float(row.get('label', 0))
            }
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'user_features': np.zeros(9, dtype=np.float32),
                'post_features': np.zeros(13, dtype=np.float32),
                'interaction_features': np.zeros(8, dtype=np.float32),
                'post_id': 0,
                'user_id': 0,
                'label': 0.0
            }
class PersonalizedCTUModel(nn.Module):
    def __init__(self, vocab_size=30000, embedding_dim=128, user_feature_dim=9,
                 post_feature_dim=13, interaction_feature_dim=8, dropout_rate=0.1):
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
if SPARK_AVAILABLE:
    def append_faculty_features_spark(user_features: pd.Series, user_faculty_encoded: pd.Series,
                                    post_features: pd.Series, post_author_faculty_encoded: pd.Series) -> pd.DataFrame:
        result = []
        for uf, ufe, pf, pafe in zip(user_features, user_faculty_encoded, post_features, post_author_faculty_encoded):
            try:
                uf = eval(uf) if isinstance(uf, str) else uf
                pf = eval(pf) if isinstance(pf, str) else pf
                ufe = float(ufe) if pd.notnull(ufe) else 0.0
                pafe = float(pafe) if pd.notnull(pafe) else 0.0
                uf = list(uf) + [ufe]
                pf = list(pf) + [pafe]
            except Exception as e:
                print(f"Error in UDF: {e}")
                uf = [0.0] * 10  # Adjusted for user_faculty_encoded
                pf = [0.0] * 14  # Adjusted for post_author_faculty_encoded
            result.append({'user_features': uf, 'post_features': pf})
        return pd.DataFrame(result)
    append_features_udf = pandas_udf(append_faculty_features_spark, StructType([
        StructField('user_features', ArrayType(FloatType())),
        StructField('post_features', ArrayType(FloatType()))
    ]))
def load_and_prepare_data_spark(train_path, posts_path):
    try:
        data_df = spark.read.csv(train_path, header=True, inferSchema=True)
        posts_df = spark.read.csv(posts_path, header=True, inferSchema=True)
        print(f"Training records: {data_df.count()}")
        print(f"Posts records: {posts_df.count()}")
        data_df = data_df.join(
            posts_df.select('id', 'title', 'content', 'created_at'),
            data_df.post_id == posts_df.id,
            'left'
        )
        data_df = data_df.fillna({
            'title': 'No Title',
            'content': 'No Content',
            'user_features': '[]',
            'post_features': '[]',
            'interaction_features': '[]',
            'user_faculty': 'Unknown',
            'post_author_faculty': 'Unknown'
        })
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
            result = result.select([col(c) for c in df.columns] + [col(f"{new_column}_temp").alias(new_column)])
            return result.drop(temp_column)
        data_df = encode_column(data_df, 'user_faculty', 'user_faculty_encoded')
        print(f"Records after user_faculty encode: {data_df.count()}")
        data_df = encode_column(data_df, 'post_author_faculty', 'post_author_faculty_encoded')
        print(f"Records after post_author_faculty encode: {data_df.count()}")
        data_df = data_df.withColumn('features', append_features_udf(
            col('user_features'), col('user_faculty_encoded'),
            col('post_features'), col('post_author_faculty_encoded')
        ))
        data_df = data_df.withColumn('user_features', col('features.user_features'))
        data_df = data_df.withColumn('post_features', col('features.post_features')).drop('features')
        data_df = data_df.orderBy(col('timestamp').desc()).dropDuplicates(['post_id'])
        print(f"Final records: {data_df.count()}")
        return data_df.toPandas().head(5000)
    except Exception as e:
        print(f"Error in Spark data loading: {e}")
        raise e
def load_and_prepare_data_pandas(train_path, posts_path):
    try:
        print("Loading data with pandas...")
        data_df = pd.read_csv(train_path)
        posts_df = pd.read_csv(posts_path)
        print(f"Training records: {len(data_df)}")
        print(f"Posts records: {len(posts_df)}")
        merged_df = data_df.merge(
            posts_df[['id', 'title', 'content', 'created_at']],
            left_on='post_id',
            right_on='id',
            how='left'
        )
        merged_df['title'] = merged_df['title'].fillna('No Title')
        merged_df['content'] = merged_df['content'].fillna('No Content')
        merged_df['user_features'] = merged_df['user_features'].fillna('[]')
        merged_df['post_features'] = merged_df['post_features'].fillna('[]')
        merged_df['interaction_features'] = merged_df['interaction_features'].fillna('[]')
        le_user = LabelEncoder()
        merged_df['user_faculty_encoded'] = le_user.fit_transform(
            merged_df['user_faculty'].fillna('Unknown')
        )
        le_post = LabelEncoder()
        merged_df['post_author_faculty_encoded'] = le_post.fit_transform(
            merged_df['post_author_faculty'].fillna('Unknown')
        )
        def append_features(row):
            try:
                uf = eval(row['user_features']) if isinstance(row['user_features'], str) else row['user_features']
                pf = eval(row['post_features']) if isinstance(row['post_features'], str) else row['post_features']
                uf = list(uf) + [float(row['user_faculty_encoded'])] if pd.notnull(row['user_faculty_encoded']) else list(uf) + [0.0]
                pf = list(pf) + [float(row['post_author_faculty_encoded'])] if pd.notnull(row['post_author_faculty_encoded']) else list(pf) + [0.0]
                return pd.Series({'user_features': uf, 'post_features': pf})
            except Exception as e:
                print(f"Error appending features: {e}")
                return pd.Series({'user_features': [0.0] * 10, 'post_features': [0.0] * 14})
        features_df = merged_df.apply(append_features, axis=1)
        merged_df['user_features'] = features_df['user_features']
        merged_df['post_features'] = features_df['post_features']
        merged_df = merged_df.sort_values('timestamp', ascending=False).drop_duplicates('post_id')
        print(f"Final records: {len(merged_df)}")
        return merged_df.head(5000)
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()
def load_and_prepare_data(train_path, posts_path):
    if SPARK_AVAILABLE:
        try:
            return load_and_prepare_data_spark(train_path, posts_path)
        except Exception as e:
            print(f"Spark data loading failed: {e}")
            print("Falling back to pandas...")
            return load_and_prepare_data_pandas(train_path, posts_path)
    return load_and_prepare_data_pandas(train_path, posts_path)
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
        'label': torch.stack([item['label'] for item in padded_batch])
    }
def train_model(model, train_loader, optimizer, device, num_epochs=3):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            try:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                user_features = batch['user_features'].to(device)
                post_features = batch['post_features'].to(device)
                interaction_features = batch['interaction_features'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids, attention_mask, user_features,
                              post_features, interaction_features)
                loss = criterion(outputs['prediction'].squeeze(), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
def compute_embeddings(model, tokenizer, df, device):
    model.eval()
    dataset = CTUConnectDataset(df, tokenizer, max_length=128)
    loader = DataLoader(dataset, batch_size=4, collate_fn=custom_collate_fn)
    user_embeddings = []
    post_embeddings = []
    content_embeddings = []
    for batch in tqdm(loader, desc="Computing embeddings"):
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            user_features = batch['user_features'].to(device)
            post_features = batch['post_features'].to(device)
            interaction_features = batch['interaction_features'].to(device)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask, user_features,
                              post_features, interaction_features)
            user_embeddings.extend(outputs['user_embedding'].cpu().numpy())
            post_embeddings.extend(outputs['post_embedding'].cpu().numpy())
            content_embeddings.extend(outputs['content_embedding'].cpu().numpy())
        except Exception as e:
            print(f"Error computing embeddings: {e}")
            continue
    df['user_embedding'] = user_embeddings
    df['post_embedding'] = post_embeddings
    df['content_embedding'] = content_embeddings
    return df
def build_faiss_index(df, index_path):
    if not FAISS_AVAILABLE:
        print("Faiss not available, skipping index creation")
        return None, None
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
        train_path = '/content/drive/MyDrive/data/ctu_connect_training.csv' if IN_COLAB else 'data/ctu_connect_training.csv'
        posts_path = '/content/drive/MyDrive/data/ctu_connect_posts.csv' if IN_COLAB else 'data/ctu_connect_posts.csv'
        embedding_path = '/content/drive/MyDrive/embeddings/daily_embeddings.parquet' if IN_COLAB else 'embeddings/daily_embeddings.parquet'
        index_path = '/content/drive/MyDrive/embeddings/faiss_index.bin' if IN_COLAB else 'embeddings/faiss_index.bin'
        model_path = '/content/drive/MyDrive/trained_models/ctu_model.pt' if IN_COLAB else 'trained_models/ctu_model.pt'
        data_df = load_and_prepare_data(train_path, posts_path)
        if len(data_df) == 0:
            print("No data loaded, skipping precomputation")
            return
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base') if TRANSFORMERS_AVAILABLE else SimpleTokenizer()
        model = PersonalizedCTUModel(
            vocab_size=30000,
            embedding_dim=128,
            user_feature_dim=10,  # Adjusted for user_faculty_encoded
            post_feature_dim=14,  # Adjusted for post_author_faculty_encoded
            interaction_feature_dim=8,
            dropout_rate=0.1
        ).to(device)
        data_df = compute_embeddings(model, tokenizer, data_df, device)
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        data_df.to_parquet(embedding_path)
        print(f"✅ Saved embeddings to {embedding_path}")
        build_faiss_index(data_df, index_path)
        print("Daily precomputation completed!")
    except Exception as e:
        print(f"Error in daily precomputation: {e}")
def main():
    try:
        if IN_COLAB:
            drive.mount('/content/drive')
        train_path = '/content/drive/MyDrive/data/ctu_connect_training.csv' if IN_COLAB else 'data/ctu_connect_training.csv'
        posts_path = '/content/drive/MyDrive/data/ctu_connect_posts.csv' if IN_COLAB else 'data/ctu_connect_posts.csv'
        model_dir = '/content/drive/MyDrive/trained_models/' if IN_COLAB else 'trained_models/'
        embedding_path = '/content/drive/MyDrive/embeddings/daily_embeddings.parquet' if IN_COLAB else 'embeddings/daily_embeddings.parquet'
        index_path = '/content/drive/MyDrive/embeddings/faiss_index.bin' if IN_COLAB else 'embeddings/faiss_index.bin'
        os.makedirs(model_dir, exist_ok=True)
        if not os.path.exists(train_path) or not os.path.exists(posts_path):
            print("❌ Data files not found. Please check your paths:")
            print(f"Training data: {train_path}")
            print(f"Posts data: {posts_path}")
            return
        print("📊 Loading and preparing data...")
        data_df = load_and_prepare_data(train_path, posts_path)
        if len(data_df) == 0:
            print("❌ No data loaded. Exiting.")
            return
        print(f"✅ Loaded {len(data_df)} records")
        train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42)
        print(f"📊 Train: {len(train_df)}, Test: {len(test_df)}")
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
        train_dataset = CTUConnectDataset(train_df, tokenizer, max_length=128)
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            collate_fn=custom_collate_fn
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        print("🚀 Starting training...")
        train_model(model, train_loader, optimizer, device, num_epochs=3)
        model_path = os.path.join(model_dir, 'ctu_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'vocab_size': 30000,
                'embedding_dim': 128,
                'user_feature_dim': 10,
                'post_feature_dim': 14,
                'interaction_feature_dim': 8,
                'dropout_rate': 0.1
            }
        }, model_path)
        print(f"✅ Model saved to {model_path}")
        print("📊 Computing embeddings...")
        data_df = compute_embeddings(model, tokenizer, data_df, device)
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        data_df.to_parquet(embedding_path)
        print(f"✅ Saved embeddings to {embedding_path}")
        build_faiss_index(data_df, index_path)
        print("🎉 Training and precomputation completed!")
        schedule.every().day.at("09:00").do(daily_precompute)
        while True:
            schedule.run_pending()
            time.sleep(60)
    except Exception as e:
        print(f"❌ Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if SPARK_AVAILABLE:
            try:
                spark.stop()
                print("✅ Spark session stopped")
            except:
                pass
if __name__ == "__main__":
    main()