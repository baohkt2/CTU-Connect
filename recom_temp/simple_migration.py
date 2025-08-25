#!/usr/bin/env python3
"""
Simple data migration script - chỉ đưa dữ liệu vào MongoDB và PostgreSQL
Bỏ qua Neo4j để tránh lỗi kết nối
"""

import pandas as pd
import pymongo
from pymongo import MongoClient
import psycopg2
from psycopg2.extras import execute_values
import logging
import sys
import os
from datetime import datetime

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configurations
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = "recommendation_db"

POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "database": os.getenv("POSTGRES_DB", "auth_db"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres")
}

def load_csv_data():
    """Load data from CSV files"""
    try:
        users_df = pd.read_csv("generated_users.csv")
        posts_df = pd.read_csv("generated_posts.csv")
        logger.info(f"📊 Loaded {len(users_df)} users and {len(posts_df)} posts from CSV")
        return users_df, posts_df
    except FileNotFoundError as e:
        logger.error(f"❌ CSV file not found: {e}")
        logger.info("💡 Generating new data...")
        return generate_sample_data()

def generate_sample_data():
    """Generate sample data if CSV files don't exist"""
    import random
    from faker import Faker

    fake = Faker('vi_VN')  # Vietnamese locale

    # Generate users
    users_data = []
    for i in range(100):  # Reduced for faster testing
        user = {
            'user_id': f"user_{i+1:03d}",
            'username': fake.user_name(),
            'email': fake.email(),
            'full_name': fake.name(),
            'major': random.choice(['Computer Science', 'Engineering', 'Business', 'Medicine', 'Arts']),
            'year': random.choice([1, 2, 3, 4]),
            'created_at': fake.date_time_between(start_date='-2y', end_date='now').isoformat()
        }
        users_data.append(user)

    # Generate posts
    posts_data = []
    for i in range(300):  # Reduced for faster testing
        post = {
            'post_id': f"post_{i+1:03d}",
            'user_id': random.choice([u['user_id'] for u in users_data]),
            'title': fake.sentence(nb_words=6),
            'content': fake.text(max_nb_chars=500),
            'category': random.choice(['academic', 'social', 'announcement', 'question', 'discussion']),
            'likes': random.randint(0, 50),
            'comments': random.randint(0, 20),
            'created_at': fake.date_time_between(start_date='-1y', end_date='now').isoformat()
        }
        posts_data.append(post)

    users_df = pd.DataFrame(users_data)
    posts_df = pd.DataFrame(posts_data)

    # Save to CSV for backup
    users_df.to_csv("generated_users.csv", index=False)
    posts_df.to_csv("generated_posts.csv", index=False)

    logger.info(f"✅ Generated {len(users_df)} users and {len(posts_df)} posts")
    return users_df, posts_df

def migrate_to_postgres(users_df):
    """Migrate users to PostgreSQL"""
    try:
        logger.info("🔄 Connecting to PostgreSQL...")
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cur = conn.cursor()

        # Create table if not exists
        create_table_query = """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(50) UNIQUE NOT NULL,
            username VARCHAR(100) NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            full_name VARCHAR(255),
            major VARCHAR(100),
            year INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cur.execute(create_table_query)

        # Clear existing data
        cur.execute("DELETE FROM users;")

        # Prepare data for insertion
        user_records = []
        for _, user in users_df.iterrows():
            record = (
                user['user_id'],
                user['username'],
                user['email'],
                user['full_name'],
                user['major'],
                user['year'],
                user['created_at']
            )
            user_records.append(record)

        # Bulk insert
        insert_query = """
        INSERT INTO users (user_id, username, email, full_name, major, year, created_at)
        VALUES %s
        """
        execute_values(cur, insert_query, user_records)

        conn.commit()
        cur.close()
        conn.close()

        logger.info(f"✅ Successfully migrated {len(user_records)} users to PostgreSQL")
        return True

    except Exception as e:
        logger.error(f"❌ Error migrating to PostgreSQL: {e}")
        return False

def migrate_to_mongodb(posts_df):
    """Migrate posts to MongoDB"""
    try:
        logger.info("🔄 Connecting to MongoDB...")
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)

        # Test connection
        client.admin.command('ismaster')

        db = client[MONGO_DB]
        posts_collection = db.posts

        # Clear existing data
        posts_collection.delete_many({})

        # Convert DataFrame to dict records
        posts_records = posts_df.to_dict('records')

        # Convert datetime strings to proper format
        for post in posts_records:
            if 'created_at' in post and isinstance(post['created_at'], str):
                try:
                    post['created_at'] = datetime.fromisoformat(post['created_at'])
                except:
                    post['created_at'] = datetime.now()

        # Bulk insert
        result = posts_collection.insert_many(posts_records)

        client.close()

        logger.info(f"✅ Successfully migrated {len(result.inserted_ids)} posts to MongoDB")
        return True

    except Exception as e:
        logger.error(f"❌ Error migrating to MongoDB: {e}")
        return False

def main():
    """Main migration process"""
    logger.info("🚀 Starting simple data migration...")

    # Load or generate data
    users_df, posts_df = load_csv_data()

    success_count = 0

    # Migrate to PostgreSQL
    logger.info("\n📊 Migrating users to PostgreSQL...")
    if migrate_to_postgres(users_df):
        success_count += 1

    # Migrate to MongoDB
    logger.info("\n📊 Migrating posts to MongoDB...")
    if migrate_to_mongodb(posts_df):
        success_count += 1

    # Summary
    logger.info(f"\n🎉 Migration completed!")
    logger.info(f"📈 Successfully migrated to {success_count}/2 databases")

    if success_count == 2:
        logger.info("✅ All migrations successful!")
        return 0
    else:
        logger.warning("⚠️ Some migrations failed. Check logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
