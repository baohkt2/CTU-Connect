-- CTU Connect Recommendation Service Database Initialization
-- PostgreSQL with pgvector extension

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Post Embeddings Table
CREATE TABLE IF NOT EXISTS post_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id VARCHAR(255) UNIQUE NOT NULL,
    author_id VARCHAR(255) NOT NULL,
    content TEXT,
    embedding_vector vector(768),
    academic_score FLOAT DEFAULT 0,
    academic_category VARCHAR(50),
    popularity_score FLOAT DEFAULT 0,
    like_count INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    share_count INTEGER DEFAULT 0,
    view_count INTEGER DEFAULT 0,
    faculty VARCHAR(100),
    major VARCHAR(100),
    tags TEXT[],
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    embedding_updated_at TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_post_id ON post_embeddings(post_id);
CREATE INDEX IF NOT EXISTS idx_author_id ON post_embeddings(author_id);
CREATE INDEX IF NOT EXISTS idx_created_at ON post_embeddings(created_at);
CREATE INDEX IF NOT EXISTS idx_academic_score ON post_embeddings(academic_score);
CREATE INDEX IF NOT EXISTS idx_embedding_vector ON post_embeddings 
USING ivfflat (embedding_vector vector_cosine_ops) WITH (lists = 100);

-- User Feedback Table
CREATE TABLE IF NOT EXISTS user_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    post_id VARCHAR(255) NOT NULL,
    feedback_type VARCHAR(50) NOT NULL,
    feedback_value FLOAT NOT NULL,
    session_id VARCHAR(100),
    context JSONB,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_user_post ON user_feedback(user_id, post_id);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON user_feedback(feedback_type);

-- Recommendation Cache Table
CREATE TABLE IF NOT EXISTS recommendation_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) UNIQUE NOT NULL,
    post_ids TEXT[],
    scores REAL[],
    ab_variant VARCHAR(50),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_user_cache ON recommendation_cache(user_id);
