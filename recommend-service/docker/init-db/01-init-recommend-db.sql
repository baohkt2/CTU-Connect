-- Initialize Recommendation Service Database
-- Created: December 2024

-- Create database if not exists (already created by POSTGRES_DB env var)
-- This script runs inside recommend_db

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schema
CREATE SCHEMA IF NOT EXISTS recommend;

-- Set search path
SET search_path TO recommend, public;

-- Table: post_embeddings
-- Stores embeddings for posts
CREATE TABLE IF NOT EXISTS recommend.post_embeddings (
    id SERIAL PRIMARY KEY,
    post_id VARCHAR(255) UNIQUE NOT NULL,
    embedding FLOAT[] NOT NULL,
    dimension INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT embedding_dimension_check CHECK (dimension > 0)
);

-- Create indexes for post_embeddings
CREATE INDEX idx_post_embeddings_post_id ON recommend.post_embeddings(post_id);
CREATE INDEX idx_post_embeddings_created_at ON recommend.post_embeddings(created_at DESC);
CREATE INDEX idx_post_embeddings_metadata ON recommend.post_embeddings USING gin(metadata);

-- Table: user_embeddings
-- Stores embeddings for user profiles
CREATE TABLE IF NOT EXISTS recommend.user_embeddings (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    embedding FLOAT[] NOT NULL,
    dimension INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb,
    CONSTRAINT user_embedding_dimension_check CHECK (dimension > 0)
);

-- Create indexes for user_embeddings
CREATE INDEX idx_user_embeddings_user_id ON recommend.user_embeddings(user_id);
CREATE INDEX idx_user_embeddings_updated_at ON recommend.user_embeddings(updated_at DESC);
CREATE INDEX idx_user_embeddings_metadata ON recommend.user_embeddings USING gin(metadata);

-- Table: recommendation_cache
-- Caches recommendation results
CREATE TABLE IF NOT EXISTS recommend.recommendation_cache (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    cache_key VARCHAR(500) NOT NULL,
    recommendations JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    CONSTRAINT unique_cache_key UNIQUE(user_id, cache_key)
);

-- Create indexes for recommendation_cache
CREATE INDEX idx_recommendation_cache_user_id ON recommend.recommendation_cache(user_id);
CREATE INDEX idx_recommendation_cache_expires_at ON recommend.recommendation_cache(expires_at);
CREATE INDEX idx_recommendation_cache_created_at ON recommend.recommendation_cache(created_at DESC);

-- Table: recommendation_logs
-- Logs recommendation requests and results for analytics
CREATE TABLE IF NOT EXISTS recommend.recommendation_logs (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    request_type VARCHAR(100) NOT NULL,
    request_params JSONB,
    result_count INTEGER,
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for recommendation_logs
CREATE INDEX idx_recommendation_logs_user_id ON recommend.recommendation_logs(user_id);
CREATE INDEX idx_recommendation_logs_created_at ON recommend.recommendation_logs(created_at DESC);
CREATE INDEX idx_recommendation_logs_request_type ON recommend.recommendation_logs(request_type);

-- Table: similarity_cache
-- Caches similarity computations
CREATE TABLE IF NOT EXISTS recommend.similarity_cache (
    id SERIAL PRIMARY KEY,
    key_hash VARCHAR(64) UNIQUE NOT NULL,
    similarity_score FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 1
);

-- Create indexes for similarity_cache
CREATE INDEX idx_similarity_cache_key_hash ON recommend.similarity_cache(key_hash);
CREATE INDEX idx_similarity_cache_accessed_at ON recommend.similarity_cache(accessed_at DESC);

-- Function: Update updated_at timestamp
CREATE OR REPLACE FUNCTION recommend.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for auto-updating updated_at
CREATE TRIGGER update_post_embeddings_updated_at 
    BEFORE UPDATE ON recommend.post_embeddings
    FOR EACH ROW EXECUTE FUNCTION recommend.update_updated_at_column();

CREATE TRIGGER update_user_embeddings_updated_at 
    BEFORE UPDATE ON recommend.user_embeddings
    FOR EACH ROW EXECUTE FUNCTION recommend.update_updated_at_column();

-- Function: Clean expired cache
CREATE OR REPLACE FUNCTION recommend.clean_expired_cache()
RETURNS void AS $$
BEGIN
    DELETE FROM recommend.recommendation_cache
    WHERE expires_at < CURRENT_TIMESTAMP;
    
    -- Also clean old similarity cache (keep only last 7 days)
    DELETE FROM recommend.similarity_cache
    WHERE accessed_at < CURRENT_TIMESTAMP - INTERVAL '7 days';
END;
$$ language 'plpgsql';

-- Function: Update similarity cache access
CREATE OR REPLACE FUNCTION recommend.update_similarity_cache_access()
RETURNS TRIGGER AS $$
BEGIN
    NEW.accessed_at = CURRENT_TIMESTAMP;
    NEW.access_count = OLD.access_count + 1;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for similarity cache access tracking
CREATE TRIGGER update_similarity_cache_access_trigger
    BEFORE UPDATE ON recommend.similarity_cache
    FOR EACH ROW EXECUTE FUNCTION recommend.update_similarity_cache_access();

-- Create view for recent recommendations
CREATE OR REPLACE VIEW recommend.recent_recommendations AS
SELECT 
    rl.user_id,
    rl.request_type,
    rl.result_count,
    rl.processing_time_ms,
    rl.created_at
FROM recommend.recommendation_logs rl
WHERE rl.created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
ORDER BY rl.created_at DESC;

-- Create view for embedding statistics
CREATE OR REPLACE VIEW recommend.embedding_stats AS
SELECT 
    'posts' as type,
    COUNT(*) as total_count,
    AVG(dimension) as avg_dimension,
    MIN(created_at) as oldest,
    MAX(created_at) as newest
FROM recommend.post_embeddings
UNION ALL
SELECT 
    'users' as type,
    COUNT(*) as total_count,
    AVG(dimension) as avg_dimension,
    MIN(created_at) as oldest,
    MAX(created_at) as newest
FROM recommend.user_embeddings;

-- Grant permissions
GRANT USAGE ON SCHEMA recommend TO recommend_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA recommend TO recommend_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA recommend TO recommend_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA recommend TO recommend_user;

-- Create maintenance function to be run periodically
CREATE OR REPLACE FUNCTION recommend.maintenance()
RETURNS void AS $$
BEGIN
    -- Clean expired cache
    PERFORM recommend.clean_expired_cache();
    
    -- Analyze tables for better query planning
    ANALYZE recommend.post_embeddings;
    ANALYZE recommend.user_embeddings;
    ANALYZE recommend.recommendation_cache;
    ANALYZE recommend.recommendation_logs;
    ANALYZE recommend.similarity_cache;
    
    RAISE NOTICE 'Maintenance completed successfully';
END;
$$ language 'plpgsql';

-- Log completion
DO $$
BEGIN
    RAISE NOTICE 'Recommendation database initialized successfully';
    RAISE NOTICE 'Schema: recommend';
    RAISE NOTICE 'Tables created: 5';
    RAISE NOTICE 'Views created: 2';
    RAISE NOTICE 'Functions created: 4';
END $$;
