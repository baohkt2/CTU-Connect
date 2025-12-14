-- Friend Recommendation Schema
-- Created: December 2024
-- Purpose: Support ML-enhanced friend recommendation system

SET search_path TO recommend, public;

-- ============================================================
-- Table: user_embeddings (EXTEND existing table)
-- Add columns for friend recommendation
-- ============================================================
ALTER TABLE recommend.user_embeddings 
ADD COLUMN IF NOT EXISTS major VARCHAR(100),
ADD COLUMN IF NOT EXISTS faculty VARCHAR(100),
ADD COLUMN IF NOT EXISTS bio TEXT,
ADD COLUMN IF NOT EXISTS interests TEXT[],
ADD COLUMN IF NOT EXISTS batch_year VARCHAR(20),
ADD COLUMN IF NOT EXISTS skills TEXT[];

-- ============================================================
-- Table: friend_recommendation_log
-- Logs friend suggestions for analytics and feedback learning
-- ============================================================
CREATE TABLE IF NOT EXISTS recommend.friend_recommendation_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    recommended_user_id VARCHAR(255) NOT NULL,
    
    -- Scoring details
    relevance_score REAL NOT NULL,
    content_similarity REAL,
    mutual_friends_score REAL,
    academic_score REAL,
    activity_score REAL,
    recency_score REAL,
    
    -- Suggestion metadata
    suggestion_type VARCHAR(50) NOT NULL,
    suggestion_reason TEXT,
    rank_position INTEGER,
    
    -- Timestamps for funnel tracking
    shown_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    clicked_at TIMESTAMP WITH TIME ZONE,
    friend_request_sent_at TIMESTAMP WITH TIME ZONE,
    accepted_at TIMESTAMP WITH TIME ZONE,
    rejected_at TIMESTAMP WITH TIME ZONE,
    dismissed_at TIMESTAMP WITH TIME ZONE,
    
    -- Request context
    request_source VARCHAR(50), -- 'feed', 'profile', 'search', 'notification'
    session_id VARCHAR(255),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for friend_recommendation_log
CREATE INDEX IF NOT EXISTS idx_friend_log_user_id 
ON recommend.friend_recommendation_log(user_id);

CREATE INDEX IF NOT EXISTS idx_friend_log_recommended_user 
ON recommend.friend_recommendation_log(recommended_user_id);

CREATE INDEX IF NOT EXISTS idx_friend_log_shown_at 
ON recommend.friend_recommendation_log(shown_at DESC);

CREATE INDEX IF NOT EXISTS idx_friend_log_suggestion_type 
ON recommend.friend_recommendation_log(suggestion_type);

CREATE INDEX IF NOT EXISTS idx_friend_log_user_recommended 
ON recommend.friend_recommendation_log(user_id, recommended_user_id);

-- ============================================================
-- Table: user_activity_score
-- Tracks user engagement metrics for friend recommendation
-- ============================================================
CREATE TABLE IF NOT EXISTS recommend.user_activity_score (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) UNIQUE NOT NULL,
    
    -- Activity counts
    post_count INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    like_count INTEGER DEFAULT 0,
    share_count INTEGER DEFAULT 0,
    friend_count INTEGER DEFAULT 0,
    
    -- Engagement metrics
    posts_last_30_days INTEGER DEFAULT 0,
    comments_last_30_days INTEGER DEFAULT 0,
    likes_last_30_days INTEGER DEFAULT 0,
    
    -- Calculated scores
    activity_score REAL DEFAULT 0.0,
    engagement_rate REAL DEFAULT 0.0,
    
    -- Timestamps
    last_activity_at TIMESTAMP WITH TIME ZONE,
    last_post_at TIMESTAMP WITH TIME ZONE,
    last_login_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for user_activity_score
CREATE INDEX IF NOT EXISTS idx_user_activity_user_id 
ON recommend.user_activity_score(user_id);

CREATE INDEX IF NOT EXISTS idx_user_activity_score 
ON recommend.user_activity_score(activity_score DESC);

CREATE INDEX IF NOT EXISTS idx_user_activity_last_activity 
ON recommend.user_activity_score(last_activity_at DESC);

-- ============================================================
-- Table: user_interaction_graph
-- Stores user-to-user interaction strength (denormalized from Neo4j)
-- ============================================================
CREATE TABLE IF NOT EXISTS recommend.user_interaction_graph (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    target_user_id VARCHAR(255) NOT NULL,
    
    -- Interaction counts
    profile_views INTEGER DEFAULT 0,
    message_count INTEGER DEFAULT 0,
    post_likes INTEGER DEFAULT 0,
    post_comments INTEGER DEFAULT 0,
    
    -- Calculated interaction strength
    interaction_strength REAL DEFAULT 0.0,
    
    -- Timestamps
    first_interaction_at TIMESTAMP WITH TIME ZONE,
    last_interaction_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT uk_user_interaction UNIQUE(user_id, target_user_id)
);

-- Indexes for user_interaction_graph
CREATE INDEX IF NOT EXISTS idx_interaction_user_id 
ON recommend.user_interaction_graph(user_id);

CREATE INDEX IF NOT EXISTS idx_interaction_target_user 
ON recommend.user_interaction_graph(target_user_id);

CREATE INDEX IF NOT EXISTS idx_interaction_strength 
ON recommend.user_interaction_graph(interaction_strength DESC);

-- ============================================================
-- Table: friend_recommendation_metrics
-- Aggregated metrics for monitoring and optimization
-- ============================================================
CREATE TABLE IF NOT EXISTS recommend.friend_recommendation_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_date DATE NOT NULL,
    
    -- Volume metrics
    total_suggestions_shown INTEGER DEFAULT 0,
    total_suggestions_clicked INTEGER DEFAULT 0,
    total_friend_requests_sent INTEGER DEFAULT 0,
    total_friend_requests_accepted INTEGER DEFAULT 0,
    
    -- Rate metrics
    click_through_rate REAL DEFAULT 0.0,
    request_rate REAL DEFAULT 0.0,
    acceptance_rate REAL DEFAULT 0.0,
    
    -- Performance metrics
    avg_latency_ms INTEGER,
    p99_latency_ms INTEGER,
    cache_hit_rate REAL DEFAULT 0.0,
    ml_success_rate REAL DEFAULT 0.0,
    fallback_rate REAL DEFAULT 0.0,
    
    -- Breakdown by type
    mutual_friends_suggestions INTEGER DEFAULT 0,
    academic_suggestions INTEGER DEFAULT 0,
    content_similarity_suggestions INTEGER DEFAULT 0,
    activity_based_suggestions INTEGER DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT uk_metrics_date UNIQUE(metric_date)
);

-- Index for metrics
CREATE INDEX IF NOT EXISTS idx_metrics_date 
ON recommend.friend_recommendation_metrics(metric_date DESC);

-- ============================================================
-- Function: Update activity score
-- ============================================================
CREATE OR REPLACE FUNCTION recommend.calculate_activity_score(
    p_post_count INTEGER,
    p_comment_count INTEGER,
    p_like_count INTEGER,
    p_posts_30d INTEGER,
    p_last_activity TIMESTAMP WITH TIME ZONE
) RETURNS REAL AS $$
DECLARE
    v_base_score REAL;
    v_recency_bonus REAL;
    v_final_score REAL;
BEGIN
    -- Base score from activity counts
    v_base_score := (
        LEAST(p_post_count, 100) * 0.3 +
        LEAST(p_comment_count, 500) * 0.1 +
        LEAST(p_like_count, 1000) * 0.05 +
        LEAST(p_posts_30d, 30) * 0.5
    ) / 100.0;
    
    -- Recency bonus (decays over time)
    IF p_last_activity IS NOT NULL THEN
        v_recency_bonus := GREATEST(0, 1 - EXTRACT(EPOCH FROM (NOW() - p_last_activity)) / (30 * 24 * 3600));
    ELSE
        v_recency_bonus := 0;
    END IF;
    
    -- Final score (0-1 range)
    v_final_score := LEAST(1.0, v_base_score * 0.7 + v_recency_bonus * 0.3);
    
    RETURN v_final_score;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- Trigger: Auto-update updated_at
-- ============================================================
CREATE OR REPLACE FUNCTION recommend.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to tables
DROP TRIGGER IF EXISTS trg_user_activity_updated_at ON recommend.user_activity_score;
CREATE TRIGGER trg_user_activity_updated_at
    BEFORE UPDATE ON recommend.user_activity_score
    FOR EACH ROW
    EXECUTE FUNCTION recommend.update_updated_at_column();

DROP TRIGGER IF EXISTS trg_user_interaction_updated_at ON recommend.user_interaction_graph;
CREATE TRIGGER trg_user_interaction_updated_at
    BEFORE UPDATE ON recommend.user_interaction_graph
    FOR EACH ROW
    EXECUTE FUNCTION recommend.update_updated_at_column();

-- ============================================================
-- Comments for documentation
-- ============================================================
COMMENT ON TABLE recommend.friend_recommendation_log IS 
'Logs all friend suggestions shown to users for analytics and ML feedback';

COMMENT ON TABLE recommend.user_activity_score IS 
'Stores calculated activity scores for users, used in friend ranking';

COMMENT ON TABLE recommend.user_interaction_graph IS 
'Denormalized user interaction data for fast similarity queries';

COMMENT ON TABLE recommend.friend_recommendation_metrics IS 
'Daily aggregated metrics for monitoring friend recommendation system';

COMMENT ON COLUMN recommend.friend_recommendation_log.suggestion_type IS 
'Type: MUTUAL_FRIENDS, ACADEMIC_CONNECTION, CONTENT_SIMILARITY, ACTIVITY_BASED, FRIENDS_OF_FRIENDS';

COMMENT ON COLUMN recommend.user_activity_score.activity_score IS 
'Normalized score 0-1, calculated from activity metrics and recency';
