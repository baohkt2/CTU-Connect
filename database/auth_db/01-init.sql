-- auth_db schema (this runs automatically in auth_db due to POSTGRES_DB setting)
-- Create table for users first (because email_verification references it)
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    email VARCHAR(50) NOT NULL UNIQUE,
    username VARCHAR(25) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'USER',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

-- Create table for email verification
CREATE TABLE IF NOT EXISTS email_verification (
    id BIGSERIAL PRIMARY KEY,
    token VARCHAR(255) NOT NULL UNIQUE,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expiry_date BIGINT NOT NULL,
    is_verified BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create table for refresh tokens
CREATE TABLE IF NOT EXISTS refresh_tokens (
    id BIGSERIAL PRIMARY KEY,
    token VARCHAR(255) NOT NULL UNIQUE,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expiry_date TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create table for password reset tokens
CREATE TABLE IF NOT EXISTS password_reset_tokens (
    id BIGSERIAL PRIMARY KEY,
    token VARCHAR(255) NOT NULL UNIQUE,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expiry_date TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster token lookups
CREATE INDEX IF NOT EXISTS idx_email_verification_token ON email_verification(token);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_token ON refresh_tokens(token);
CREATE INDEX IF NOT EXISTS idx_password_reset_tokens_token ON password_reset_tokens(token);

-- Optional: Insert sample data for auth_db (uncomment to use)
-- INSERT INTO user_entity (email, username, password, created_at, updated_at, is_active)
-- VALUES ('admin@example.com', 'admin', '$2a$12$dPb.ZCsJh/YhGGosxQ1KAedVmtaXIZB9kMQJaEKsW6/V3fmJE5hs.', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, TRUE);

-- Optional: Insert sample data for users_db (uncomment to use)
-- INSERT INTO user_entity (email, username, password, created_at, updated_at, is_active)
-- VALUES ('user@example.com', 'testuser', '$2a$12$dPb.ZCsJh/YhGGosxQ1KAedVmtaXIZB9kMQJaEKsW6/V3fmJE5hs.', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, TRUE);

-- Insert sample users
-- Note: These passwords are BCrypt-hashed, all are 'password123'
INSERT INTO users (email, username, password, role, created_at, updated_at, is_active)
VALUES
    ('admin@ctuconnect.edu.vn', 'admin', '$2a$10$dXJ3SW6G7P50lGmMkkmwe.20cQQubK3.HZWzG3YB1tlRy.fqvM/BG', 'ADMIN', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, TRUE),
    ('student1@student.ctu.edu.vn', 'student1', '$2a$10$dXJ3SW6G7P50lGmMkkmwe.20cQQubK3.HZWzG3YB1tlRy.fqvM/BG', 'STUDENT', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, TRUE),
    ('student2@student.ctu.edu.vn', 'student2', '$2a$10$dXJ3SW6G7P50lGmMkkmwe.20cQQubK3.HZWzG3YB1tlRy.fqvM/BG', 'STUDENT', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, TRUE),
    ('teacher1@ctu.edu.vn', 'teacher1', '$2a$10$dXJ3SW6G7P50lGmMkkmwe.20cQQubK3.HZWzG3YB1tlRy.fqvM/BG', 'TEACHER', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, TRUE),
    ('user1@example.com', 'user1', '$2a$10$dXJ3SW6G7P50lGmMkkmwe.20cQQubK3.HZWzG3YB1tlRy.fqvM/BG', 'USER', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, TRUE)
ON CONFLICT (email) DO NOTHING;

-- Insert sample verified emails for the above users
INSERT INTO email_verification (token, user_id, expiry_date, is_verified, created_at)
VALUES
    ('verified-token-admin', 1, extract(epoch from (CURRENT_TIMESTAMP + interval '1 day')) * 1000, TRUE, CURRENT_TIMESTAMP),
    ('verified-token-student1', 2, extract(epoch from (CURRENT_TIMESTAMP + interval '1 day')) * 1000, TRUE, CURRENT_TIMESTAMP),
    ('verified-token-student2', 3, extract(epoch from (CURRENT_TIMESTAMP + interval '1 day')) * 1000, TRUE, CURRENT_TIMESTAMP),
    ('verified-token-teacher', 4, extract(epoch from (CURRENT_TIMESTAMP + interval '1 day')) * 1000, TRUE, CURRENT_TIMESTAMP),
    ('pending-token-user1', 5, extract(epoch from (CURRENT_TIMESTAMP + interval '1 day')) * 1000, FALSE, CURRENT_TIMESTAMP)
ON CONFLICT DO NOTHING;
