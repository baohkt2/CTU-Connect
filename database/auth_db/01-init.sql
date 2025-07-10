-- auth_db schema (this runs automatically in auth_db due to POSTGRES_DB setting)

-- ---
-- Drop existing tables and types to recreate them safely (if running this script multiple times)
-- ---
DROP TABLE IF EXISTS email_verification CASCADE;
DROP TABLE IF EXISTS refresh_tokens CASCADE;
DROP TABLE IF EXISTS password_reset_tokens CASCADE;
DROP TABLE IF EXISTS users CASCADE;
-- DROP EXTENSION IF EXISTS "uuid-ossp"; -- Uncomment this if you need to recreate the extension for testing purposes

-- ---
-- Create Extension for UUID (if not already enabled and using PostgreSQL < 13)
-- ---
-- CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; -- Uncomment and run this if your PostgreSQL version is below 13
-- and you use uuid_generate_v4() instead of gen_random_uuid()

-- ---
-- Create table for users
-- ---
CREATE TABLE IF NOT EXISTS users (
                                     id UUID PRIMARY KEY DEFAULT gen_random_uuid(), -- Sử dụng UUID và tự động sinh
                                     email VARCHAR(50) NOT NULL UNIQUE,
                                     username VARCHAR(25) NOT NULL UNIQUE,
                                     password VARCHAR(255) NOT NULL,
                                     role VARCHAR(20) NOT NULL DEFAULT 'USER',
                                     created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                                     updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                                     is_active BOOLEAN NOT NULL DEFAULT TRUE
);

-- ---
-- Create table for email verification
-- ---
CREATE TABLE IF NOT EXISTS email_verification (
                                                  id BIGSERIAL PRIMARY KEY,
                                                  token VARCHAR(255) NOT NULL UNIQUE,
                                                  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE, -- Đã đổi sang UUID
                                                  expiry_date BIGINT NOT NULL, -- Keep as BIGINT for Unix epoch milliseconds
                                                  is_verified BOOLEAN NOT NULL DEFAULT FALSE,
                                                  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ---
-- Create table for refresh tokens
-- ---
CREATE TABLE IF NOT EXISTS refresh_tokens (
                                              id BIGSERIAL PRIMARY KEY,
                                              token VARCHAR(255) NOT NULL UNIQUE,
                                              user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE, -- Đã đổi sang UUID
                                              expiry_date TIMESTAMP NOT NULL,
                                              created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ---
-- Create table for password reset tokens
-- ---
CREATE TABLE IF NOT EXISTS password_reset_tokens (
                                                     id BIGSERIAL PRIMARY KEY,
                                                     token VARCHAR(255) NOT NULL UNIQUE,
                                                     user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE, -- Đã đổi sang UUID
                                                     expiry_date TIMESTAMP NOT NULL,
                                                     created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ---
-- Create indexes for faster token lookups
-- ---
CREATE INDEX IF NOT EXISTS idx_email_verification_token ON email_verification(token);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_token ON refresh_tokens(token);
CREATE INDEX IF NOT EXISTS idx_password_reset_tokens_token ON password_reset_tokens(token);

-- ---
-- Insert sample users
-- Note: These passwords are BCrypt-hashed, all are 'password123'
-- We'll capture the generated UUIDs to use for foreign key inserts
-- ---
-- Sử dụng WITH để lấy UUID của các user vừa được chèn
WITH inserted_users AS (
    INSERT INTO users (email, username, password, role, created_at, updated_at, is_active)
        VALUES
            ('admin@ctuconnect.edu.vn', 'admin', '$2a$10$dXJ3SW6G7P50lGmMkkmwe.20cQQubK3.HZWzG3YB1tlRy.fqvM/BG', 'ADMIN', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, TRUE),
            ('student1@student.ctu.edu.vn', 'student1', '$2a$10$dXJ3SW6G7P50lGmMkkmwe.20cQQubK3.HZWzG3YB1tlRy.fqvM/BG', 'STUDENT', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, TRUE),
            ('student2@student.ctu.edu.vn', 'student2', '$2a$10$dXJ3SW6G7P50lGmMkkmwe.20cQQubK3.HZWzG3YB1tlRy.fqvM/BG', 'STUDENT', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, TRUE),
            ('teacher1@ctu.edu.vn', 'teacher1', '$2a$10$dXJ3SW6G7P50lGmMkkmwe.20cQQubK3.HZWzG3YB1tlRy.fqvM/BG', 'TEACHER', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, TRUE),
            ('user1@example.com', 'user1', '$2a$10$dXJ3SW6G7P50lGmMkkmwe.20cQQubK3.HZWzG3YB1tlRy.fqvM/BG', 'USER', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, TRUE)
        ON CONFLICT (email) DO NOTHING
        RETURNING id, email -- Lấy ID và email của các user vừa được chèn
)
-- ---
-- Insert sample verified emails for the above users
-- ---
INSERT INTO email_verification (token, user_id, expiry_date, is_verified, created_at)
SELECT
    CASE i.email
        WHEN 'admin@ctuconnect.edu.vn' THEN 'verified-token-admin'
        WHEN 'student1@student.ctu.edu.vn' THEN 'verified-token-student1'
        WHEN 'student2@student.ctu.edu.vn' THEN 'verified-token-student2'
        WHEN 'teacher1@ctu.edu.vn' THEN 'verified-token-teacher'
        WHEN 'user1@example.com' THEN 'pending-token-user1'
        END,
    i.id, -- Sử dụng ID UUID từ inserted_users
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP + INTERVAL '1 day')) * 1000,
    CASE i.email
        WHEN 'user1@example.com' THEN FALSE
        ELSE TRUE
        END,
    CURRENT_TIMESTAMP
FROM inserted_users i
ON CONFLICT (token) DO NOTHING;