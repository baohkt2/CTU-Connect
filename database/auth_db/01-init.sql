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
                                                  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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
                                              id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                                              token VARCHAR(255) NOT NULL UNIQUE,
                                              user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE, -- Đã đổi sang UUID
                                              expiry_date TIMESTAMP NOT NULL,
                                              created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ---
-- Create table for password reset tokens
-- ---
CREATE TABLE IF NOT EXISTS password_reset_tokens (
                                                     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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
