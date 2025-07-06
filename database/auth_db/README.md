# Authentication Service Database

This directory contains initialization scripts for the authentication service database (auth_db).

## Overview

The auth_db is a PostgreSQL database that manages user authentication and authorization for the CTU Connect platform. It stores user accounts, verification tokens, refresh tokens, and password reset tokens.

## Database Schema

### Tables

1. **users** - Core user accounts
   - `id`: Primary key
   - `email`: User's email address (unique)
   - `username`: User's chosen username (unique)
   - `password`: BCrypt hashed password
   - `role`: User role (ADMIN, STUDENT, TEACHER, USER)
   - `created_at`: Account creation timestamp
   - `updated_at`: Last update timestamp
   - `is_active`: Account status flag

2. **email_verification** - Email verification tokens
   - `id`: Primary key
   - `token`: Unique verification token
   - `user_id`: Foreign key to users table
   - `expiry_date`: Token expiration timestamp
   - `is_verified`: Verification status
   - `created_at`: Token creation timestamp

3. **refresh_tokens** - JWT refresh tokens
   - `id`: Primary key
   - `token`: Unique refresh token
   - `user_id`: Foreign key to users table
   - `expiry_date`: Token expiration timestamp
   - `created_at`: Token creation timestamp

4. **password_reset_tokens** - Password reset tokens
   - `id`: Primary key
   - `token`: Unique reset token
   - `user_id`: Foreign key to users table
   - `expiry_date`: Token expiration timestamp
   - `created_at`: Token creation timestamp

### Indexes

The database includes indexes on token fields for faster lookups:
- `idx_email_verification_token`
- `idx_refresh_tokens_token`
- `idx_password_reset_tokens_token`

## Sample Data

The initialization script includes sample users for development and testing:
- Admin user: admin@ctuconnect.edu.vn / password123
- Student users: student1@student.ctu.edu.vn, student2@student.ctu.edu.vn / password123
- Teacher user: teacher1@ctu.edu.vn / password123
- Regular user: user1@example.com / password123

## Integration with Microservices

This database is specifically for the auth-service microservice and contains only authentication and authorization related data. User profile data and relationships are stored in the user-service's Neo4j database.

# PostgreSQL Initialization Scripts

This directory contains SQL scripts that will be automatically executed when the PostgreSQL container starts for the first time.

Scripts are executed in alphabetical order, so they are prefixed with numbers (01-, 02-, etc.) to ensure proper execution sequence.

## Files

- `01-init.sql` - Creates the users_db database and sets up tables for auth_db

## How it works

1. The PostgreSQL container creates the `auth_db` based on the `POSTGRES_DB` environment variable
2. The script `01-init.sql` runs and creates both the database structure for auth_db and creates the users_db database

## Note

These scripts only run when the database is first initialized. If you need to apply changes to an existing database, you'll need to use a migration tool like Flyway or Liquibase, or manually apply the changes.

If you change the initialization scripts and want them to take effect, you'll need to delete the volume with `docker-compose down -v` and restart.
