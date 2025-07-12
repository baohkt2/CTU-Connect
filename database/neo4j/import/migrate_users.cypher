// =================================================================
// MIGRATION SCRIPT: Fix NULL values in existing User nodes
// =================================================================
// This script updates existing User nodes to add missing username and isActive fields
// Run this script in Neo4j Browser or Neo4j Desktop to fix the current database

// 1. Update all users with NULL username and isActive fields
MATCH (u:User)
WHERE u.username IS NULL OR u.isActive IS NULL
SET u.username = COALESCE(u.username, split(u.email, '@')[0]),
    u.isActive = COALESCE(u.isActive, true),
    u.updatedAt = datetime()
RETURN count(u) as updatedUsers;

// 2. Verify the update
MATCH (u:User)
RETURN u.id, u.email, u.username, u.isActive, u.fullName
ORDER BY u.email;

// 3. Check for any remaining NULL values
MATCH (u:User)
WHERE u.username IS NULL OR u.isActive IS NULL
RETURN count(u) as remainingNullUsers;
