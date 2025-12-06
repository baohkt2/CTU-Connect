// Cleanup script for duplicate relationships in Neo4j
// Run this script in Neo4j Browser or via cypher-shell

// 1. Find and remove duplicate ENROLLED_IN relationships
// Keep only the first relationship, delete the rest
MATCH (u:User)-[r:ENROLLED_IN]->(m:Major)
WITH u, COLLECT(r) as rels, COLLECT(m) as majors
WHERE SIZE(rels) > 1
UNWIND TAIL(rels) as duplicateRel
DELETE duplicateRel;

// 2. Find and remove duplicate IN_BATCH relationships
MATCH (u:User)-[r:IN_BATCH]->(b:Batch)
WITH u, COLLECT(r) as rels, COLLECT(b) as batches
WHERE SIZE(rels) > 1
UNWIND TAIL(rels) as duplicateRel
DELETE duplicateRel;

// 3. Find and remove duplicate HAS_GENDER relationships
MATCH (u:User)-[r:HAS_GENDER]->(g:Gender)
WITH u, COLLECT(r) as rels, COLLECT(g) as genders
WHERE SIZE(rels) > 1
UNWIND TAIL(rels) as duplicateRel
DELETE duplicateRel;

// 4. Verify the cleanup
MATCH (u:User)
OPTIONAL MATCH (u)-[r1:ENROLLED_IN]->(:Major)
WITH u, COUNT(r1) as majorCount
OPTIONAL MATCH (u)-[r2:IN_BATCH]->(:Batch)
WITH u, majorCount, COUNT(r2) as batchCount
OPTIONAL MATCH (u)-[r3:HAS_GENDER]->(:Gender)
WITH u, majorCount, batchCount, COUNT(r3) as genderCount
WHERE majorCount > 1 OR batchCount > 1 OR genderCount > 1
RETURN u.id, u.email, majorCount, batchCount, genderCount
ORDER BY u.email;

// If the above query returns no results, cleanup was successful
