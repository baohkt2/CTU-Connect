// Neo4j query to check relationships
MATCH (u:User {id: '31ba8a23-8a4e-4b24-99c2-0d768e617e71'})
OPTIONAL MATCH (u)-[:ENROLLED_IN]->(m:Major)
OPTIONAL MATCH (u)-[:IN_BATCH]->(b:Batch)
OPTIONAL MATCH (u)-[:HAS_GENDER]->(g:Gender)
RETURN u.fullName, u.studentId, m.code, m.name, b.year, g.name
