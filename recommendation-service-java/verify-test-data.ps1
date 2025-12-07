# Verify Test Data

Write-Host "Verifying test data..." -ForegroundColor Cyan
Write-Host ""

Write-Host "PostgreSQL Data:" -ForegroundColor Yellow
docker exec -it postgres-recommend-dev psql -U postgres -d recommendation_db -c "SELECT COUNT(*) as total_posts FROM post_embeddings;"
docker exec -it postgres-recommend-dev psql -U postgres -d recommendation_db -c "SELECT post_id, LEFT(content, 50) as preview FROM post_embeddings LIMIT 3;"

Write-Host ""
Write-Host "Neo4j Data:" -ForegroundColor Yellow
Write-Host "Open: http://localhost:7474" -ForegroundColor Cyan
Write-Host "Run: MATCH (u:User) RETURN u.username;" -ForegroundColor White
Write-Host "Run: MATCH ()-[r:FRIEND]->() RETURN count(r);" -ForegroundColor White
Write-Host ""
