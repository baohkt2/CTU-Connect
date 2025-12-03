#!/bin/bash

# Đợi Neo4j sẵn sàng (Bolt port)
echo "⏳ Waiting for Neo4j to be ready..."
until cypher-shell -u neo4j -p password "RETURN 1;" > /dev/null 2>&1; do
  sleep 2
done

echo "🚀 Running init.cypher..."
cypher-shell -u neo4j -p password < /import/init.cypher
