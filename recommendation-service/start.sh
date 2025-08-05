#!/bin/bash

# CTU Connect Recommendation Service Startup Script

set -e

echo "🚀 Starting CTU Connect Recommendation Service..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check if .env exists, if not create a template
if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment configuration..."
    cat > .env << EOF
# Database Configuration
DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5433/recommendation_db

# Redis Configuration
REDIS_URL=redis://localhost:6380

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Service Configuration
SECRET_KEY=your-secret-key-here
DEBUG=true
HOST=0.0.0.0
PORT=8000

# ML Configuration
MLFLOW_TRACKING_URI=http://localhost:5001
MODEL_PATH=./models/recommendation_model.pt

# Recommendation Settings
TOP_K_RECOMMENDATIONS=10
EMBEDDING_DIM=256
NUM_HEADS=8
EOF
    echo "✅ Created .env file. Please review and update the configuration."
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start supporting services with Docker Compose
echo "🐳 Starting supporting services (Database, Redis, MLflow)..."
docker-compose up -d recommendation-db recommendation-redis mlflow

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check if database is ready
echo "🔍 Checking database connection..."
python3 -c "
import asyncio
import asyncpg
import sys

async def check_db():
    try:
        conn = await asyncpg.connect('postgresql://postgres:password@localhost:5433/recommendation_db')
        await conn.close()
        print('✅ Database connection successful')
        return True
    except Exception as e:
        print(f'❌ Database connection failed: {e}')
        return False

if not asyncio.run(check_db()):
    sys.exit(1)
"

# Initialize database tables
echo "🗄️ Initializing database tables..."
python3 -c "
import asyncio
from db.models import create_tables

async def init_db():
    try:
        await create_tables()
        print('✅ Database tables created successfully')
    except Exception as e:
        print(f'❌ Database initialization failed: {e}')
        raise

asyncio.run(init_db())
"

# Create models directory if it doesn't exist
mkdir -p models

# Start the recommendation service
echo "🎯 Starting recommendation service..."
echo "📍 Service will be available at: http://localhost:${PORT:-8000}"
echo "📊 MLflow UI available at: http://localhost:5001"
echo "📚 API Documentation available at: http://localhost:${PORT:-8000}/docs"
echo ""
echo "🔧 To stop the service, press Ctrl+C"
echo "🔧 To stop all services, run: docker-compose down"
echo ""

# Start the main application
python3 main.py
