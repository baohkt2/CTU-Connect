# Python Service Setup Guide - Recommend Service

## Prerequisites

- Python 3.10+ installed
- PowerShell (Windows)
- Docker Desktop running (for databases)

## Quick Setup

### 1. Setup Virtual Environment

Run the setup script from `recommend-service/python-model` directory:

```powershell
cd d:\LVTN\CTU-Connect-demo\recommend-service\python-model
.\setup-venv.ps1
```

This script will:
- ✓ Check Python installation
- ✓ Create virtual environment in `venv/` folder
- ✓ Activate the virtual environment
- ✓ Upgrade pip
- ✓ Install all dependencies from `requirements.txt`
- ✓ Create `.env` file from `.env.example`

**Note:** Installing dependencies may take 5-10 minutes as PyTorch is large (~2GB).

### 2. Start Required Databases

From the root project directory, start only the databases:

```powershell
cd d:\LVTN\CTU-Connect-demo
docker-compose up recommend-postgres recommend-redis neo4j kafka
```

Or start all databases:

```powershell
docker-compose up -d
```

### 3. Run Python Service

From `python-model` directory:

```powershell
cd d:\LVTN\CTU-Connect-demo\recommend-service\python-model
.\run-dev.ps1
```

Or manually:

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Run server
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Verify Service is Running

Open browser:
- **API Documentation:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health
- **API Root:** http://localhost:8000

## Configuration

### Environment Variables (`.env` file)

```env
# Service
PORT=8000
DEBUG=true
LOG_LEVEL=INFO

# Database (Docker)
DATABASE_URL=postgresql://recommend_user:recommend_pass@localhost:5435/recommend_db

# Redis (Docker - recommend-redis)
REDIS_HOST=localhost
REDIS_PORT=6380
REDIS_PASSWORD=recommend_redis_pass

# Kafka (Docker)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Model Configuration
MODEL_PATH=./model/academic_posts_model
PHOBERT_MODEL_NAME=vinai/phobert-base
EMBEDDING_DIMENSION=768
MAX_SEQUENCE_LENGTH=256
```

## Running in IDE (PyCharm/VS Code)

### PyCharm

1. Open `recommend-service/python-model` as project root
2. Configure Python Interpreter:
   - **Settings** → **Project** → **Python Interpreter**
   - Click **Add Interpreter** → **Existing environment**
   - Select: `d:\LVTN\CTU-Connect-demo\recommend-service\python-model\venv\Scripts\python.exe`
3. Run Configuration:
   - **Script:** `server.py`
   - **Working directory:** `d:\LVTN\CTU-Connect-demo\recommend-service\python-model`
   - **Environment variables:** Load from `.env`

### VS Code

1. Open `recommend-service/python-model` folder
2. Install Python extension
3. Select Interpreter:
   - Press `Ctrl+Shift+P`
   - Type: "Python: Select Interpreter"
   - Choose: `.\venv\Scripts\python.exe`
4. Create `launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: FastAPI",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "server:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ],
            "jinja": true,
            "justMyCode": true,
            "envFile": "${workspaceFolder}/.env"
        }
    ]
}
```

## Troubleshooting

### Issue: Python not found
```powershell
# Install Python 3.10+
# Download from: https://www.python.org/downloads/
```

### Issue: Permission denied on script execution
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Port 8000 already in use
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process (replace PID)
taskkill /PID <PID> /F
```

### Issue: Cannot connect to Redis
```powershell
# Check if Redis is running
docker ps | findstr redis

# Check Redis connection
docker exec -it ctu-recommend-redis redis-cli -a recommend_redis_pass ping
```

### Issue: PyTorch installation fails
```powershell
# Install CPU version (smaller, faster)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

## Development Workflow

1. **Start databases:**
   ```powershell
   docker-compose up recommend-postgres recommend-redis
   ```

2. **Activate venv:**
   ```powershell
   cd recommend-service\python-model
   .\venv\Scripts\Activate.ps1
   ```

3. **Run with auto-reload:**
   ```powershell
   uvicorn server:app --reload --port 8000
   ```

4. **Run tests:**
   ```powershell
   pytest tests/ -v
   ```

5. **Deactivate venv:**
   ```powershell
   deactivate
   ```

## API Endpoints

- `GET /health` - Health check
- `GET /` - Root endpoint
- `POST /api/model/predict` - Generate recommendations
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation

## Model Files

The PhoBERT model will be downloaded automatically on first run:
- Model: `vinai/phobert-base`
- Cache location: `~/.cache/huggingface/transformers/`
- Size: ~400MB

## Dependencies

Major packages:
- **FastAPI** (0.104.1) - Web framework
- **Uvicorn** (0.24.0) - ASGI server
- **PyTorch** (2.1.0) - Deep learning
- **Transformers** (4.35.0) - Hugging Face models
- **Redis** (5.0.1) - Caching
- **Kafka-Python** (2.0.2) - Event streaming

Total installation size: ~3-4 GB (including PyTorch)

## Next Steps

After Python service is running, start the Java service:
1. Open `recommend-service/java-api` in IntelliJ IDEA
2. Set active profile to `dev`
3. Run `RecommendationServiceApplication.java`
4. Java service will connect to Python service at `http://localhost:8000`

## Support

For issues or questions, check:
- API Docs: http://localhost:8000/docs
- Logs: Console output or `logs/` directory
- Docker logs: `docker-compose logs recommend-python`
