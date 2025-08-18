#!/usr/bin/env python3
"""
Startup script for the CTU Connect Recommendation Service
This script ensures proper module imports and starts the service
"""

import sys
import os

# Add the project root to Python path to ensure proper imports
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import and run the main application
from app.main import app
import uvicorn
from app.config import SERVICE_PORT

if __name__ == "__main__":
    print("Starting CTU Connect Recommendation Service...")
    print(f"Service will be available at http://0.0.0.0:{SERVICE_PORT}")
    print("Health check: http://0.0.0.0:{SERVICE_PORT}/health")
    print("API docs: http://0.0.0.0:{SERVICE_PORT}/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=SERVICE_PORT,
        reload=False,
        log_level="info"
    )
