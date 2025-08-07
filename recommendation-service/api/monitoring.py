from prometheus_client import Counter, Histogram, Gauge, Info
from fastapi import FastAPI
import structlog

logger = structlog.get_logger()

# Define Prometheus metrics
class Metrics:
    def __init__(self):
        # Request metrics
        self.request_count = Counter(
            'recommendation_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status_code']
        )

        self.request_duration = Histogram(
            'recommendation_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint']
        )

        # Recommendation metrics
        self.recommendations_served = Counter(
            'recommendations_served_total',
            'Total recommendations served',
            ['variant']
        )

        self.feedback_received = Counter(
            'feedback_received_total',
            'Total feedback received',
            ['feedback_type']
        )

        # Model metrics
        self.model_predictions = Counter(
            'model_predictions_total',
            'Total model predictions made'
        )

        self.cache_hits = Counter(
            'cache_hits_total',
            'Cache hits',
            ['cache_type']
        )

        self.cache_misses = Counter(
            'cache_misses_total',
            'Cache misses',
            ['cache_type']
        )

        # Error metrics
        self.error_count = Counter(
            'errors_total',
            'Total errors',
            ['endpoint']
        )

        # System metrics
        self.active_users = Gauge(
            'active_users',
            'Number of active users'
        )

        self.model_accuracy = Gauge(
            'model_accuracy',
            'Current model accuracy'
        )

        # Service info
        self.service_info = Info(
            'recommendation_service_info',
            'Service information'
        )

# Global metrics instance
metrics = Metrics()

def setup_monitoring(app: FastAPI):
    """Setup monitoring and metrics collection"""
    try:
        # Set service info
        metrics.service_info.info({
            'version': '1.0.0',
            'environment': 'production',
            'model_version': 'v1.0'
        })

        logger.info("Monitoring setup completed")

    except Exception as e:
        logger.error(f"Failed to setup monitoring: {e}")
        raise
