"""Gunicorn configuration for production deployment.

Environment variables:
    WORKERS       — Number of worker processes (default: CPU count * 2 + 1)
    PORT          — Port to bind (default: 8000)
    HOST          — Host to bind (default: 0.0.0.0)
    LOG_LEVEL     — Log level (default: info)
    TIMEOUT       — Worker timeout in seconds (default: 120)
    GRACEFUL_TIMEOUT — Graceful shutdown timeout (default: 30)
    MAX_REQUESTS  — Max requests before worker restart (default: 1000)
    MAX_REQUESTS_JITTER — Random jitter for max_requests (default: 50)
"""

import os

# Server socket
bind = f"{os.environ.get('HOST', '0.0.0.0')}:{os.environ.get('PORT', '8000')}"

# Worker processes
workers = int(os.environ.get("WORKERS", "2"))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = int(os.environ.get("TIMEOUT", "120"))
graceful_timeout = int(os.environ.get("GRACEFUL_TIMEOUT", "30"))
keepalive = 5

# Restart workers periodically to prevent memory leaks
max_requests = int(os.environ.get("MAX_REQUESTS", "1000"))
max_requests_jitter = int(os.environ.get("MAX_REQUESTS_JITTER", "50"))

# Logging
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = os.environ.get("LOG_LEVEL", "info").lower()

# Process naming
proc_name = "highlight-extractor"

# Security
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190

# Preloading (loads app before forking workers — saves memory)
preload_app = False  # Set to True if models are shared across workers
