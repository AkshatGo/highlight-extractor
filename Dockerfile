# ============================================================
# Multi-stage Dockerfile for Highlight Extraction Service
# ============================================================
# Stage 1: Build dependencies
# Stage 2: Production runtime (smaller image)
# ============================================================

# --- Build stage ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first (cache layer)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Production stage ---
FROM python:3.11-slim AS production

WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create non-root user for security
RUN groupadd -r highlight && useradd -r -g highlight -d /app highlight \
    && mkdir -p /tmp/highlight_artifacts /app/artifacts \
    && chown -R highlight:highlight /app /tmp/highlight_artifacts /app/artifacts

USER highlight

# Copy .env.example as a reference (HF_TOKEN must be set at runtime)
COPY .env.example .env.example

# Environment defaults (override at runtime)
ENV HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=2 \
    LOG_LEVEL=INFO \
    ARTIFACT_STORE=/app/artifacts \
    WHISPER_MODEL=base \
    DIARIZATION_MODEL=pyannote/speaker-diarization-3.1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/healthz')" || exit 1

# Start script that respects $PORT (Render, Railway, etc.)
COPY <<'STARTUP' /app/startup.sh
#!/bin/bash
PORT=${PORT:-8000}
exec gunicorn highlight_extractor.api.app:app \
     --bind 0.0.0.0:$PORT \
     --workers 1 \
     --worker-class uvicorn.workers.UvicornWorker \
     --timeout 120 \
     --graceful-timeout 30 \
     --access-logfile - \
     --error-logfile - \
     --log-level info
STARTUP
RUN chmod +x /app/startup.sh

# Default port (overridden by platform)
ENV PORT=8000

CMD ["/app/startup.sh"]
