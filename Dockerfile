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
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

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

# Environment defaults (override at runtime)
ENV HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=2 \
    LOG_LEVEL=INFO \
    ARTIFACT_STORE=/app/artifacts \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

# Run with gunicorn (production) or uvicorn (dev override)
CMD ["gunicorn", "highlight_extractor.api.app:app", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info"]
