# ── Stage 1: Builder ─────────────────────────────────────────────────────────
# Install dependencies in a separate stage to keep final image lean
FROM python:3.11-slim AS builder

WORKDIR /app

# System dependencies required for scientific Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install all Python dependencies into a target directory
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --target=/app/packages -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /app/packages /usr/local/lib/python3.11/site-packages/

# Copy application source code
COPY src/          ./src/
COPY config/       ./config/
COPY scripts/      ./scripts/

# Create required directories
RUN mkdir -p data/raw data/processed models/registry logs

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: start the API server
CMD ["python3", "-m", "uvicorn", "src.serving.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
