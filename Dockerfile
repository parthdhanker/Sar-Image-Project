# ── Stage 1: Builder ──────────────────────────────────────────────
# Install dependencies in a separate layer so they are cached
# and not re-downloaded on every code change.
FROM python:3.10-slim AS builder

WORKDIR /build

# Install system deps needed to compile some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /code

# Runtime system libraries only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy the app source code
COPY app/ ./app/

# Create writable directories for uploads/outputs at runtime
# (Azure App Service may mount these to persistent storage)
RUN mkdir -p \
    app/static/uploads \
    app/static/outputs \
    app/static/segments

# Run as non-root user for security
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /code
USER appuser

# Azure App Service uses port 8000 by default;
# the PORT env var is also respected if set.
EXPOSE 8000

# Uvicorn with multiple workers for production performance
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--forwarded-allow-ips", "*"]
