# ─────────────────────────────────────────────────────────────────────────────
# Fraud Detection API — Production Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
#
# Build:  docker build -t fraud-api .
# Run:    docker run -p 8000:8000 fraud-api
# Test:   curl http://localhost:8000/health
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── Security: never run as root in production ─────────────────────────────
RUN groupadd -r appuser && useradd -r -g appuser appuser

# ── Working directory inside the container ────────────────────────────────
WORKDIR /app

# ── Install curl for the HEALTHCHECK ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies first — Docker caches this layer ─────────────────────────
# Copying requirements.txt before the rest of the code means Docker only
# re-runs pip install when requirements.txt changes, not on every code change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────
COPY src/      ./src/
COPY api/      ./api/
COPY models/   ./models/

# ── Hand ownership to the non-root user, then switch to it ────────────────
RUN chown -R appuser:appuser /app
USER appuser

# ── Health check ──────────────────────────────────────────────────────────
# Docker calls /health every 30s. If it fails 3 times in a row the
# container is marked unhealthy. --start-period=10s gives the model
# time to load before checks begin.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Document which port the app listens on ────────────────────────────────
EXPOSE 8000

# ── Start the API server ──────────────────────────────────────────────────
# --workers 2     : handle 2 requests simultaneously
# --host 0.0.0.0  : accept connections from outside the container
# --log-level info: log each request
CMD ["uvicorn", "api.serve:app", \
     "--host",      "0.0.0.0",  \
     "--port",      "8000",     \
     "--workers",   "2",        \
     "--log-level", "info"]
