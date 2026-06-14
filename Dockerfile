# WSL Prediction Engine — Dockerfile
# Builds a lean production image for Azure Container Apps.

FROM python:3.11-slim

# Metadata
LABEL maintainer="WSLAnalytics"
LABEL description="WSL xG-driven Dixon-Coles match prediction API"

# Don't buffer Python output (important for Azure log streaming)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
COPY constraints.txt .
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Copy application code
COPY data/      ./data/
COPY model/     ./model/
COPY api/       ./api/
COPY evaluation/ ./evaluation/

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# Azure Container Apps injects PORT env var; default to 8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2"]
