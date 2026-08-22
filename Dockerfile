# Production Dockerfile for Azure VM (CPU or GPU)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml /app/
COPY src/ /app/src/
COPY checkpoints/ /app/checkpoints/

# Install python dependencies
RUN uv pip install --system fastapi uvicorn pydantic torch transformers accelerate safetensors

# Expose API port
EXPOSE 8000

ENV PORTFOLIO_API_KEY="kj_live_sec_789f2a4b1c"

# Run Uvicorn server
CMD ["uvicorn", "src.tinylm.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
