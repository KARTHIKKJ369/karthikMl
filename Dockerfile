# Ultra-lean CPU-optimized Dockerfile for Azure VM (Inference Only)
FROM python:3.11-slim

WORKDIR /app

# Install curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 1. Install lightweight CPU-only PyTorch (Only ~180MB instead of ~3GB Nvidia CUDA bloat)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install minimal production dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# 3. Copy standalone inference server & model checkpoints
COPY serve.py .
COPY checkpoints/ /app/checkpoints/

EXPOSE 8000

ENV PORTFOLIO_API_KEY="kj_live_sec_789f2a4b1c" \
    PYTHONUNBUFFERED=1

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
