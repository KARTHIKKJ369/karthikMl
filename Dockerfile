# Ultra-lean Production Dockerfile for Azure VM (Inference Only)
FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck validation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install only minimal production dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy only the standalone server and checkpoints
COPY serve.py .
COPY checkpoints/ /app/checkpoints/

EXPOSE 8000

ENV PORTFOLIO_API_KEY="kj_live_sec_789f2a4b1c" \
    PYTHONUNBUFFERED=1

# Run standalone inference server
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
