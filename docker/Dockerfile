# Multi-stage Dockerfile for ContractLens
# Stage 1: Base Python environment
FROM python:3.11-slim as base

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Development environment
FROM base as dev

# Copy requirements
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Set PYTHONPATH
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose API port
EXPOSE 8000

# Default command for development
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 3: Production inference environment (minimal)
FROM base as inference

# Create lightweight runtime dependencies
RUN echo "torch>=2.0.0" > /tmp/runtime-reqs.txt && \
    echo "transformers>=4.36.0" >> /tmp/runtime-reqs.txt && \
    echo "fastapi>=0.104.0" >> /tmp/runtime-reqs.txt && \
    echo "uvicorn>=0.24.0" >> /tmp/runtime-reqs.txt && \
    echo "pydantic>=2.0.0" >> /tmp/runtime-reqs.txt && \
    echo "chromadb>=0.4.0" >> /tmp/runtime-reqs.txt && \
    echo "openai>=1.3.0" >> /tmp/runtime-reqs.txt && \
    echo "langchain>=0.1.0" >> /tmp/runtime-reqs.txt

RUN pip install --no-cache-dir -r /tmp/runtime-reqs.txt

# Copy only necessary source code
COPY src/ /app/src/
COPY .env.example /app/.env.example

ENV PYTHONPATH=/app/src:$PYTHONPATH

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 4: Training environment (Kaggle-compatible)
FROM base as training

COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .
ENV PYTHONPATH=/app/src:$PYTHONPATH

WORKDIR /app/kaggle

CMD ["python", "train_lora.py"]
