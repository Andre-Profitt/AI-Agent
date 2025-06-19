# Dockerfile for Multi-Agent Platform
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Additional requirements for Phase 3
RUN pip install --no-cache-dir \
    aioredis==2.0.1 \
    msgpack==1.0.4 \
    psutil==5.9.5 \
    scikit-learn==1.3.0 \
    networkx==3.1 \
    pynvml==11.5.0 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    websockets==12.0

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

# Expose ports
EXPOSE 8080

# Run the application
CMD ["python", "-m", "uvicorn", "multiagent_api_deployment:app", "--host", "0.0.0.0", "--port", "8080"] 