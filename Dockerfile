# Multi-stage build for optimized image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 agent

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/agent/.local

# Copy application code
COPY --chown=agent:agent . .

# Set Python path
ENV PATH=/home/agent/.local/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/cache && \
    chown -R agent:agent /app

# Switch to non-root user
USER agent

# Expose ports
EXPOSE 7860 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Set environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT="7860"

# Run the application
CMD ["python", "app.py"] 