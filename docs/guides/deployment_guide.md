# Multi-Agent Platform API Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Multi-Agent Platform API server using Docker, Docker Compose, and Kubernetes. The platform includes real-time monitoring, load balancing, and production-ready configurations.

## Prerequisites

- Docker and Docker Compose installed
- Kubernetes cluster (for K8s deployment)
- Redis server (included in Docker Compose)
- SSL certificates (for production HTTPS)

## Quick Start with Docker Compose

### 1. Clone and Setup

```bash
git clone <repository-url>
cd AI-Agent
```

### 2. Environment Configuration

Create a `.env` file:

```bash
REDIS_URL=redis://redis:6379
LOG_LEVEL=INFO
API_TOKEN=your_secure_token_here
```

### 3. Build and Run

```bash
# Build the platform
docker-compose build

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

### 4. Verify Deployment

```bash
# Check API health
curl http://localhost:8080/health

# Access dashboard
open http://localhost:8080/dashboard

# Check Prometheus metrics
curl http://localhost:9090

# Access Grafana (admin/admin)
open http://localhost:3000
```

## Kubernetes Deployment

### 1. Build and Push Docker Image

```bash
# Build image
docker build -t multiagent-platform:latest .

# Tag for registry
docker tag multiagent-platform:latest your-registry/multiagent-platform:latest

# Push to registry
docker push your-registry/multiagent-platform:latest
```

### 2. Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace multi-agent-platform

# Apply configuration
kubectl apply -f multiagent-platform-k8s.yaml

# Check deployment status
kubectl get pods -n multi-agent-platform
kubectl get services -n multi-agent-platform
```

### 3. Configure Ingress

Update the ingress configuration with your domain:

```yaml
# In multiagent-platform-k8s.yaml
spec:
  rules:
  - host: your-domain.com  # Update this
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: platform-api-service
            port:
              number: 80
```

### 4. SSL Certificate Setup

For production, configure SSL certificates:

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Create ClusterIssuer for Let's Encrypt
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

## API Usage Examples

### 1. Register an Agent

```bash
curl -X POST "http://localhost:8080/api/v2/agents/register" \
  -H "Authorization: Bearer valid_token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "MyAgent",
    "version": "1.0.0",
    "capabilities": ["REASONING", "COLLABORATION"],
    "tags": ["custom", "production"],
    "resources": {"cpu_cores": 1.0, "memory_mb": 512},
    "description": "A custom reasoning agent"
  }'
```

### 2. Submit a Task

```bash
curl -X POST "http://localhost:8080/api/v2/tasks/submit" \
  -H "Authorization: Bearer valid_token" \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "analysis",
    "priority": 5,
    "payload": {"data": "sample_data"},
    "required_capabilities": ["REASONING"],
    "deadline": "2024-01-01T12:00:00Z"
  }'
```

### 3. List Agents

```bash
curl -X GET "http://localhost:8080/api/v2/agents" \
  -H "Authorization: Bearer valid_token"
```

### 4. WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/dashboard');

ws.onopen = () => {
    console.log('Connected to dashboard');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Received update:', data);
};
```

## Monitoring and Observability

### 1. Prometheus Metrics

The platform exposes metrics at `/metrics`:

```bash
curl http://localhost:8080/metrics
```

Key metrics include:
- `agent_registrations_total`
- `tasks_submitted_total`
- `task_execution_duration_seconds`
- `resource_utilization_percent`

### 2. Grafana Dashboards

Access Grafana at `http://localhost:3000` (admin/admin) and import dashboards for:
- Agent performance metrics
- Task execution statistics
- Resource utilization
- Error rates and latency

### 3. Alerting

Configure alerts in Prometheus for:
- High error rates
- Low agent availability
- Resource utilization thresholds
- Task latency issues

## Performance Testing

### 1. Run Load Tests

```bash
# Install dependencies
pip install aiohttp

# Run performance tests
python performance_test.py
```

### 2. Custom Load Testing

```python
from performance_test import PerformanceTester
import asyncio

async def custom_test():
    tester = PerformanceTester("http://localhost:8080", "valid_token")
    
    # Test with custom parameters
    results = await tester.run_load_test(
        agent_count=50,
        task_count=500,
        concurrent_tasks=25
    )
    
    print(f"Results: {results}")

asyncio.run(custom_test())
```

## Security Considerations

### 1. Authentication

- Replace the simple token authentication with proper JWT or OAuth2
- Implement rate limiting per user/IP
- Use HTTPS in production

### 2. Network Security

- Configure firewall rules
- Use private networks for internal communication
- Implement proper CORS policies

### 3. Data Protection

- Encrypt sensitive data at rest
- Use secure communication channels
- Implement proper logging and audit trails

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```bash
   # Check Redis status
   docker-compose logs redis
   
   # Restart Redis
   docker-compose restart redis
   ```

2. **API Not Responding**
   ```bash
   # Check API logs
   docker-compose logs platform-api
   
   # Check health endpoint
   curl http://localhost:8080/health
   ```

3. **WebSocket Connection Issues**
   ```bash
   # Check nginx configuration
   docker-compose logs nginx
   
   # Verify WebSocket proxy settings
   ```

4. **High Resource Usage**
   ```bash
   # Check resource utilization
   docker stats
   
   # Scale up if needed
   docker-compose up -d --scale platform-api=5
   ```

### Log Analysis

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f platform-api

# Search for errors
docker-compose logs | grep ERROR
```

## Scaling

### Horizontal Scaling

```bash
# Scale API instances
docker-compose up -d --scale platform-api=5

# Or in Kubernetes
kubectl scale deployment platform-api --replicas=5 -n multi-agent-platform
```

### Vertical Scaling

Update resource limits in `docker-compose.yml`:

```yaml
resources:
  limits:
    cpus: '4.0'
    memory: 4G
  reservations:
    cpus: '2.0'
    memory: 2G
```

## Backup and Recovery

### 1. Database Backup

```bash
# Backup Redis data
docker exec redis redis-cli BGSAVE

# Copy backup file
docker cp redis:/data/dump.rdb ./backup/
```

### 2. Configuration Backup

```bash
# Backup configuration files
tar -czf config-backup.tar.gz \
  docker-compose.yml \
  multiagent-platform-k8s.yaml \
  prometheus.yml \
  alerts.yml \
  nginx.conf
```

## Production Checklist

- [ ] SSL certificates configured
- [ ] Proper authentication implemented
- [ ] Monitoring and alerting set up
- [ ] Backup strategy in place
- [ ] Load balancing configured
- [ ] Rate limiting enabled
- [ ] Security headers configured
- [ ] Log aggregation set up
- [ ] Performance testing completed
- [ ] Disaster recovery plan ready

## Support

For issues and questions:
- Check the troubleshooting section
- Review logs for error messages
- Consult the API documentation
- Open an issue in the repository

## License

This deployment guide is part of the Multi-Agent Platform project. Please refer to the main project license for usage terms. 