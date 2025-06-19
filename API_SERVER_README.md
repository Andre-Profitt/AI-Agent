# Multi-Agent Platform API Server

A comprehensive FastAPI-based REST API server for managing multi-agent systems with real-time WebSocket support, monitoring, and scalable architecture.

## 🚀 Features

- **RESTful API**: Complete CRUD operations for agents, tasks, resources, and conflicts
- **Real-time WebSocket**: Live monitoring and event streaming
- **Authentication & Security**: JWT-based authentication and API key support
- **Monitoring & Metrics**: Prometheus metrics and Grafana dashboards
- **Scalable Architecture**: Docker, Kubernetes, and horizontal scaling support
- **Database Integration**: PostgreSQL and Supabase support
- **Performance Testing**: Built-in load testing and benchmarking tools

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [API Endpoints](#api-endpoints)
3. [WebSocket API](#websocket-api)
4. [Authentication](#authentication)
5. [Monitoring](#monitoring)
6. [Deployment](#deployment)
7. [Development](#development)
8. [Troubleshooting](#troubleshooting)

## 🏃‍♂️ Quick Start

### Prerequisites

- Python 3.11+
- Redis 7.0+
- Docker (optional)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd AI-Agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Start Redis**
```bash
# Using Docker
docker run -d -p 6379:6379 redis:7-alpine

# Or using system package manager
sudo systemctl start redis
```

5. **Run the API server**
```bash
# Development mode
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload

# Production mode
uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

6. **Test the API**
```bash
# Health check
curl http://localhost:8000/api/v1/health

# API documentation
open http://localhost:8000/docs
```

## 🔌 API Endpoints

### Authentication

All API endpoints require authentication using Bearer tokens:

```bash
curl -H "Authorization: Bearer your-token-here" \
     http://localhost:8000/api/v1/agents
```

### Agent Management

#### Register Agent
```bash
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "data_processor_agent",
    "capabilities": ["data_processing", "ml_inference"],
    "resources": {"cpu": 2, "memory": "1Gi"},
    "metadata": {"version": "1.0", "team": "ai"}
  }'
```

#### List Agents
```bash
curl -H "Authorization: Bearer your-token" \
     http://localhost:8000/api/v1/agents

# With filters
curl -H "Authorization: Bearer your-token" \
     "http://localhost:8000/api/v1/agents?status_filter=active&capability_filter=data_processing"
```

#### Get Agent Details
```bash
curl -H "Authorization: Bearer your-token" \
     http://localhost:8000/api/v1/agents/{agent_id}
```

#### Deregister Agent
```bash
curl -X DELETE -H "Authorization: Bearer your-token" \
     http://localhost:8000/api/v1/agents/{agent_id}
```

### Task Management

#### Submit Task
```bash
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Process Customer Data",
    "description": "Analyze customer behavior patterns",
    "priority": 8,
    "required_capabilities": ["data_processing"],
    "resources": {"cpu": 1, "memory": "512Mi"},
    "deadline": "2024-01-15T23:59:59Z",
    "metadata": {"customer_id": "12345"}
  }'
```

#### List Tasks
```bash
curl -H "Authorization: Bearer your-token" \
     http://localhost:8000/api/v1/tasks

# With filters
curl -H "Authorization: Bearer your-token" \
     "http://localhost:8000/api/v1/tasks?status_filter=pending&priority_filter=8"
```

#### Get Task Details
```bash
curl -H "Authorization: Bearer your-token" \
     http://localhost:8000/api/v1/tasks/{task_id}
```

### Resource Management

#### Register Resource
```bash
curl -X POST http://localhost:8000/api/v1/resources \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gpu_cluster_01",
    "type": "gpu",
    "capacity": {"gpus": 4, "memory": "32Gi"},
    "metadata": {"model": "RTX 4090", "location": "rack-1"}
  }'
```

#### List Resources
```bash
curl -H "Authorization: Bearer your-token" \
     http://localhost:8000/api/v1/resources

# Filter by type
curl -H "Authorization: Bearer your-token" \
     "http://localhost:8000/api/v1/resources?type_filter=gpu"
```

### Conflict Management

#### Report Conflict
```bash
curl -X POST http://localhost:8000/api/v1/conflicts \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent-123",
    "task_id": "task-456",
    "conflict_type": "resource_contention",
    "description": "Multiple agents requesting same GPU resource",
    "severity": "high",
    "metadata": {"resource_id": "gpu-001"}
  }'
```

#### List Conflicts
```bash
curl -H "Authorization: Bearer your-token" \
     http://localhost:8000/api/v1/conflicts

# Filter by severity
curl -H "Authorization: Bearer your-token" \
     "http://localhost:8000/api/v1/conflicts?severity_filter=high"
```

### Workflow Management

#### Create Workflow
```bash
curl -X POST http://localhost:8000/api/v1/workflows \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Data Processing Pipeline",
    "description": "End-to-end data processing workflow",
    "steps": [
      {
        "name": "data_ingestion",
        "type": "task",
        "parameters": {"source": "database"}
      },
      {
        "name": "data_processing",
        "type": "task",
        "parameters": {"algorithm": "random_forest"}
      },
      {
        "name": "result_export",
        "type": "task",
        "parameters": {"format": "json"}
      }
    ],
    "metadata": {"version": "1.0"}
  }'
```

#### Execute Workflow
```bash
curl -X POST http://localhost:8000/api/v1/workflows/{workflow_id}/execute \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {"input_file": "data.csv"},
    "metadata": {"execution_id": "exec-123"}
  }'
```

### Performance Monitoring

#### Get Performance Metrics
```bash
curl -H "Authorization: Bearer your-token" \
     http://localhost:8000/api/v1/metrics/performance

# Filter by agent
curl -H "Authorization: Bearer your-token" \
     "http://localhost:8000/api/v1/metrics/performance?agent_id=agent-123"
```

### Dashboard

#### Get Dashboard Summary
```bash
curl -H "Authorization: Bearer your-token" \
     http://localhost:8000/api/v1/dashboard/summary
```

#### Get Dashboard Activity
```bash
curl -H "Authorization: Bearer your-token" \
     http://localhost:8000/api/v1/dashboard/activity?limit=50
```

## 🔌 WebSocket API

### Connection

Connect to the WebSocket endpoint:

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/client-123');
```

### Message Format

All WebSocket messages use JSON format:

```json
{
  "type": "message_type",
  "data": {},
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Subscription

Subscribe to specific event types:

```javascript
ws.send(JSON.stringify({
  "type": "subscribe",
  "subscriptions": ["agent_activity", "task_activity", "conflict_alerts"]
}));
```

### Event Types

- `agent_activity`: Agent registration, status changes, heartbeats
- `task_activity`: Task submission, assignment, completion
- `conflict_alerts`: Conflict reports and resolutions
- `system_metrics`: Real-time system performance metrics

### Example WebSocket Client

```python
import asyncio
import aiohttp
import json

async def websocket_client():
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect('ws://localhost:8000/api/v1/ws/test-client') as ws:
            # Subscribe to events
            await ws.send_json({
                "type": "subscribe",
                "subscriptions": ["agent_activity", "task_activity"]
            })
            
            # Listen for messages
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    print(f"Received: {data}")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"WebSocket error: {ws.exception()}")

asyncio.run(websocket_client())
```

## 🔐 Authentication

### JWT Authentication

The API supports JWT-based authentication:

```python
import jwt
from datetime import datetime, timedelta

# Create token
payload = {
    "sub": "user123",
    "exp": datetime.utcnow() + timedelta(hours=24)
}
token = jwt.encode(payload, "your-secret-key", algorithm="HS256")

# Use token
headers = {"Authorization": f"Bearer {token}"}
```

### API Key Authentication

For simpler integrations, use API key authentication:

```bash
curl -H "X-API-Key: your-api-key" \
     http://localhost:8000/api/v1/agents
```

## 📊 Monitoring

### Prometheus Metrics

The API server exposes metrics at `/metrics`:

```bash
curl http://localhost:8000/metrics
```

Key metrics:
- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request duration histogram
- `agent_platform_active_agents_total`: Number of active agents
- `agent_platform_active_tasks_total`: Number of active tasks
- `websocket_connections_total`: Number of WebSocket connections

### Grafana Dashboard

1. Access Grafana at http://localhost:3000
2. Login with admin/admin
3. Import dashboard from `monitoring/grafana/dashboards/agent-platform-dashboard.json`

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "components": {
    "platform": "healthy",
    "redis": "healthy",
    "database": "healthy"
  },
  "metrics": {
    "active_agents": 5,
    "active_tasks": 12,
    "active_connections": 3
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build image
docker build -t agent-platform:latest .

# Run with Docker Compose
docker-compose up -d
```

### Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace agent-platform

# Apply manifests
kubectl apply -f k8s/

# Check status
kubectl get pods -n agent-platform
```

### Production Configuration

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed production setup instructions.

## 🛠️ Development

### Project Structure

```
src/
├── api_server.py          # Main FastAPI application
├── unified_architecture/  # Core platform components
├── infrastructure/        # Database, monitoring, etc.
└── tools/                # Utility tools and helpers

monitoring/
├── prometheus.yml        # Prometheus configuration
└── grafana/             # Grafana dashboards and datasources

k8s/                     # Kubernetes manifests
scripts/                 # Utility scripts
```

### Running Tests

```bash
# Unit tests
pytest tests/

# Performance tests
python scripts/performance_test.py --agents 100 --tasks 200

# Load testing
python scripts/performance_test.py \
  --agents 500 \
  --tasks 1000 \
  --websocket-connections 50
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Endpoints

1. **Define Pydantic models** in `api_server.py`
2. **Add route handlers** with proper authentication
3. **Update WebSocket broadcasts** for real-time updates
4. **Add metrics** for monitoring
5. **Write tests** for the new functionality

Example:
```python
@router.post("/custom-endpoint", response_model=CustomResponse)
async def custom_endpoint(
    request: CustomRequest,
    platform: MultiAgentPlatform = Depends(get_platform),
    token: str = Depends(verify_token)
):
    # Implementation
    result = platform.custom_operation(request)
    
    # Broadcast update
    await manager.broadcast(
        WebSocketMessage(
            type="custom_event",
            data={"result": result}
        ).json()
    )
    
    return CustomResponse(**result)
```

## 🔧 Troubleshooting

### Common Issues

#### API Server Won't Start
```bash
# Check logs
docker-compose logs api-server

# Verify dependencies
pip list | grep fastapi

# Check environment variables
echo $DATABASE_URL
```

#### Redis Connection Issues
```bash
# Test Redis connection
redis-cli ping

# Check Redis logs
docker-compose logs redis
```

#### WebSocket Connection Issues
```bash
# Test WebSocket connection
wscat -c ws://localhost:8000/api/v1/ws/test

# Check WebSocket logs
docker-compose logs api-server | grep websocket
```

### Performance Issues

#### High Response Times
```bash
# Check resource usage
docker stats

# Monitor slow queries
docker-compose exec postgres psql -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"

# Check API metrics
curl http://localhost:8000/metrics | grep http_request_duration
```

#### Memory Issues
```bash
# Check memory usage
free -h
docker stats --no-stream

# Analyze memory leaks
python scripts/memory_profiler.py
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
export LOG_LEVEL=DEBUG
export DEBUG=true
uvicorn src.api_server:app --reload --log-level debug
```

## 📚 Additional Resources

- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [Architecture Guide](ARCHITECTURE.md) - System architecture overview
- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Production deployment instructions
- [Performance Testing](scripts/performance_test.py) - Load testing tools

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:

1. Check the [documentation](http://localhost:8000/docs)
2. Review the [troubleshooting guide](#troubleshooting)
3. Open an issue on GitHub
4. Contact the development team

---

**Happy coding! 🚀** 