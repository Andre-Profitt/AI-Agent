# Multi-Agent Platform API Server

A comprehensive FastAPI-based platform for managing and orchestrating AI agents with real-time monitoring, load balancing, and production-ready deployment configurations.

## 🚀 Features

- **FastAPI Server**: High-performance API with automatic documentation
- **WebSocket Support**: Real-time dashboard updates and monitoring
- **Agent Management**: Register, discover, and manage AI agents
- **Task Orchestration**: Submit and track tasks across multiple agents
- **Marketplace**: Agent discovery and rating system
- **Resource Management**: CPU, memory, and GPU allocation
- **Conflict Resolution**: Automated conflict detection and resolution
- **Real-time Monitoring**: Prometheus metrics and Grafana dashboards
- **Load Balancing**: Nginx configuration with rate limiting
- **Container Orchestration**: Docker Compose and Kubernetes support
- **SSL/TLS Support**: Production-ready HTTPS configuration

## 📋 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Redis (included in Docker Compose)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd AI-Agent
```

### 2. Environment Configuration

```bash
# Copy example environment file
cp env.example .env

# Edit configuration
nano .env
```

### 3. Start the Platform

```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps
```

### 4. Access Services

- **API Documentation**: http://localhost:8080/docs
- **Dashboard**: http://localhost:8080/dashboard
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx Load    │    │   FastAPI API   │    │   Redis Cache   │
│    Balancer     │◄──►│     Server      │◄──►│   & State       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Prometheus    │    │   Grafana       │    │   Agent Pool    │
│   Monitoring    │    │   Dashboards    │    │   Registry      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔧 API Endpoints

### Agent Management

- `POST /api/v2/agents/register` - Register a new agent
- `GET /api/v2/agents` - List all agents
- `GET /api/v2/agents/{agent_id}` - Get agent details
- `DELETE /api/v2/agents/{agent_id}` - Unregister agent

### Task Management

- `POST /api/v2/tasks/submit` - Submit a task
- `GET /api/v2/tasks/{task_id}` - Get task status
- `POST /api/v2/tasks/batch` - Submit multiple tasks

### Marketplace

- `POST /api/v2/marketplace/publish` - Publish agent to marketplace
- `POST /api/v2/marketplace/search` - Search for agents
- `POST /api/v2/marketplace/rate/{agent_id}` - Rate an agent

### Monitoring

- `GET /api/v2/dashboard/overview` - System overview
- `GET /api/v2/resources/utilization` - Resource utilization
- `POST /api/v2/conflicts/report` - Report conflicts

### WebSocket

- `WS /ws/dashboard` - Real-time dashboard updates

## 📊 Monitoring

### Prometheus Metrics

The platform exposes comprehensive metrics:

- `agent_registrations_total` - Total agent registrations
- `tasks_submitted_total` - Total tasks submitted
- `task_execution_duration_seconds` - Task execution time
- `resource_utilization_percent` - Resource usage
- `agent_task_failures_total` - Task failures

### Grafana Dashboards

Pre-configured dashboards include:

- Agent performance metrics
- Task execution statistics
- Resource utilization
- Error rates and latency
- System health overview

### Alerting

Configured alerts for:

- High error rates (>10% for 5 minutes)
- Low agent availability (<50% for 10 minutes)
- High resource utilization (>90% CPU, >85% memory)
- Redis connection failures
- High task latency (>30s 95th percentile)
- Low success rates (<80% for 10 minutes)

## 🚀 Deployment Options

### Docker Compose (Development/Testing)

```bash
# Start all services
docker-compose up -d

# Scale API instances
docker-compose up -d --scale platform-api=3

# View logs
docker-compose logs -f platform-api
```

### Kubernetes (Production)

```bash
# Deploy to Kubernetes
kubectl apply -f multiagent-platform-k8s.yaml

# Check deployment status
kubectl get pods -n multi-agent-platform

# Scale deployment
kubectl scale deployment platform-api --replicas=5 -n multi-agent-platform
```

### Manual Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export REDIS_URL=redis://localhost:6379
export API_TOKEN=your_token

# Run the server
python multiagent_api_deployment.py
```

## 🔒 Security

### Authentication

- Bearer token authentication (configurable)
- Rate limiting per IP/user
- CORS configuration
- Security headers

### Network Security

- HTTPS/TLS encryption
- Firewall configuration
- Private network isolation
- SSL certificate management

### Data Protection

- Encrypted data at rest
- Secure communication channels
- Audit logging
- Access control

## 📈 Performance

### Load Testing

Run performance tests:

```bash
# Install test dependencies
pip install aiohttp

# Run load tests
python performance_test.py
```

### Scaling

- **Horizontal**: Add more API instances
- **Vertical**: Increase resource limits
- **Auto-scaling**: Kubernetes HPA configuration
- **Load balancing**: Nginx with health checks

### Optimization

- Connection pooling
- Caching strategies
- Async processing
- Resource monitoring

## 🛠️ Configuration

### Environment Variables

Key configuration options:

```bash
# Redis
REDIS_URL=redis://redis:6379

# API
API_TOKEN=your_secure_token
LOG_LEVEL=INFO

# Security
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Platform
MAX_AGENTS=1000
TASK_TIMEOUT_SECONDS=300
```

### Nginx Configuration

- Rate limiting
- SSL/TLS termination
- Load balancing
- WebSocket proxy
- Security headers

### Prometheus Configuration

- Metrics collection
- Alert rules
- Service discovery
- Data retention

## 🔍 Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   ```bash
   docker-compose logs redis
   docker-compose restart redis
   ```

2. **API Not Responding**
   ```bash
   docker-compose logs platform-api
   curl http://localhost:8080/health
   ```

3. **WebSocket Issues**
   ```bash
   docker-compose logs nginx
   # Check nginx.conf WebSocket configuration
   ```

4. **High Resource Usage**
   ```bash
   docker stats
   docker-compose up -d --scale platform-api=5
   ```

### Log Analysis

```bash
# View all logs
docker-compose logs -f

# Search for errors
docker-compose logs | grep ERROR

# Monitor specific service
docker-compose logs -f platform-api
```

## 📚 Documentation

- [Deployment Guide](deployment_guide.md) - Detailed deployment instructions
- [API Documentation](http://localhost:8080/docs) - Interactive API docs
- [Architecture Overview](ARCHITECTURE.md) - System architecture details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- Check the troubleshooting section
- Review logs for error messages
- Consult the API documentation
- Open an issue in the repository

## 🔄 Updates

Stay updated with the latest features and improvements:

```bash
git pull origin main
docker-compose down
docker-compose up -d --build
```

---

**Multi-Agent Platform API Server** - Empowering AI agent collaboration at scale. 