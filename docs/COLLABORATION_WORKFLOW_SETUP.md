# Real-time Collaboration & Workflow Automation Setup Guide

## 📋 Prerequisites

- Python 3.11+
- Redis 7.0+ (for real-time features)
- PostgreSQL or Supabase (existing)
- WebSocket support in your infrastructure

## 🚀 Installation

### 1. Install Additional Dependencies

The new dependencies have been added to `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

### 2. Update Environment Variables

Add to your `.env` file:

```env
# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=your_redis_password
REDIS_SSL=false

# Collaboration Settings
COLLAB_SESSION_TTL=86400  # 24 hours
COLLAB_MAX_PARTICIPANTS=50
COLLAB_ENABLE_RECORDING=true

# Workflow Settings
WORKFLOW_MAX_PARALLEL_STEPS=10
WORKFLOW_DEFAULT_TIMEOUT=3600
WORKFLOW_RETRY_ATTEMPTS=3
WORKFLOW_TEMPLATE_DIR=./workflow_templates

# WebSocket Settings
WS_HEARTBEAT_INTERVAL=30
WS_MAX_CONNECTIONS_PER_USER=5
WS_MESSAGE_QUEUE_SIZE=1000
```

### 3. Database Schema Updates

Run these migrations for workflow and collaboration tracking:

```sql
-- Collaboration sessions table
CREATE TABLE collaboration_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Session participants
CREATE TABLE session_participants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(255) REFERENCES collaboration_sessions(session_id),
    user_id VARCHAR(255) NOT NULL,
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    left_at TIMESTAMP WITH TIME ZONE,
    role VARCHAR(50) DEFAULT 'participant'
);

-- Workflow templates
CREATE TABLE workflow_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    template_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    version VARCHAR(50) DEFAULT '1.0.0',
    steps JSONB NOT NULL,
    input_schema JSONB,
    output_schema JSONB,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Workflow executions
CREATE TABLE workflow_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id VARCHAR(255) UNIQUE NOT NULL,
    template_id VARCHAR(255) REFERENCES workflow_templates(template_id),
    status VARCHAR(50) NOT NULL,
    context JSONB DEFAULT '{}'::jsonb,
    step_results JSONB DEFAULT '{}'::jsonb,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error TEXT,
    created_by VARCHAR(255)
);

-- Agent handoffs
CREATE TABLE agent_handoffs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    handoff_id VARCHAR(255) UNIQUE NOT NULL,
    session_id VARCHAR(255),
    from_agent VARCHAR(255) NOT NULL,
    to_agent VARCHAR(255) NOT NULL,
    context JSONB DEFAULT '{}'::jsonb,
    reason TEXT,
    accepted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_sessions_status ON collaboration_sessions(status);
CREATE INDEX idx_participants_user ON session_participants(user_id);
CREATE INDEX idx_executions_status ON workflow_executions(status);
CREATE INDEX idx_executions_template ON workflow_executions(template_id);
```

## 🔧 Configuration

### 1. Create Workflow Templates Directory

The directory structure has been created:

```
workflow_templates/
├── research/
│   └── academic_research.yaml
├── support/
├── analytics/
└── content/
```

### 2. Example Workflow Template

An example academic research workflow has been created in `workflow_templates/research/academic_research.yaml`.

### 3. Update Main API Server

The collaboration and workflow systems are now integrated into the main application. The new features include:

- **Real-time Collaboration**: WebSocket-based communication with Redis pub/sub
- **Workflow Automation**: DAG-based workflow execution with multiple step types
- **Agent Handoffs**: Seamless agent transitions with context preservation
- **Live Progress Tracking**: Real-time updates on workflow and task progress

## 🎯 Usage Examples

### 1. Starting a Collaborative Workflow

```python
import aiohttp
import asyncio

async def start_collaborative_research():
    async with aiohttp.ClientSession() as session:
        # Start collaborative workflow
        response = await session.post(
            "http://localhost:8000/api/v1/collaborative-workflows/execute",
            json={
                "template_id": "research_report",
                "session_name": "Q1 Market Analysis",
                "participants": ["analyst1", "analyst2", "manager1"],
                "context": {
                    "topic": "AI Agent Market Trends",
                    "deadline": "2025-03-31",
                    "scope": "North America"
                }
            },
            headers={"Authorization": "Bearer your-token"}
        )
        
        result = await response.json()
        print(f"Started workflow: {result}")
        
        # Connect to WebSocket for real-time updates
        session_id = result["session"]["session_id"]
        execution_id = result["execution_id"]
        
        ws_url = f"ws://localhost:8000/ws/collaboration/analyst1"
        async with session.ws_connect(ws_url) as ws:
            # Join session
            await ws.send_json({
                "type": "join_session",
                "session_id": session_id
            })
            
            # Listen for updates
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    print(f"Received: {data}")
```

### 2. Creating Custom Workflow Templates

```python
# Create via API
async def create_custom_template():
    template = {
        "id": "custom_analysis",
        "name": "Custom Analysis Workflow",
        "description": "Tailored analysis workflow",
        "category": "custom",
        "steps": [
            {
                "id": "data_collection",
                "name": "Collect Data",
                "type": "agent_task",
                "agent_id": "data_agent",
                "parameters": {
                    "sources": ["api", "database", "files"]
                }
            },
            {
                "id": "parallel_processing",
                "name": "Process Data",
                "type": "parallel_agents",
                "parameters": {
                    "agents": ["processor1", "processor2"],
                    "aggregation": "merge"
                },
                "dependencies": ["data_collection"]
            }
        ],
        "input_schema": {
            "type": "object",
            "properties": {
                "data_source": {"type": "string"}
            }
        }
    }
    
    # Upload template
    response = await session.post(
        "http://localhost:8000/api/v1/workflows/templates/upload",
        data=yaml.dump(template),
        headers={"Content-Type": "text/yaml"}
    )
```

### 3. Monitoring Workflow Progress

```javascript
// JavaScript WebSocket client example
const ws = new WebSocket('ws://localhost:8000/ws/workflow/execution-123');

ws.onopen = () => {
    console.log('Connected to workflow monitor');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'workflow_status') {
        updateProgressBar(data.data.progress);
        updateCurrentStep(data.data.current_step);
        
        // Check for completion
        if (data.data.status === 'completed') {
            showCompletionMessage('Workflow completed successfully!');
        }
    }
};

// Function to update UI
function updateProgressBar(progress) {
    document.getElementById('progress-bar').style.width = `${progress}%`;
    document.getElementById('progress-text').textContent = `${Math.round(progress)}%`;
}
```

## 🔍 Monitoring & Debugging

### 1. Enable Debug Logging

```python
# In your settings
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "collaboration": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/collaboration.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "detailed"
        },
        "workflow": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/workflow.log",
            "maxBytes": 10485760,
            "backupCount": 5,
            "formatter": "detailed"
        }
    },
    "loggers": {
        "src.collaboration": {
            "handlers": ["collaboration"],
            "level": "DEBUG"
        },
        "src.workflow": {
            "handlers": ["workflow"],
            "level": "DEBUG"
        }
    }
}
```

### 2. Monitoring Dashboards

Create a Grafana dashboard with these queries:

```sql
-- Active collaboration sessions
SELECT COUNT(*) as active_sessions
FROM collaboration_sessions
WHERE status = 'active';

-- Workflow execution success rate
SELECT 
    DATE_TRUNC('hour', completed_at) as hour,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) * 100.0 / COUNT(*) as success_rate
FROM workflow_executions
WHERE completed_at > NOW() - INTERVAL '24 hours'
GROUP BY hour
ORDER BY hour;

-- Average workflow duration by template
SELECT 
    template_id,
    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration_seconds
FROM workflow_executions
WHERE status IN ('completed', 'failed')
GROUP BY template_id;
```

## 🚨 Troubleshooting

### Common Issues

1. **WebSocket Connection Failures**
   - Check Redis is running: `redis-cli ping`
   - Verify WebSocket upgrade headers in nginx/reverse proxy
   - Check firewall rules for WebSocket ports

2. **Workflow Timeout Issues**
   - Increase step timeouts in workflow templates
   - Check agent response times in metrics
   - Consider breaking large steps into smaller ones

3. **Memory Issues with Large Collaborations**
   - Implement session cleanup for inactive sessions
   - Use Redis expiration for temporary data
   - Limit concurrent workflow executions

### Performance Optimization

1. **Optimize Redis Usage**
```python
# Use Redis pipelines for batch operations
async def batch_update_context(updates: List[Dict]):
    pipe = redis_client.pipeline()
    for update in updates:
        pipe.setex(f"context:{update['key']}", 3600, json.dumps(update['value']))
    await pipe.execute()
```

2. **Implement Caching**
```python
# Cache workflow templates
@lru_cache(maxsize=100)
def get_cached_template(template_id: str):
    return workflow_engine.templates.get(template_id)
```

3. **Use Connection Pooling**
```python
# Redis connection pool
redis_pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    max_connections=50,
    decode_responses=True
)
```

## 🎉 Next Steps

1. **Implement Authentication**: Add JWT tokens for WebSocket connections
2. **Add Persistence**: Store collaboration history in database
3. **Enable Recordings**: Record sessions for playback
4. **Add Analytics**: Track collaboration patterns and workflow performance
5. **Scale Horizontally**: Implement Redis Cluster for high availability

## 📚 Additional Resources

- [WebSocket API Documentation](https://websockets.readthedocs.io/)
- [Redis Pub/Sub Guide](https://redis.io/docs/manual/pubsub/)
- [Workflow Patterns](https://www.workflowpatterns.com/)
- [Real-time Collaboration Patterns](https://www.ably.io/blog/realtime-collaboration-patterns/) 