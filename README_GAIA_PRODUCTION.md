# GAIA-Enhanced FSMReActAgent - Production Ready

A production-ready AI agent system that integrates GAIA (General AI Agent) components with advanced FSM (Finite State Machine) architecture for robust, scalable, and intelligent AI operations.

## 🚀 Features

### Core Components
- **Advanced Reasoning Engine**: Multi-layered reasoning with confidence scoring
- **Enhanced Memory System**: Episodic, semantic, and working memory with consolidation
- **Adaptive Tool System**: ML-based tool recommendation with failure recovery
- **Multi-Agent Orchestrator**: Specialized agents for complex task coordination
- **Production Vector Store**: Multiple providers (ChromaDB, Pinecone, in-memory)
- **Comprehensive Monitoring**: Prometheus metrics and OpenTelemetry tracing

### Production Features
- **High Performance**: Optimized caching, batching, and parallel processing
- **Fault Tolerance**: Circuit breakers, retry mechanisms, and graceful degradation
- **Scalability**: Connection pooling, resource management, and load balancing
- **Observability**: Health checks, performance monitoring, and detailed logging
- **Security**: Input validation, API key management, and secure configurations

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB disk space for models and data

### API Keys Required
- **Required**: GROQ_API_KEY (for LLM inference)
- **Optional**: 
  - OPENAI_API_KEY (for embeddings)
  - PINECONE_API_KEY (for vector store)
  - ANTHROPIC_API_KEY (for Claude)
  - SERPAPI_API_KEY (for web search)

## 🛠️ Installation

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd AI-Agent

# Run the installation script
chmod +x install_dependencies.sh
./install_dependencies.sh

# Setup environment
python setup_environment.py
```

### Manual Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm

# Setup environment
python setup_environment.py
```

## ⚙️ Configuration

### Environment Variables
Create a `.env` file with your configuration:

```bash
# Required API Keys
GROQ_API_KEY=your-groq-api-key-here
OPENAI_API_KEY=your-openai-api-key-here

# Vector Store Configuration
VECTOR_STORE_TYPE=chroma  # or pinecone, weaviate
CHROMA_PERSIST_DIR=./chroma_db

# Memory Settings
MEMORY_PERSIST_PATH=./data/agent_memories
MEMORY_MAX_SIZE=10000

# Performance Settings
CACHE_SIZE=1000
MAX_CONCURRENT_TASKS=5
BATCH_SIZE=32

# Monitoring
LOG_LEVEL=INFO
PROMETHEUS_PORT=9090
ENABLE_TRACING=true
```

### Advanced Configuration
```python
from src.agents.advanced_agent_fsm import FSMReActAgent
from src.gaia_components.production_vector_store import create_vector_store

# Create optimized agent
agent = FSMReActAgent(
    tools=your_tools,
    model_name="llama-3.3-70b-versatile",
    quality_level="THOROUGH",
    reasoning_type="LAYERED",
    vector_store=create_vector_store("chroma"),
    enable_monitoring=True
)
```

## 🚀 Usage

### Basic Usage
```python
from examples.gaia_usage_example import create_gaia_agent

# Create agent
agent = create_gaia_agent()

# Simple query
result = await agent.run("What is the capital of France?")
print(result['answer'])

# Complex reasoning
result = await agent.run(
    "Explain quantum computing and its applications in modern cryptography",
    reasoning_type="LAYERED",
    quality_level="THOROUGH"
)
print(result['reasoning_path'])
```

### Advanced Usage
```python
# Multi-agent coordination
result = await agent.run(
    "Analyze the weather data for New York and compare it with historical trends",
    use_multi_agent=True,
    agent_types=["researcher", "analyst", "synthesizer"]
)

# Memory integration
result = await agent.run(
    "What did we discuss about AI safety in our previous conversation?",
    use_memory=True,
    memory_types=["episodic", "semantic"]
)

# Adaptive tool selection
result = await agent.run(
    "Calculate the correlation between stock prices and market indices",
    use_adaptive_tools=True,
    tool_confidence_threshold=0.8
)
```

### Production Deployment
```python
from src.gaia_components.monitoring import HealthCheckHandler, initialize_monitoring

# Initialize monitoring
tracer = initialize_monitoring()

# Create health check handler
health_handler = HealthCheckHandler(agent)

# Get system health
health_status = await health_handler.get_health_status()
print(f"System status: {health_status['status']}")

# Get performance metrics
from src.gaia_components.performance_optimization import PerformanceOptimizer
optimizer = PerformanceOptimizer()
stats = optimizer.get_optimization_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
```

## 📊 Monitoring and Observability

### Health Checks
```python
# Get comprehensive health status
health_status = await health_handler.get_health_status()

# Check specific components
reasoning_health = health_status['components']['reasoning_engine']
memory_health = health_status['components']['memory_system']
tools_health = health_status['components']['adaptive_tools']
```

### Prometheus Metrics
The system exposes Prometheus metrics at `/metrics`:
- `gaia_queries_total`: Total queries processed
- `gaia_query_duration_seconds`: Query processing time
- `gaia_tool_executions_total`: Tool execution counts
- `gaia_memory_size`: Memory system size
- `gaia_active_agents`: Active agent count

### Performance Monitoring
```python
from src.gaia_components.performance_optimization import OptimizedGAIASystem

# Create optimized system
optimized_system = OptimizedGAIASystem()

# Get performance stats
stats = optimized_system.get_system_performance_stats()
print(f"Memory usage: {stats['optimizer']['memory_usage_mb']:.1f}MB")
print(f"Cache hit rate: {stats['optimizer']['cache_hit_rate']:.2%}")
```

## 🧪 Testing

### Run All Tests
```bash
# Run comprehensive test suite
pytest tests/test_gaia_production.py -v

# Run with coverage
pytest tests/test_gaia_production.py --cov=src --cov-report=html

# Run performance benchmarks
pytest tests/test_gaia_production.py::TestPerformanceBenchmarks -v
```

### Individual Test Categories
```bash
# Integration tests
pytest tests/test_gaia_production.py::TestProductionGAIAIntegration -v

# Performance tests
pytest tests/test_gaia_production.py::TestPerformanceBenchmarks -v

# Monitoring tests
pytest tests/test_gaia_production.py::TestMonitoringAndObservability -v
```

## 🔧 Development

### Project Structure
```
AI-Agent/
├── src/
│   ├── agents/
│   │   └── advanced_agent_fsm.py          # Main FSM agent
│   ├── gaia_components/
│   │   ├── advanced_reasoning_engine.py   # Reasoning engine
│   │   ├── enhanced_memory_system.py      # Memory system
│   │   ├── adaptive_tool_system.py        # Tool management
│   │   ├── multi_agent_orchestrator.py    # Multi-agent coordination
│   │   ├── production_vector_store.py     # Vector stores
│   │   ├── monitoring.py                  # Monitoring system
│   │   └── performance_optimization.py    # Performance utilities
│   └── utils/
├── examples/
│   └── gaia_usage_example.py              # Usage examples
├── tests/
│   └── test_gaia_production.py            # Test suite
├── requirements.txt                       # Dependencies
├── install_dependencies.sh               # Installation script
└── setup_environment.py                  # Environment setup
```

### Adding New Tools
```python
from src.gaia_components.adaptive_tool_system import Tool, ToolType, ToolCapability

# Create new tool
new_tool = Tool(
    id="custom_tool",
    name="Custom Analysis Tool",
    tool_type=ToolType.ANALYSIS,
    capabilities=[
        ToolCapability(
            name="data_analysis",
            description="Analyze complex datasets",
            input_schema={"data": "array", "analysis_type": "string"},
            output_schema={"result": "object", "confidence": "float"},
            examples=[{"data": [1,2,3], "analysis_type": "statistical"}]
        )
    ]
)

# Register with agent
agent.adaptive_tools.register_tool(new_tool)
```

### Custom Memory Types
```python
from src.gaia_components.enhanced_memory_system import MemoryType, MemoryPriority

# Store custom memory
memory_id = agent.memory_system.store_episodic(
    content="Custom event description",
    event_type="custom_event",
    metadata={"custom_field": "value"},
    priority=MemoryPriority.HIGH
)

# Retrieve with custom filters
memories = agent.memory_system.search_memories(
    "custom event",
    memory_types=[MemoryType.EPISODIC],
    filters={"custom_field": "value"}
)
```

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python setup_environment.py

EXPOSE 8000
CMD ["python", "app.py"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gaia-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gaia-agent
  template:
    metadata:
      labels:
        app: gaia-agent
    spec:
      containers:
      - name: gaia-agent
        image: gaia-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: GROQ_API_KEY
          valueFrom:
            secretKeyRef:
              name: gaia-secrets
              key: groq-api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1000m"
```

### Environment Variables for Production
```bash
# Production settings
LOG_LEVEL=WARNING
ENABLE_TRACING=true
JAEGER_ENDPOINT=http://jaeger:14268/api/traces
PROMETHEUS_PORT=9090

# Performance tuning
CACHE_SIZE=5000
MAX_CONCURRENT_TASKS=10
BATCH_SIZE=64
MEMORY_MAX_SIZE=50000

# Security
ENABLE_INPUT_VALIDATION=true
MAX_INPUT_LENGTH=10000
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

## 📈 Performance Optimization

### Caching Strategies
```python
from src.gaia_components.performance_optimization import PerformanceOptimizer

optimizer = PerformanceOptimizer()

# Cache expensive operations
@optimizer.memoize(ttl=3600, max_size=1000)
async def expensive_operation(data):
    # Expensive computation
    return result

# Batch processing
results = await optimizer.batch_process(
    items=large_dataset,
    processor=process_item,
    batch_size=32
)
```

### Memory Optimization
```python
# Optimize memory usage
optimizations = optimizer.optimize_memory_usage()
print(f"Memory saved: {optimizations['memory_saved_mb']:.1f}MB")

# Monitor memory trends
from src.gaia_components.monitoring import MemoryMonitor
monitor = MemoryMonitor()
snapshot = monitor.take_snapshot()
trend = monitor.get_memory_trend()
```

### Connection Pooling
```python
from src.gaia_components.performance_optimization import ConnectionPool

# Create connection pool
pool = ConnectionPool(max_connections=10)

async with pool as connection:
    # Use connection
    result = await connection.get("https://api.example.com/data")
```

## 🔍 Troubleshooting

### Common Issues

#### Memory Issues
```bash
# Check memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"

# Optimize memory
python -c "from src.gaia_components.performance_optimization import PerformanceOptimizer; PerformanceOptimizer().optimize_memory_usage()"
```

#### API Key Issues
```bash
# Verify API keys
python setup_environment.py

# Check environment
python -c "import os; print('GROQ_API_KEY:', 'SET' if os.getenv('GROQ_API_KEY') else 'NOT SET')"
```

#### Performance Issues
```python
# Get performance stats
from src.gaia_components.performance_optimization import PerformanceOptimizer
stats = PerformanceOptimizer().get_optimization_stats()
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Memory usage: {stats['memory_usage_mb']:.1f}MB")
```

### Debug Mode
```python
# Enable debug logging
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Enable detailed monitoring
agent = FSMReActAgent(
    tools=[],
    enable_debug=True,
    log_level="DEBUG"
)
```

## 📚 API Reference

### FSMReActAgent
```python
class FSMReActAgent:
    def __init__(self, tools, model_name, quality_level="BASIC", 
                 reasoning_type="SIMPLE", vector_store=None, 
                 enable_monitoring=False, enable_debug=False):
        """
        Initialize GAIA-enhanced FSM agent
        
        Args:
            tools: List of available tools
            model_name: LLM model name
            quality_level: "BASIC", "STANDARD", "THOROUGH"
            reasoning_type: "SIMPLE", "LAYERED", "ADAPTIVE"
            vector_store: Vector store instance
            enable_monitoring: Enable monitoring
            enable_debug: Enable debug mode
        """
    
    async def run(self, query, reasoning_type=None, quality_level=None,
                  use_multi_agent=False, use_memory=False, 
                  use_adaptive_tools=False, **kwargs):
        """
        Execute query with GAIA components
        
        Returns:
            Dict with 'success', 'answer', 'reasoning_path', 'correlation_id'
        """
```

### Advanced Reasoning Engine
```python
class AdvancedReasoningEngine:
    def generate_reasoning_path(self, query: str) -> ReasoningPath:
        """Generate reasoning path for query"""
    
    def evaluate_reasoning_quality(self, path: ReasoningPath) -> Dict[str, float]:
        """Evaluate reasoning quality"""
```

### Enhanced Memory System
```python
class EnhancedMemorySystem:
    def store_episodic(self, content: str, event_type: str, 
                      metadata: Dict = None) -> str:
        """Store episodic memory"""
    
    def store_semantic(self, content: str, concepts: List[str], 
                      metadata: Dict = None) -> str:
        """Store semantic memory"""
    
    def search_memories(self, query: str, memory_types: List[MemoryType] = None,
                       filters: Dict = None) -> List[Memory]:
        """Search memories"""
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run code formatting
black src/ tests/

# Run linting
flake8 src/ tests/

# Run type checking
mypy src/

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- GAIA benchmark for evaluation framework
- LangChain and LangGraph for the foundation
- Groq for high-performance LLM inference
- ChromaDB and Pinecone for vector storage
- Prometheus and OpenTelemetry for monitoring

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide

---

**Happy coding with GAIA! 🚀** 