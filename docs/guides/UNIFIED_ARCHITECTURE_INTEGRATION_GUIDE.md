# Unified Architecture Integration Guide

## Overview

This guide explains how to integrate the Phase 3 Unified Architecture for Hybrid Agent System and Multi-Agent Collaboration Platform with your existing AI agent codebase.

## Architecture Components

The unified architecture consists of the following core components:

### 1. Core Interfaces (`src/unified_architecture/core.py`)
- **AgentCapability**: Enum defining agent capabilities (DATA_ANALYSIS, MACHINE_LEARNING, etc.)
- **AgentStatus**: Enum for agent states (IDLE, BUSY, ERROR, etc.)
- **AgentMetadata**: Data structure for agent information
- **UnifiedTask**: Standardized task representation
- **TaskResult**: Standardized task result format
- **IUnifiedAgent**: Abstract interface for all agents

### 2. Orchestration Engine (`src/unified_architecture/orchestration.py`)
- Manages agent registration and lifecycle
- Handles task submission and execution
- Coordinates complex multi-agent workflows
- Manages task dependencies and scheduling

### 3. State Management (`src/unified_architecture/state_management.py`)
- Distributed state storage (memory, Redis, database)
- State change notifications
- Checkpoint and recovery mechanisms
- State cleanup and optimization

### 4. Communication Protocol (`src/unified_architecture/communication.py`)
- Inter-agent messaging system
- Message types (COLLABORATION, COORDINATION, SYSTEM, etc.)
- Message queues and routing
- Topic-based subscriptions

### 5. Resource Management (`src/unified_architecture/resource_management.py`)
- CPU, memory, GPU, network, and storage monitoring
- Resource allocation and scheduling
- Resource optimization and load balancing
- Resource usage tracking

### 6. Agent Registry (`src/unified_architecture/registry.py`)
- Dynamic agent registration and discovery
- Capability-based agent lookup
- Health monitoring and status tracking
- Agent metadata management

### 7. Task Distribution (`src/unified_architecture/task_distribution.py`)
- Intelligent task routing and distribution
- Multiple distribution strategies (ROUND_ROBIN, LOAD_BALANCED, etc.)
- Performance-based agent selection
- Task priority and deadline management

### 8. Shared Memory System (`src/unified_architecture/shared_memory.py`)
- Distributed knowledge base
- Semantic search and retrieval
- Experience sharing between agents
- Memory lifecycle management

### 9. Conflict Resolution (`src/unified_architecture/conflict_resolution.py`)
- Automated conflict detection and resolution
- Multiple resolution strategies
- Negotiation and consensus building
- Conflict history and learning

### 10. Performance Tracking (`src/unified_architecture/performance.py`)
- Agent and task performance metrics
- Execution time and success rate tracking
- Collaboration efficiency metrics
- Performance alerts and optimization

### 11. Collaboration Dashboard (`src/unified_architecture/dashboard.py`)
- Real-time system monitoring
- Agent performance visualization
- Collaboration network analysis
- System health and alerts

### 12. Agent Marketplace (`src/unified_architecture/marketplace.py`)
- Agent discovery and listing
- Rating and review system
- Agent deployment and management
- Marketplace statistics and analytics

### 13. Multi-Agent Platform (`src/unified_architecture/platform.py`)
- Main platform that integrates all components
- Platform lifecycle management
- Comprehensive health monitoring
- Event handling and system coordination

## Integration Steps

### Step 1: Update Dependencies

The required dependencies are already included in `requirements.txt`:

```txt
# Distributed state management
aioredis==2.0.1
redis==5.0.1
msgpack==1.0.7

# System monitoring
psutil==5.9.6

# GPU monitoring (optional)
pynvml==11.5.0

# Advanced data structures
heapq2==0.1.0
```

### Step 2: Create Unified Agent Adapters

Create adapters to make your existing agents compatible with the unified architecture:

```python
# src/adapters/unified_agent_adapter.py
from typing import List, Dict, Any
from src.unified_architecture.core import (
    IUnifiedAgent, UnifiedTask, TaskResult, AgentStatus, AgentCapability
)

class UnifiedAgentAdapter(IUnifiedAgent):
    """Adapter to make existing agents compatible with unified architecture"""
    
    def __init__(self, existing_agent, capabilities: List[AgentCapability]):
        self.agent = existing_agent
        self.capabilities = capabilities
        self.status = AgentStatus.IDLE
    
    async def execute_task(self, task: UnifiedTask) -> TaskResult:
        """Execute task using the existing agent"""
        try:
            self.status = AgentStatus.BUSY
            
            # Convert unified task to agent-specific format
            agent_task = self._convert_task(task)
            
            # Execute using existing agent
            result = await self.agent.process(agent_task)
            
            # Convert result back to unified format
            unified_result = self._convert_result(result, task.id)
            
            self.status = AgentStatus.IDLE
            return unified_result
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e)
            )
    
    async def get_status(self) -> AgentStatus:
        return self.status
    
    async def get_capabilities(self) -> List[AgentCapability]:
        return self.capabilities
    
    def _convert_task(self, task: UnifiedTask) -> Dict[str, Any]:
        """Convert unified task to agent-specific format"""
        return {
            "description": task.description,
            "requirements": task.requirements,
            "metadata": task.metadata
        }
    
    def _convert_result(self, result: Any, task_id: str) -> TaskResult:
        """Convert agent result to unified format"""
        return TaskResult(
            task_id=task_id,
            success=True,
            data=result,
            metadata={"agent": self.agent.__class__.__name__}
        )
```

### Step 3: Initialize the Platform

```python
# src/platform_integration.py
import asyncio
from src.unified_architecture import MultiAgentPlatform, PlatformConfig
from src.adapters.unified_agent_adapter import UnifiedAgentAdapter
from src.unified_architecture.core import AgentCapability, AgentMetadata

class PlatformIntegration:
    """Integration layer for the unified architecture platform"""
    
    def __init__(self):
        self.config = PlatformConfig(
            max_concurrent_tasks=100,
            task_timeout=300,
            heartbeat_interval=30,
            cleanup_interval=3600,
            enable_marketplace=True,
            enable_dashboard=True,
            enable_performance_tracking=True,
            enable_conflict_resolution=True,
            storage_backend="memory"
        )
        self.platform = MultiAgentPlatform(self.config)
        self.agents = {}
    
    async def start(self):
        """Start the platform"""
        await self.platform.start()
    
    async def stop(self):
        """Stop the platform"""
        await self.platform.stop()
    
    async def register_existing_agent(self, agent, name: str, capabilities: List[AgentCapability]):
        """Register an existing agent with the platform"""
        # Create adapter
        adapter = UnifiedAgentAdapter(agent, capabilities)
        
        # Create metadata
        metadata = AgentMetadata(
            agent_id=str(id(agent)),
            name=name,
            capabilities=capabilities,
            status=adapter.status,
            version="1.0.0",
            description=f"Adapted {name}",
            tags=["adapted", "existing"],
            created_at=datetime.utcnow()
        )
        
        # Register with platform
        success = await self.platform.register_agent(adapter, metadata)
        if success:
            self.agents[name] = adapter
        
        return success
    
    async def submit_unified_task(self, description: str, capabilities: List[AgentCapability], **kwargs):
        """Submit a task to the platform"""
        from src.unified_architecture.core import UnifiedTask, TaskType, TaskPriority
        
        task = UnifiedTask(
            id=uuid4(),
            description=description,
            task_type=TaskType.GENERAL,
            priority=TaskPriority.MEDIUM,
            requirements={"capabilities": capabilities},
            dependencies=[],
            metadata=kwargs
        )
        
        return await self.platform.submit_task(task)
```

### Step 4: Integrate with Existing FSM Agent

```python
# src/fsm_unified_integration.py
from src.advanced_agent_fsm import FSMReActAgent
from src.unified_architecture.core import AgentCapability, IUnifiedAgent, UnifiedTask, TaskResult

class FSMUnifiedAgent(IUnifiedAgent):
    """Unified interface for FSM-based agents"""
    
    def __init__(self, fsm_agent: FSMReActAgent):
        self.fsm_agent = fsm_agent
        self.capabilities = [
            AgentCapability.GENERAL_PURPOSE,
            AgentCapability.REASONING,
            AgentCapability.TASK_EXECUTION
        ]
        self.status = AgentStatus.IDLE
    
    async def execute_task(self, task: UnifiedTask) -> TaskResult:
        """Execute task using FSM agent"""
        try:
            self.status = AgentStatus.BUSY
            
            # Execute using FSM agent
            result = await self.fsm_agent.run(task.description)
            
            self.status = AgentStatus.IDLE
            
            return TaskResult(
                task_id=task.id,
                success=True,
                data={"response": result},
                metadata={"agent_type": "fsm", "execution_time": 0}
            )
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e)
            )
    
    async def get_status(self) -> AgentStatus:
        return self.status
    
    async def get_capabilities(self) -> List[AgentCapability]:
        return self.capabilities
```

### Step 5: Update Main Application

```python
# app.py (updated)
import asyncio
from src.platform_integration import PlatformIntegration
from src.fsm_unified_integration import FSMUnifiedAgent
from src.advanced_agent_fsm import FSMReActAgent
from src.unified_architecture.core import AgentCapability

class EnhancedApp:
    """Enhanced application with unified architecture"""
    
    def __init__(self):
        self.platform_integration = PlatformIntegration()
        self.fsm_agent = None
        self.unified_fsm_agent = None
    
    async def initialize(self):
        """Initialize the enhanced application"""
        # Start platform
        await self.platform_integration.start()
        
        # Create FSM agent
        self.fsm_agent = FSMReActAgent(
            # ... existing configuration
        )
        
        # Create unified FSM agent
        self.unified_fsm_agent = FSMUnifiedAgent(self.fsm_agent)
        
        # Register with platform
        await self.platform_integration.register_existing_agent(
            self.unified_fsm_agent,
            "FSM-Agent",
            [AgentCapability.GENERAL_PURPOSE, AgentCapability.REASONING]
        )
    
    async def process_query(self, query: str):
        """Process a query using the unified platform"""
        # Submit task to platform
        task_id = await self.platform_integration.submit_unified_task(
            description=query,
            capabilities=[AgentCapability.GENERAL_PURPOSE]
        )
        
        # Monitor task execution
        while True:
            status = await self.platform_integration.platform.get_task_status(task_id)
            if status and status.get('status') == 'completed':
                return status.get('result', {}).get('data', {}).get('response', '')
            
            await asyncio.sleep(0.1)
    
    async def shutdown(self):
        """Shutdown the application"""
        await self.platform_integration.stop()

# Update existing app.py to use enhanced version
async def main():
    app = EnhancedApp()
    await app.initialize()
    
    # Your existing application logic here
    # ...
    
    await app.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

## Usage Examples

### Example 1: Basic Platform Usage

```python
import asyncio
from src.unified_architecture import MultiAgentPlatform, PlatformConfig
from src.unified_architecture.core import UnifiedTask, TaskType, TaskPriority

async def basic_example():
    # Configure and start platform
    config = PlatformConfig(enable_marketplace=True, enable_dashboard=True)
    platform = MultiAgentPlatform(config)
    
    async with platform.platform_context():
        # Submit a task
        task = UnifiedTask(
            id=uuid4(),
            description="Analyze customer data",
            task_type=TaskType.ANALYSIS,
            priority=TaskPriority.HIGH,
            requirements={"capabilities": ["data_analysis"]}
        )
        
        task_id = await platform.submit_task(task)
        print(f"Task submitted: {task_id}")
        
        # Monitor task
        while True:
            status = await platform.get_task_status(task_id)
            if status and status.get('status') == 'completed':
                print(f"Task completed: {status.get('result')}")
                break
            await asyncio.sleep(1)

asyncio.run(basic_example())
```

### Example 2: Multi-Agent Collaboration

```python
async def collaboration_example():
    platform = MultiAgentPlatform()
    
    async with platform.platform_context():
        # Register multiple agents
        agents = [
            ("DataAnalysisAgent", [AgentCapability.DATA_ANALYSIS]),
            ("ProcessingAgent", [AgentCapability.DATA_PROCESSING]),
            ("CollaborationAgent", [AgentCapability.COLLABORATION])
        ]
        
        for name, capabilities in agents:
            # Register agent (implementation depends on your agent classes)
            pass
        
        # Submit collaborative task
        task = UnifiedTask(
            description="Multi-agent data analysis and processing pipeline",
            task_type=TaskType.COLLABORATION,
            priority=TaskPriority.HIGH,
            requirements={"capabilities": ["data_analysis", "data_processing", "collaboration"]}
        )
        
        task_id = await platform.submit_task(task)
        
        # Monitor collaboration
        while True:
            status = await platform.get_task_status(task_id)
            if status and status.get('status') == 'completed':
                print("Collaboration completed successfully")
                break
            await asyncio.sleep(1)
```

### Example 3: Performance Monitoring

```python
async def performance_example():
    platform = MultiAgentPlatform()
    
    async with platform.platform_context():
        # Get platform statistics
        stats = await platform.get_platform_stats()
        print(f"Platform uptime: {stats.platform_uptime}")
        print(f"Total tasks: {stats.total_tasks}")
        print(f"Success rate: {stats.completed_tasks / max(stats.total_tasks, 1):.2%}")
        
        # Get agent performance
        agents = await platform.get_available_agents()
        for agent in agents:
            performance = await platform.get_agent_performance(agent.agent_id)
            if performance:
                print(f"{agent.name}: {performance.get('success_rate', 0):.2%} success rate")
        
        # Get collaboration network
        network = await platform.get_collaboration_network()
        print(f"Collaboration network: {len(network.get('nodes', []))} nodes")
```

## Advanced Features

### 1. Custom Event Handlers

```python
async def custom_event_handler(data):
    print(f"Event received: {data}")

platform = MultiAgentPlatform()
platform.add_event_handler("task_completed", custom_event_handler)
```

### 2. Resource Management

```python
# Allocate resources
allocation = await platform.allocate_resources({
    "cpu": 2.0,
    "memory": 4.0,
    "gpu": 1.0
})

# Use resources
# ...

# Release resources
await platform.release_resources(allocation.id)
```

### 3. Memory Sharing

```python
# Share experience
memory_entry = MemoryEntry(
    type=MemoryType.EXPERIENCE,
    content="Successfully processed large dataset",
    source_agent=agent_id,
    tags=["processing", "large_data"]
)
await platform.share_memory(memory_entry)

# Search memory
results = await platform.search_memory("large dataset processing")
```

### 4. Conflict Resolution

```python
# Create conflict
conflict = Conflict(
    conflict_type=ConflictType.RESOURCE_COMPETITION,
    agents_involved=[agent1_id, agent2_id],
    description="Both agents need GPU resources"
)

# Resolve conflict
resolution = await platform.resolve_conflict(conflict)
```

## Best Practices

### 1. Error Handling

```python
try:
    result = await platform.submit_task(task)
except Exception as e:
    logger.error(f"Task submission failed: {e}")
    # Handle error appropriately
```

### 2. Resource Cleanup

```python
async with platform.platform_context():
    # Your code here
    pass
# Platform automatically cleaned up
```

### 3. Performance Optimization

```python
# Use appropriate task priorities
task = UnifiedTask(
    priority=TaskPriority.HIGH,  # For urgent tasks
    # ...
)

# Monitor resource usage
utilization = await platform.resource_manager.get_utilization()
if utilization["cpu"] > 0.8:
    # Implement backpressure
    pass
```

### 4. Monitoring and Logging

```python
# Regular health checks
health = await platform.health_check()
if health["platform_status"] != "healthy":
    # Alert or restart
    pass

# Performance monitoring
stats = await platform.get_platform_stats()
if stats.completed_tasks / max(stats.total_tasks, 1) < 0.9:
    # Investigate performance issues
    pass
```

## Troubleshooting

### Common Issues

1. **Agent Registration Fails**
   - Check agent implements IUnifiedAgent interface
   - Verify agent capabilities are valid
   - Ensure agent metadata is complete

2. **Task Execution Fails**
   - Check task requirements match available resources
   - Verify agent capabilities match task requirements
   - Monitor agent health status

3. **Platform Startup Issues**
   - Check all dependencies are installed
   - Verify configuration parameters
   - Check system resources

4. **Performance Issues**
   - Monitor resource utilization
   - Check task distribution strategy
   - Review agent performance metrics

### Debugging

```python
# Enable debug logging
logging.getLogger("src.unified_architecture").setLevel(logging.DEBUG)

# Get detailed health information
health = await platform.health_check()
for component, status in health["components"].items():
    print(f"{component}: {status}")

# Monitor specific agent
performance = await platform.get_agent_performance(agent_id)
print(f"Agent performance: {performance}")
```

## Conclusion

The unified architecture provides a comprehensive framework for building sophisticated multi-agent systems. By following this integration guide, you can enhance your existing AI agent codebase with advanced features like:

- Multi-agent collaboration and coordination
- Intelligent task distribution and resource management
- Performance monitoring and optimization
- Conflict resolution and consensus building
- Shared memory and experience learning
- Real-time monitoring and analytics

The architecture is designed to be modular and extensible, allowing you to implement only the components you need while maintaining the flexibility to add more features as your system grows. 