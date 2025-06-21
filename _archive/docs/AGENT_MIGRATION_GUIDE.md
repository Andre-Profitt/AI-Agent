# Agent Migration Guide

## Overview
The agent implementations have been consolidated into a single `UnifiedAgent` class that combines all features from the various agent implementations.

## Migration Steps

### 1. Update Imports

**Old imports:**
```python
from src.agents.advanced_agent_fsm import FSMReActAgent
from src.agents.enhanced_fsm import EnhancedFSMAgent
from src.agents.migrated_enhanced_fsm_agent import MigratedEnhancedFSMAgent
```

**New import:**
```python
from src.agents.unified_agent import UnifiedAgent, create_agent
```

### 2. Update Agent Creation

**Old code:**
```python
# FSMReActAgent
agent = FSMReActAgent(
    tools=tools,
    model_name="gpt-4",
    max_iterations=10
)

# EnhancedFSMAgent
agent = EnhancedFSMAgent(
    tools=tools,
    enable_memory=True,
    enable_monitoring=True
)
```

**New code:**
```python
# Using factory function
agent = create_agent(
    agent_type="unified",
    name="My Agent",
    tools=tools,
    capabilities=[
        AgentCapability.REASONING,
        AgentCapability.TOOL_USE,
        AgentCapability.MEMORY
    ]
)

# Direct instantiation
from src.infrastructure.config import AgentConfig

config = AgentConfig(
    model_name="gpt-4",
    max_iterations=10,
    enable_memory=True,
    enable_monitoring=True
)

agent = UnifiedAgent(
    agent_id="agent-1",
    name="My Agent",
    config=config,
    tools=tools
)
```

### 3. Update Method Calls

**Old code:**
```python
# FSMReActAgent
result = await agent.arun({"messages": [HumanMessage(content=query)]})

# EnhancedFSMAgent
result = await agent.run(query, context=context)
```

**New code:**
```python
from src.core.entities.message import Message
from src.agents.unified_agent import AgentContext

# Create message
message = Message(content=query, role="user")

# Create context
context = AgentContext(
    session_id="session-123",
    metadata={"source": "api"}
)

# Process message
response = await agent.process(message, context)
```

### 4. Feature Mapping

| Old Feature | New Implementation |
|------------|-------------------|
| FSM States | Built-in state management with `AgentState` enum |
| Memory System | Enable with `AgentCapability.MEMORY` |
| Tool Execution | Built-in with circuit breaker protection |
| Multi-Agent | Use `collaborate()` method |
| Monitoring | Built-in metrics with `get_metrics()` |
| Error Handling | Automatic with state transitions and circuit breaker |

### 5. Configuration Migration

Create a unified configuration:

```python
from src.infrastructure.config import AgentConfig

config = AgentConfig(
    # Model settings
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=2000,
    
    # Execution settings
    max_iterations=10,
    timeout=30.0,
    
    # Memory settings
    enable_memory=True,
    memory_window_size=100,
    
    # Monitoring settings
    enable_monitoring=True,
    metrics_interval=60,
    
    # Error handling
    error_threshold=3,
    recovery_timeout=5.0,
    retry_attempts=3
)
```

## Benefits of Migration

1. **Simplified API**: One agent class with all features
2. **Better Performance**: Optimized with lazy loading and caching
3. **Improved Error Handling**: Built-in circuit breaker and state recovery
4. **Cleaner Architecture**: Clear separation of concerns
5. **Easier Testing**: Unified interface for all agent types
6. **Better Monitoring**: Comprehensive metrics and health checks

## Backward Compatibility

For temporary backward compatibility, you can use the adapter pattern:

```python
class FSMReActAgentAdapter:
    """Adapter for backward compatibility"""
    
    def __init__(self, *args, **kwargs):
        self.agent = create_agent("unified", **kwargs)
        
    async def arun(self, inputs):
        # Convert old format to new format
        message = Message(content=inputs["messages"][0].content, role="user")
        context = AgentContext(session_id="legacy")
        response = await self.agent.process(message, context)
        return {"final_answer": response.content}
```

## Deprecation Timeline

1. **Phase 1** (Current): Both old and new agents available
2. **Phase 2** (1 month): Old agents marked as deprecated with warnings
3. **Phase 3** (3 months): Old agents removed, only unified agent available

## Support

For migration assistance:
1. Check the examples in `examples/unified_agent_examples.py`
2. Run the migration script: `python scripts/migrate_to_unified_agent.py`
3. Review the test suite: `tests/test_unified_agent.py`
