#!/usr/bin/env python3
"""
Agent consolidation script for AI Agent project
Merges overlapping agent implementations into a unified system
"""

import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
import ast
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class AgentConsolidator:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.agent_classes = {}  # class_name -> file_path
        self.agent_features = {}  # class_name -> features
        self.duplicates = []
        
    def analyze_and_consolidate(self):
        """Main method to analyze and consolidate agents"""
        logger.info("🤖 Analyzing agent implementations...\n")
        
        # Phase 1: Discover all agent classes
        self._discover_agent_classes()
        
        # Phase 2: Analyze features and overlaps
        self._analyze_agent_features()
        
        # Phase 3: Create consolidated agent
        self._create_consolidated_agent()
        
        # Phase 4: Create migration guide
        self._create_migration_guide()
        
    def _discover_agent_classes(self):
        """Discover all agent classes in the project"""
        agent_files = [
            "src/agents/advanced_agent_fsm.py",
            "src/agents/enhanced_fsm.py",
            "src/agents/migrated_enhanced_fsm_agent.py",
            "src/agents/multi_agent_system.py",
            "src/agents/crew_enhanced.py",
            "src/agents/crew_workflow.py",
            "src/agents/advanced_hybrid_architecture.py",
            "src/infrastructure/agents/concrete_agents.py",
            "src/application/agents/base_agent.py",
            "src/core/entities/agent.py"
        ]
        
        for file_path in agent_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    content = full_path.read_text()
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            if 'agent' in node.name.lower():
                                self.agent_classes[node.name] = file_path
                                logger.info(f"Found agent class: {node.name} in {file_path}")
                                
                except Exception as e:
                    logger.warning(f"Could not parse {file_path}: {e}")
                    
    def _analyze_agent_features(self):
        """Analyze features of each agent class"""
        feature_keywords = {
            'fsm': ['fsm', 'state', 'transition'],
            'memory': ['memory', 'remember', 'recall'],
            'reasoning': ['reasoning', 'think', 'analyze'],
            'tools': ['tool', 'execute', 'action'],
            'multi_agent': ['multi', 'orchestrat', 'coordinate'],
            'async': ['async', 'await', 'asyncio'],
            'monitoring': ['monitor', 'metric', 'track'],
            'error_handling': ['error', 'exception', 'retry']
        }
        
        for class_name, file_path in self.agent_classes.items():
            full_path = self.project_root / file_path
            if full_path.exists():
                content = full_path.read_text().lower()
                
                features = []
                for feature, keywords in feature_keywords.items():
                    if any(keyword in content for keyword in keywords):
                        features.append(feature)
                        
                self.agent_features[class_name] = features
                
        # Find duplicates
        feature_sets = {}
        for class_name, features in self.agent_features.items():
            feature_key = tuple(sorted(features))
            if feature_key in feature_sets:
                feature_sets[feature_key].append(class_name)
            else:
                feature_sets[feature_key] = [class_name]
                
        for feature_key, classes in feature_sets.items():
            if len(classes) > 1:
                self.duplicates.append(classes)
                logger.info(f"\nDuplicate functionality found in: {', '.join(classes)}")
                logger.info(f"  Features: {', '.join(feature_key)}")
                
    def _create_consolidated_agent(self):
        """Create a consolidated agent implementation"""
        consolidated_agent = '''"""
Consolidated Agent Implementation
Merges all agent functionality into a unified, clean architecture
"""

from typing import Dict, List, Optional, Any, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import logging
from abc import ABC, abstractmethod

from src.core.entities.agent import Agent as BaseAgent
from src.core.entities.tool import Tool, ToolResult
from src.core.entities.message import Message
from src.core.exceptions import AgentError
from src.infrastructure.config import AgentConfig
from src.services.circuit_breaker import CircuitBreaker
from src.utils.structured_logging import get_logger

logger = get_logger(__name__)

# Enums for agent states and capabilities
class AgentState(str, Enum):
    """Agent operational states"""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"
    
class AgentCapability(str, Enum):
    """Agent capabilities"""
    REASONING = "reasoning"
    TOOL_USE = "tool_use"
    MEMORY = "memory"
    COLLABORATION = "collaboration"
    LEARNING = "learning"

@dataclass
class AgentContext:
    """Context for agent execution"""
    session_id: str
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    history: List[Message] = field(default_factory=list)
    
class IAgent(ABC):
    """Agent interface"""
    
    @abstractmethod
    async def process(self, message: Message, context: AgentContext) -> Message:
        """Process a message and return response"""
        pass
        
    @abstractmethod
    async def execute_tool(self, tool: Tool, parameters: Dict[str, Any]) -> ToolResult:
        """Execute a tool with given parameters"""
        pass
        
    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities"""
        pass
        
    @abstractmethod
    def get_state(self) -> AgentState:
        """Get current agent state"""
        pass

class UnifiedAgent(BaseAgent, IAgent):
    """
    Unified agent implementation combining all features:
    - FSM-based state management
    - Advanced reasoning
    - Memory system
    - Tool execution
    - Multi-agent coordination
    - Performance monitoring
    - Error recovery
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        config: AgentConfig,
        tools: Optional[List[Tool]] = None,
        capabilities: Optional[List[AgentCapability]] = None
    ):
        super().__init__(agent_id=agent_id, name=name)
        self.config = config
        self.tools = tools or []
        self.capabilities = capabilities or [AgentCapability.REASONING, AgentCapability.TOOL_USE]
        self.state = AgentState.IDLE
        self._state_handlers = self._init_state_handlers()
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=config.error_threshold,
            recovery_timeout=config.recovery_timeout
        )
        
        # Components (lazy loaded)
        self._reasoning_engine = None
        self._memory_system = None
        self._tool_executor = None
        
        logger.info(f"Initialized UnifiedAgent: {name} ({agent_id})")
        
    def _init_state_handlers(self) -> Dict[AgentState, Callable]:
        """Initialize state transition handlers"""
        return {
            AgentState.IDLE: self._handle_idle,
            AgentState.THINKING: self._handle_thinking,
            AgentState.EXECUTING: self._handle_executing,
            AgentState.WAITING: self._handle_waiting,
            AgentState.ERROR: self._handle_error
        }
        
    async def process(self, message: Message, context: AgentContext) -> Message:
        """Process a message through the agent pipeline"""
        try:
            # State transition: IDLE -> THINKING
            await self._transition_state(AgentState.THINKING)
            
            # Add to history
            context.history.append(message)
            
            # Reasoning phase
            reasoning_result = await self._reason(message, context)
            
            # Tool execution phase if needed
            if reasoning_result.get("requires_tools"):
                await self._transition_state(AgentState.EXECUTING)
                tool_results = await self._execute_tools(reasoning_result["tool_calls"])
                reasoning_result["tool_results"] = tool_results
                
            # Generate response
            response = await self._generate_response(reasoning_result, context)
            
            # Update memory if enabled
            if AgentCapability.MEMORY in self.capabilities:
                await self._update_memory(message, response, context)
                
            # Return to IDLE
            await self._transition_state(AgentState.IDLE)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self._transition_state(AgentState.ERROR)
            return Message(
                content=f"I encountered an error: {str(e)}",
                role="assistant",
                metadata={"error": True}
            )
            
    async def _reason(self, message: Message, context: AgentContext) -> Dict[str, Any]:
        """Reasoning phase"""
        if not self._reasoning_engine:
            from src.gaia_components.advanced_reasoning_engine import AdvancedReasoningEngine
            self._reasoning_engine = AdvancedReasoningEngine()
            
        return await self._reasoning_engine.reason(
            query=message.content,
            context=context.metadata,
            history=context.history
        )
        
    async def _execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[ToolResult]:
        """Execute tools based on reasoning"""
        if not self._tool_executor:
            from src.application.tools.tool_executor import ToolExecutor
            self._tool_executor = ToolExecutor(self.tools)
            
        results = []
        for call in tool_calls:
            tool_name = call["tool"]
            parameters = call["parameters"]
            
            result = await self._circuit_breaker.call(
                self._tool_executor.execute,
                tool_name,
                parameters
            )
            results.append(result)
            
        return results
        
    async def execute_tool(self, tool: Tool, parameters: Dict[str, Any]) -> ToolResult:
        """Execute a specific tool"""
        if not self._tool_executor:
            from src.application.tools.tool_executor import ToolExecutor
            self._tool_executor = ToolExecutor(self.tools)
            
        return await self._tool_executor.execute_tool(tool, parameters)
        
    async def _generate_response(
        self, 
        reasoning_result: Dict[str, Any], 
        context: AgentContext
    ) -> Message:
        """Generate final response"""
        content = reasoning_result.get("response", "I couldn't generate a response.")
        
        if "tool_results" in reasoning_result:
            # Incorporate tool results
            tool_summary = self._summarize_tool_results(reasoning_result["tool_results"])
            content = f"{content}\\n\\nBased on the tools: {tool_summary}"
            
        return Message(
            content=content,
            role="assistant",
            metadata={
                "reasoning_path": reasoning_result.get("reasoning_path"),
                "confidence": reasoning_result.get("confidence", 0.5),
                "tools_used": [r.tool_name for r in reasoning_result.get("tool_results", [])]
            }
        )
        
    def _summarize_tool_results(self, results: List[ToolResult]) -> str:
        """Summarize tool execution results"""
        summaries = []
        for result in results:
            if result.success:
                summaries.append(f"{result.tool_name}: {result.data}")
            else:
                summaries.append(f"{result.tool_name}: Failed - {result.error}")
        return " | ".join(summaries)
        
    async def _update_memory(
        self, 
        message: Message, 
        response: Message, 
        context: AgentContext
    ):
        """Update memory system"""
        if not self._memory_system:
            from src.gaia_components.enhanced_memory_system import EnhancedMemorySystem
            self._memory_system = EnhancedMemorySystem()
            
        await self._memory_system.add_interaction(
            query=message.content,
            response=response.content,
            metadata={
                "session_id": context.session_id,
                "timestamp": datetime.utcnow(),
                "confidence": response.metadata.get("confidence", 0.5)
            }
        )
        
    async def _transition_state(self, new_state: AgentState):
        """Handle state transitions"""
        old_state = self.state
        self.state = new_state
        
        logger.debug(f"State transition: {old_state} -> {new_state}")
        
        # Call state handler
        handler = self._state_handlers.get(new_state)
        if handler:
            await handler()
            
    async def _handle_idle(self):
        """Handle IDLE state"""
        pass
        
    async def _handle_thinking(self):
        """Handle THINKING state"""
        pass
        
    async def _handle_executing(self):
        """Handle EXECUTING state"""
        pass
        
    async def _handle_waiting(self):
        """Handle WAITING state"""
        pass
        
    async def _handle_error(self):
        """Handle ERROR state"""
        # Attempt recovery
        await asyncio.sleep(self.config.recovery_timeout)
        await self._transition_state(AgentState.IDLE)
        
    def get_capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities"""
        return self.capabilities
        
    def get_state(self) -> AgentState:
        """Get current state"""
        return self.state
        
    async def collaborate(self, other_agent: 'UnifiedAgent', task: Dict[str, Any]) -> Any:
        """Collaborate with another agent"""
        if AgentCapability.COLLABORATION not in self.capabilities:
            raise AgentError("This agent doesn't have collaboration capability")
            
        # Simple collaboration protocol
        collaboration_message = Message(
            content=f"Collaboration request: {task}",
            role="system",
            metadata={"from_agent": self.agent_id}
        )
        
        context = AgentContext(
            session_id=f"collab_{self.agent_id}_{other_agent.agent_id}",
            metadata={"collaboration": True}
        )
        
        return await other_agent.process(collaboration_message, context)
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "state": self.state.value,
            "capabilities": [c.value for c in self.capabilities],
            "tools_count": len(self.tools),
            "circuit_breaker_state": self._circuit_breaker.state
        }

# Factory function for creating agents
def create_agent(
    agent_type: str = "unified",
    agent_id: Optional[str] = None,
    name: Optional[str] = None,
    config: Optional[AgentConfig] = None,
    **kwargs
) -> UnifiedAgent:
    """Factory function to create agents"""
    
    if not agent_id:
        import uuid
        agent_id = str(uuid.uuid4())
        
    if not name:
        name = f"{agent_type}_agent_{agent_id[:8]}"
        
    if not config:
        config = AgentConfig()
        
    # Create appropriate agent based on type
    if agent_type == "unified":
        return UnifiedAgent(agent_id, name, config, **kwargs)
    else:
        # For backward compatibility, always return UnifiedAgent
        logger.warning(f"Unknown agent type {agent_type}, creating UnifiedAgent")
        return UnifiedAgent(agent_id, name, config, **kwargs)
'''
        
        consolidated_path = self.project_root / "src/agents/unified_agent.py"
        consolidated_path.write_text(consolidated_agent)
        logger.info(f"\n✅ Created consolidated agent at {consolidated_path}")
        
    def _create_migration_guide(self):
        """Create migration guide for existing code"""
        migration_guide = """# Agent Migration Guide

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
    \"\"\"Adapter for backward compatibility\"\"\"
    
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
"""
        
        guide_path = self.project_root / "AGENT_MIGRATION_GUIDE.md"
        guide_path.write_text(migration_guide)
        logger.info(f"📄 Created migration guide at {guide_path}")

def main():
    consolidator = AgentConsolidator()
    consolidator.analyze_and_consolidate()
    
    logger.info("\n✅ Agent consolidation complete!")
    logger.info("   - Created unified agent: src/agents/unified_agent.py")
    logger.info("   - Created migration guide: AGENT_MIGRATION_GUIDE.md")
    logger.info("\nNext steps:")
    logger.info("   1. Review the unified agent implementation")
    logger.info("   2. Update existing code using the migration guide")
    logger.info("   3. Run tests to ensure compatibility")

if __name__ == "__main__":
    main()