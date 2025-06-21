"""
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

from src.core.entities.base_agent import Agent as BaseAgent
from src.core.entities.base_tool import Tool, ToolResult
from src.core.entities.base_message import Message
from src.core.exceptions import AgentError
from src.infrastructure.agent_config import AgentConfig
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
            content = f"{content}\n\nBased on the tools: {tool_summary}"
            
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
