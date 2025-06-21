from agent import query
from agent import tools
from examples.enhanced_unified_example import adapter
from examples.enhanced_unified_example import bridge
from examples.enhanced_unified_example import execution_time
from examples.enhanced_unified_example import metrics
from examples.enhanced_unified_example import start_time
from examples.enhanced_unified_example import task
from tests.load_test import success

from src.adapters.fsm_unified_adapter import capabilities
from src.adapters.fsm_unified_adapter import unified_result
from src.adapters.fsm_unified_adapter import user_input
from src.agents.advanced_hybrid_architecture import fsm_result
from src.api_server import fsm_agent
from src.api_server import redis_url
from src.database.models import agent_id
from src.database.models import metadata
from src.database.models import priority
from src.database.models import tool
from src.gaia_components.multi_agent_orchestrator import task_id
from src.tools import advanced_file_reader
from src.tools import file_reader
from src.tools import python_interpreter
from src.tools_introspection import name
from src.unified_architecture.enhanced_platform import task_type
from src.utils.base_tool import PythonREPLTool
from src.utils.base_tool import web_researcher
from src.utils.semantic_search_tool import semantic_search_tool
from src.utils.tavily_search import tavily_search_backoff
from src.utils.weather import get_weather

from src.agents.advanced_agent_fsm import AgentCapability

from src.agents.advanced_agent_fsm import AgentStatus

from src.tools.base_tool import Tool

from src.agents.advanced_agent_fsm import Agent

from src.agents.advanced_agent_fsm import MultiAgentPlatform

from src.agents.advanced_agent_fsm import IUnifiedAgent

from src.agents.advanced_agent_fsm import AgentMetadata
from src.unified_architecture.enhanced_platform import AgentCapability
from src.unified_architecture.enhanced_platform import AgentMetadata
from src.unified_architecture.enhanced_platform import AgentStatus
from src.unified_architecture.enhanced_platform import IUnifiedAgent
from src.unified_architecture.enhanced_platform import TaskResult
from src.unified_architecture.enhanced_platform import UnifiedTask
# TODO: Fix undefined variables: Any, Dict, List, Optional, adapter, agent_id, bridge, cap, capabilities, datetime, e, execution_time, fsm_agent, fsm_result, logging, metadata, metrics, name, other_agent, priority, query, redis_url, result, start_time, success, tags, task, task_id, task_type, tools, unified_result, user_input, uuid
from langchain_experimental.tools import PythonREPLTool

from src.core.entities.agent import AgentCapability
from src.core.entities.agent import AgentMetadata
from src.tools.base_tool import tool
from src.utils.advanced_file_reader import advanced_file_reader
from src.utils.base_tool import file_reader
from src.utils.base_tool import get_weather
from src.utils.base_tool import semantic_search_tool
from src.utils.base_tool import tavily_search_backoff
from src.utils.tools_enhanced import web_researcher
from src.utilsthon_interpreter import python_interpreter

# TODO: Fix undefined variables: PythonREPLTool, adapter, advanced_file_reader, agent_id, bridge, cap, capabilities, e, execution_time, file_reader, fsm_agent, fsm_result, get_weather, metadata, metrics, name, other_agent, priority, python_interpreter, query, redis_url, result, self, semantic_search_tool, start_time, success, tags, task, task_id, task_type, tavily_search_backoff, tool, tools, unified_result, user_input, web_researcher

"""
FSM to Unified Architecture Adapter

This adapter bridges the existing FSMReActAgent with the enhanced unified architecture,
allowing seamless integration between the two systems.
"""

from typing import Optional
from typing import Any
from typing import List

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from src.agents.advanced_agent_fsm import FSMReActAgent
from src.unified_architecture.enhanced_platform import (
    IUnifiedAgent, AgentCapability, AgentStatus,
    AgentMetadata, UnifiedTask, TaskResult
)

logger = logging.getLogger(__name__)

class FSMUnifiedAdapter(IUnifiedAgent):
    """Adapter that makes FSMReActAgent compatible with the unified architecture"""

    def __init__(self, fsm_agent: FSMReActAgent, agent_id: str, name: str):
        self.fsm_agent = fsm_agent
        self.agent_id = agent_id
        self.name = name
        self.status = AgentStatus.IDLE
        self.capabilities = self._map_fsm_capabilities()
        self._initialized = False

    def _map_fsm_capabilities(self) -> List[AgentCapability]:
        """Map FSM agent capabilities to unified architecture capabilities"""
        capabilities = [
            AgentCapability.REASONING,
            AgentCapability.TOOL_USE,
            AgentCapability.STATE_BASED,
            AgentCapability.EXECUTION
        ]

        # Add additional capabilities based on available tools
        if hasattr(self.fsm_agent, 'tools') and self.fsm_agent.tools:
            if any('search' in str(tool).lower() for tool in self.fsm_agent.tools):
                capabilities.append(AgentCapability.MEMORY_ACCESS)

            if any('python' in str(tool).lower() for tool in self.fsm_agent.tools):
                capabilities.append(AgentCapability.LEARNING)

        return capabilities

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the FSM agent adapter"""
        try:
            # The FSM agent is already initialized when created
            self._initialized = True
            self.status = AgentStatus.AVAILABLE

            logger.info("FSM agent adapter {} initialized successfully", extra={"self_name": self.name})
            return True

        except Exception as e:
            logger.error("Failed to initialize FSM agent adapter: {}", extra={"e": e})
            self.status = AgentStatus.ERROR
            return False

    async def execute(self, task: UnifiedTask) -> TaskResult:
        """Execute a task using the FSM agent"""
        if not self._initialized:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                result=None,
                execution_time=0.0,
                agent_id=self.agent_id,
                error="Agent not initialized"
            )

        self.status = AgentStatus.BUSY
        start_time = datetime.now()

        try:
            # Convert unified task to FSM agent input
            user_input = self._convert_task_to_input(task)

            # Execute using FSM agent
            result = await self.fsm_agent.run(user_input)

            execution_time = (datetime.now() - start_time).total_seconds()

            # Convert FSM result to unified result
            unified_result = self._convert_fsm_result(result, task.task_id)

            return TaskResult(
                task_id=task.task_id,
                success=True,
                result=unified_result,
                execution_time=execution_time,
                agent_id=self.agent_id,
                metadata={"fsm_agent": self.name}
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error("FSM agent execution failed: {}", extra={"e": e})

            return TaskResult(
                task_id=task.task_id,
                success=False,
                result=None,
                execution_time=execution_time,
                agent_id=self.agent_id,
                error=str(e)
            )
        finally:
            self.status = AgentStatus.IDLE

    def _convert_task_to_input(self, task: UnifiedTask) -> str:
        """Convert unified task to FSM agent input format"""
        # Extract the main question/request from the task payload
        if isinstance(task.payload, dict):
            if 'question' in task.payload:
                return task.payload['question']
            elif 'query' in task.payload:
                return task.payload['query']
            elif 'input' in task.payload:
                return task.payload['input']
            else:
                # Fallback: convert payload to string
                return str(task.payload)
        else:
            return str(task.payload)

    def _convert_fsm_result(self, fsm_result: Any, task_id: str) -> Dict[str, Any]:
        """Convert FSM agent result to unified result format"""
        if isinstance(fsm_result, dict):
            return {
                "task_id": task_id,
                "fsm_result": fsm_result,
                "output": fsm_result.get("output", str(fsm_result)),
                "steps": fsm_result.get("steps", []),
                "final_answer": fsm_result.get("final_answer", "")
            }
        else:
            return {
                "task_id": task_id,
                "fsm_result": str(fsm_result),
                "output": str(fsm_result),
                "steps": [],
                "final_answer": str(fsm_result)
            }

    async def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities"""
        return self.capabilities

    async def get_status(self) -> AgentStatus:
        """Return current agent status"""
        return self.status

    async def shutdown(self) -> bool:
        """Gracefully shutdown the agent"""
        try:
            self.status = AgentStatus.OFFLINE
            self._initialized = False
            logger.info("FSM agent adapter {} shut down", extra={"self_name": self.name})
            return True
        except Exception as e:
            logger.error("Error shutting down FSM agent adapter: {}", extra={"e": e})
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status"""
        return {
            "healthy": self._initialized and self.status != AgentStatus.ERROR,
            "status": self.status.name,
            "agent_type": "FSM",
            "capabilities": [cap.name for cap in self.capabilities],
            "tools_available": len(self.fsm_agent.tools) if hasattr(self.fsm_agent, 'tools') else 0
        }

    async def collaborate(self, other_agent: IUnifiedAgent,
                         task: UnifiedTask) -> TaskResult:
        """Collaborate with another agent on a task"""
        # For now, delegate to the other agent
        # In the future, this could implement more sophisticated collaboration
        return await other_agent.execute(task)

class UnifiedArchitectureBridge:
    """Bridge between existing FSM agents and the enhanced unified architecture"""

    def __init__(self):
        self.adapters: Dict[str, FSMUnifiedAdapter] = {}
        self.platform = None

    async def initialize_platform(self, redis_url: Optional[str] = None):
        """Initialize the enhanced unified platform"""
        from src.unified_architecture.enhanced_platform import EnhancedMultiAgentPlatform

        self.platform = EnhancedMultiAgentPlatform(redis_url)
        await self.platform.initialize()

        logger.info("Unified architecture platform initialized")

    async def register_fsm_agent(self, fsm_agent: FSMReActAgent,
                                agent_id: str, name: str,
                                tags: Optional[List[str]] = None) -> bool:
        """Register an FSM agent with the unified architecture"""
        if not self.platform:
            logger.error("Platform not initialized")
            return False

        # Create adapter
        adapter = FSMUnifiedAdapter(fsm_agent, agent_id, name)

        # Create metadata
        metadata = AgentMetadata(
            agent_id=agent_id,
            name=name,
            version="1.0.0",
            capabilities=await adapter.get_capabilities(),
            tags=tags or ["fsm", "react"],
            created_at=datetime.now(),
            last_seen=datetime.now(),
            status=AgentStatus.IDLE,
            reliability_score=1.0
        )

        # Register with platform
        success = await self.platform.register_agent(adapter, metadata)

        if success:
            self.adapters[agent_id] = adapter
            logger.info("FSM agent {} registered with unified architecture", extra={"name": name})

        return success

    async def submit_task(self, task: UnifiedTask) -> TaskResult:
        """Submit a task to the unified architecture"""
        if not self.platform:
            raise RuntimeError("Platform not initialized")

        return await self.platform.submit_task(task)

    async def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get performance metrics for an agent"""
        if not self.platform:
            return {}

        return await self.platform.get_agent_metrics(agent_id)

    async def create_task_from_query(self, query: str, task_type: str = "general",
                                   priority: int = 5) -> UnifiedTask:
        """Create a unified task from a simple query"""
        return UnifiedTask(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            priority=priority,
            payload={"question": query},
            required_capabilities=[AgentCapability.REASONING, AgentCapability.TOOL_USE],
            deadline=None,
            dependencies=[],
            metadata={"source": "query", "created_at": datetime.now().isoformat()}
        )

    async def shutdown(self):
        """Shutdown the bridge and platform"""
        if self.platform:
            await self.platform.shutdown()

        logger.info("Unified architecture bridge shut down")

# Example usage function
async def example_fsm_integration():
    """Example of integrating FSM agents with the unified architecture"""

    # Create bridge
    bridge = UnifiedArchitectureBridge()
    await bridge.initialize_platform()

    try:
        # Create FSM agent (using your existing implementation)
        from src.tools import (
            file_reader, advanced_file_reader, web_researcher,
            semantic_search_tool, python_interpreter, tavily_search_backoff,
            get_weather, PythonREPLTool
        )

        tools = [
            file_reader, advanced_file_reader, web_researcher,
            semantic_search_tool, python_interpreter, tavily_search_backoff,
            get_weather, PythonREPLTool
        ]

        # Filter out None tools
        tools = [tool for tool in tools if tool is not None]

        fsm_agent = FSMReActAgent(tools=tools)

        # Register FSM agent with unified architecture
        success = await bridge.register_fsm_agent(
            fsm_agent,
            "fsm-agent-001",
            "FSM Reasoning Agent",
            tags=["reasoning", "tools", "fsm"]
        )

        if success:
            logger.info("FSM agent registered successfully")

            # Create and submit a task
            task = await bridge.create_task_from_query(
                "What is the current weather in New York?",
                task_type="weather_query",
                priority=3
            )

            result = await bridge.submit_task(task)
            logger.info("Task result: {}", extra={"result": result})

            # Get agent metrics
            metrics = await bridge.get_agent_metrics("fsm-agent-001")
            logger.info("Agent metrics: {}", extra={"metrics": metrics})

    finally:
        await bridge.shutdown()

if __name__ == "__main__":
    asyncio.run(example_fsm_integration())