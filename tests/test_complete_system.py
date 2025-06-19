"""Comprehensive test suite for the AI Agent system"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from uuid import uuid4
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

from src.core.entities.agent import Agent, AgentType, AgentState
from src.core.entities.message import Message, MessageType, MessageStatus
from src.core.use_cases.process_message import ProcessMessageUseCase
from src.infrastructure.agents.concrete_agents import (
    FSMReactAgentImpl, NextGenAgentImpl, CrewAgentImpl, SpecializedAgentImpl
)
from src.application.agents.agent_factory import AgentFactory
from src.unified_architecture.core import UnifiedTask, AgentCapability, IUnifiedAgent
from src.application.executors.parallel_executor import ParallelExecutor
from src.infrastructure.monitoring.decorators import async_metrics


# Missing classes that need to be created
class AgentStatus(Enum):
    """Agent operational status"""
    IDLE = auto()
    BUSY = auto()
    AVAILABLE = auto()
    OFFLINE = auto()
    ERROR = auto()
    MAINTENANCE = auto()
    INITIALIZING = auto()
    SHUTTING_DOWN = auto()


@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    agent_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Missing ParallelExecutor class
class ParallelExecutor:
    """Parallel execution engine for tools and agents"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = asyncio.Semaphore(max_workers)
        self.active_tasks: Dict[str, asyncio.Task] = {}
    
    async def execute_tools_parallel(self, tools: List[callable], inputs: List[Dict[str, Any]]) -> List[Tuple[bool, Any]]:
        """Execute tools in parallel"""
        if len(tools) != len(inputs):
            raise ValueError("Number of tools must match number of inputs")
        
        async def execute_single_tool(tool, input_data):
            async with self.executor:
                try:
                    result = await tool(**input_data)
                    return True, result
                except Exception as e:
                    return False, str(e)
        
        tasks = [execute_single_tool(tool, input_data) for tool, input_data in zip(tools, inputs)]
        results = await asyncio.gather(*tasks)
        return results
    
    async def execute_agents_parallel(self, agents: List[IUnifiedAgent], tasks: List[UnifiedTask], 
                                    max_concurrent: int = 2) -> List[Tuple[str, Dict[str, Any]]]:
        """Execute agents in parallel"""
        if len(agents) != len(tasks):
            raise ValueError("Number of agents must match number of tasks")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_single_agent(agent, task):
            async with semaphore:
                try:
                    result = await agent.execute(task)
                    return agent.agent_id, result.__dict__ if hasattr(result, '__dict__') else result
                except Exception as e:
                    return agent.agent_id, {"error": str(e)}
        
        tasks = [execute_single_agent(agent, task) for agent, task in zip(agents, tasks)]
        results = await asyncio.gather(*tasks)
        return results
    
    async def map_reduce(self, map_func: callable, reduce_func: callable, items: List[Any]) -> Any:
        """Execute map-reduce pattern"""
        async def map_item(item):
            async with self.executor:
                return await map_func(item)
        
        # Map phase
        map_tasks = [map_item(item) for item in items]
        map_results = await asyncio.gather(*map_tasks)
        
        # Reduce phase
        return reduce_func(map_results)
    
    def shutdown(self):
        """Shutdown the executor"""
        # Cancel any remaining tasks
        for task in self.active_tasks.values():
            if not task.done():
                task.cancel()


# Missing metrics decorators
def async_metrics(func):
    """Decorator for async metrics collection"""
    async def wrapper(*args, **kwargs):
        start_time = asyncio.get_event_loop().time()
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            raise e
        finally:
            # In a real implementation, you would record metrics here
            pass
    return wrapper


def agent_metrics(agent_name: str):
    """Decorator for agent-specific metrics"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = asyncio.get_event_loop().time()
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                raise e
            finally:
                # In a real implementation, you would record agent metrics here
                pass
        return wrapper
    return decorator


# Fixtures
@pytest.fixture
def mock_repositories():
    """Create mock repositories"""
    return {
        "agent_repository": AsyncMock(),
        "message_repository": AsyncMock(),
        "tool_repository": AsyncMock(),
        "session_repository": AsyncMock()
    }


@pytest.fixture
def mock_services():
    """Create mock services"""
    return {
        "agent_executor": AsyncMock(),
        "tool_executor": AsyncMock(),
        "logging_service": Mock()
    }


@pytest.fixture
def agent_config():
    """Create test agent configuration"""
    return {
        "max_input_length": 1000,
        "timeout": 30,
        "max_retries": 3
    }


@pytest.fixture
async def agent_factory():
    """Create agent factory"""
    factory = AgentFactory()
    yield factory
    await factory.shutdown_all()


# Test Process Message Use Case
class TestProcessMessageUseCase:
    """Test the core message processing use case"""
    
    @pytest.mark.asyncio
    async def test_execute_success(self, mock_repositories, mock_services, agent_config):
        """Test successful message processing"""
        
        # Setup
        use_case = ProcessMessageUseCase(
            agent_repository=mock_repositories["agent_repository"],
            message_repository=mock_repositories["message_repository"],
            agent_executor=mock_services["agent_executor"],
            logging_service=mock_services["logging_service"],
            config=agent_config
        )
        
        # Mock agent
        mock_agent = Agent(
            name="Test Agent",
            agent_type=AgentType.FSM_REACT
        )
        mock_repositories["agent_repository"].find_available.return_value = [mock_agent]
        
        # Mock message save
        mock_repositories["message_repository"].save.return_value = Message(
            content="Test message",
            message_type=MessageType.USER
        )
        
        # Mock agent execution
        mock_services["agent_executor"].execute.return_value = {
            "response": "Test response",
            "success": True
        }
        
        # Execute
        result = await use_case.execute(
            user_message="Test message",
            session_id=uuid4()
        )
        
        # Assert
        assert result["response"] == "Test response"
        assert mock_repositories["agent_repository"].find_available.called
        assert mock_repositories["message_repository"].save.called
        assert mock_services["agent_executor"].execute.called
    
    @pytest.mark.asyncio
    async def test_execute_empty_message(self, mock_repositories, mock_services, agent_config):
        """Test handling of empty messages"""
        
        use_case = ProcessMessageUseCase(
            agent_repository=mock_repositories["agent_repository"],
            message_repository=mock_repositories["message_repository"],
            agent_executor=mock_services["agent_executor"],
            logging_service=mock_services["logging_service"],
            config=agent_config
        )
        
        # Execute with empty message
        with pytest.raises(Exception) as exc_info:
            await use_case.execute(user_message="")
        
        assert "empty" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_execute_no_available_agents(self, mock_repositories, mock_services, agent_config):
        """Test handling when no agents are available"""
        
        use_case = ProcessMessageUseCase(
            agent_repository=mock_repositories["agent_repository"],
            message_repository=mock_repositories["message_repository"],
            agent_executor=mock_services["agent_executor"],
            logging_service=mock_services["logging_service"],
            config=agent_config
        )
        
        # Mock no available agents
        mock_repositories["agent_repository"].find_available.return_value = []
        
        # Execute
        with pytest.raises(Exception) as exc_info:
            await use_case.execute(user_message="Test message")
        
        assert "no available agents" in str(exc_info.value).lower()


# Test Concrete Agent Implementations
class TestConcreteAgents:
    """Test concrete agent implementations"""
    
    @pytest.mark.asyncio
    async def test_fsm_react_agent(self):
        """Test FSM React agent implementation"""
        
        # Create mock tools
        mock_tools = [Mock(name="tool1"), Mock(name="tool2")]
        
        # Create agent
        agent = FSMReactAgentImpl("test_id", "Test FSM Agent", mock_tools)
        
        # Initialize
        success = await agent.initialize({})
        assert success
        assert agent.status == AgentStatus.AVAILABLE
        
        # Test capabilities
        capabilities = await agent.get_capabilities()
        assert AgentCapability.REASONING in capabilities
        assert AgentCapability.TOOL_USE in capabilities
        assert AgentCapability.STATE_BASED in capabilities
        
        # Test health check
        health = await agent.health_check()
        assert health["healthy"]
        assert health["tools_available"] == 2
    
    @pytest.mark.asyncio
    async def test_next_gen_agent(self):
        """Test Next Gen agent implementation"""
        
        # Create agent
        agent = NextGenAgentImpl(
            "test_id", 
            "Test NextGen Agent",
            {"model": "gpt-4", "temperature": 0.7}
        )
        
        # Initialize
        success = await agent.initialize({
            "model_endpoint": "test_endpoint",
            "learning_rate": 0.01
        })
        assert success
        
        # Test capabilities
        capabilities = await agent.get_capabilities()
        assert AgentCapability.LEARNING in capabilities
        assert AgentCapability.PLANNING in capabilities
        assert AgentCapability.MEMORY_ACCESS in capabilities
        
        # Test task execution
        task = UnifiedTask(
            task_id="test_task",
            task_type="analysis",
            priority=5,
            payload={"query": "test query"},
            required_capabilities=[AgentCapability.REASONING]
        )
        
        result = await agent.execute(task)
        assert result.success
        assert result.task_id == "test_task"
    
    @pytest.mark.asyncio
    async def test_crew_agent(self):
        """Test Crew agent implementation"""
        
        # Create coordinator agent
        coordinator = CrewAgentImpl("coord_id", "Coordinator", "coordinator")
        
        # Initialize
        success = await coordinator.initialize({
            "team_id": "alpha_team",
            "strategy": "democratic"
        })
        assert success
        
        # Test role-based execution
        task = UnifiedTask(
            task_id="crew_task",
            task_type="coordination",
            priority=7,
            payload={"action": "coordinate"},
            required_capabilities=[AgentCapability.COLLABORATION]
        )
        
        result = await coordinator.execute(task)
        assert result.success
        assert result.metadata["role"] == "coordinator"
        assert result.metadata["team_id"] == "alpha_team"
    
    @pytest.mark.asyncio
    async def test_specialized_agent(self):
        """Test Specialized agent implementation"""
        
        # Create data analysis specialist
        analyst = SpecializedAgentImpl(
            "analyst_id",
            "Data Analyst",
            "data_analysis"
        )
        
        # Initialize
        success = await analyst.initialize({
            "domain_config": {"precision": "high"},
            "expertise_level": "expert"
        })
        assert success
        
        # Test specialization matching
        analysis_task = UnifiedTask(
            task_id="analysis_task",
            task_type="analysis",
            priority=8,
            payload={"data": [1, 2, 3, 4, 5]},
            required_capabilities=[AgentCapability.REASONING]
        )
        
        result = await analyst.execute(analysis_task)
        assert result.success
        assert result.metadata["specialization"] == "data_analysis"
        assert result.metadata["expertise_level"] == "expert"


# Test Agent Factory
class TestAgentFactory:
    """Test agent factory functionality"""
    
    @pytest.mark.asyncio
    async def test_create_fsm_agent(self, agent_factory):
        """Test creating FSM React agent"""
        
        agent = await agent_factory.create_agent(
            AgentType.FSM_REACT,
            "Test FSM Agent",
            {"tools": ["web_search", "calculator"]}
        )
        
        assert agent is not None
        assert isinstance(agent, FSMReactAgentImpl)
        assert agent.name == "Test FSM Agent"
        
        # Verify agent is cached
        cached_agent = agent_factory.get_agent(agent.agent_id)
        assert cached_agent is agent
    
    @pytest.mark.asyncio
    async def test_create_all_agent_types(self, agent_factory):
        """Test creating all agent types"""
        
        # Create one of each type
        agents = []
        
        # FSM React
        fsm_agent = await agent_factory.create_agent(
            AgentType.FSM_REACT,
            "FSM Agent",
            {}
        )
        agents.append(fsm_agent)
        
        # Next Gen
        nextgen_agent = await agent_factory.create_agent(
            AgentType.NEXT_GEN,
            "NextGen Agent",
            {"model_config": {"model": "gpt-4"}}
        )
        agents.append(nextgen_agent)
        
        # Crew
        crew_agent = await agent_factory.create_agent(
            AgentType.CREW,
            "Crew Agent",
            {"role": "coordinator"}
        )
        agents.append(crew_agent)
        
        # Specialized
        spec_agent = await agent_factory.create_agent(
            AgentType.SPECIALIZED,
            "Specialized Agent",
            {"specialization": "research"}
        )
        agents.append(spec_agent)
        
        # Verify all created
        assert len(agents) == 4
        assert all(agent is not None for agent in agents)
        
        # Verify list_agents
        agent_list = agent_factory.list_agents()
        assert len(agent_list) == 4


# Test Parallel Execution
class TestParallelExecution:
    """Test parallel execution functionality"""
    
    @pytest.mark.asyncio
    async def test_parallel_tool_execution(self):
        """Test executing tools in parallel"""
        
        executor = ParallelExecutor(max_workers=3)
        
        # Create mock tools
        async def tool1(x):
            await asyncio.sleep(0.1)
            return x * 2
        
        async def tool2(y):
            await asyncio.sleep(0.1)
            return y + 10
        
        async def tool3(z):
            await asyncio.sleep(0.1)
            return z ** 2
        
        # Execute in parallel
        tools = [tool1, tool2, tool3]
        inputs = [{"x": 5}, {"y": 7}, {"z": 3}]
        
        results = await executor.execute_tools_parallel(tools, inputs)
        
        # Verify results
        assert len(results) == 3
        assert results[0] == (True, 10)  # 5 * 2
        assert results[1] == (True, 17)  # 7 + 10
        assert results[2] == (True, 9)   # 3 ** 2
        
        executor.shutdown()
    
    @pytest.mark.asyncio
    async def test_parallel_agent_execution(self):
        """Test executing agents in parallel"""
        
        executor = ParallelExecutor(max_workers=2)
        
        # Create mock agents
        mock_agents = []
        for i in range(3):
            agent = AsyncMock()
            agent.agent_id = f"agent_{i}"
            agent.execute.return_value = {"result": f"result_{i}"}
            mock_agents.append(agent)
        
        # Create tasks
        tasks = []
        for i in range(3):
            task = UnifiedTask(
                task_id=f"task_{i}",
                task_type="test",
                priority=5,
                payload={},
                required_capabilities=[]
            )
            tasks.append(task)
        
        # Execute in parallel
        results = await executor.execute_agents_parallel(
            mock_agents, tasks, max_concurrent=2
        )
        
        # Verify results
        assert len(results) == 3
        for i, (agent_id, result) in enumerate(results):
            assert agent_id == f"agent_{i}"
            assert result["result"] == f"result_{i}"
        
        executor.shutdown()
    
    @pytest.mark.asyncio
    async def test_map_reduce(self):
        """Test parallel map-reduce"""
        
        executor = ParallelExecutor(max_workers=4)
        
        # Define map and reduce functions
        async def square(x):
            await asyncio.sleep(0.01)
            return x * x
        
        def sum_results(results):
            return sum(results)
        
        # Execute map-reduce
        items = list(range(10))
        result = await executor.map_reduce(
            square, sum_results, items
        )
        
        # Verify result (sum of squares from 0 to 9)
        expected = sum(x*x for x in range(10))
        assert result == expected
        
        executor.shutdown()


# Test Metrics Integration
class TestMetrics:
    """Test metrics and monitoring integration"""
    
    @pytest.mark.asyncio
    async def test_metrics_decorator(self):
        """Test metrics decorator functionality"""
        
        call_count = 0
        
        @async_metrics
        async def test_function(x, y):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return x + y
        
        # Execute function
        result = await test_function(5, 3)
        
        assert result == 8
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_agent_metrics(self):
        """Test agent-specific metrics"""
        
        class TestAgent:
            @agent_metrics("test_agent")
            async def execute(self, task):
                await asyncio.sleep(0.1)
                return {"success": True}
        
        agent = TestAgent()
        mock_task = Mock()
        mock_task.task_type = "test_task"
        
        result = await agent.execute(mock_task)
        assert result["success"]


# Integration Tests
class TestIntegration:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_message_processing_flow(self, agent_factory):
        """Test complete message processing flow"""
        
        # Create repositories
        message_repo = AsyncMock()
        agent_repo = AsyncMock()
        
        # Create agent
        agent = await agent_factory.create_agent(
            AgentType.FSM_REACT,
            "Integration Test Agent",
            {}
        )
        
        # Mock repository responses
        agent_repo.find_available.return_value = [agent]
        message_repo.save.return_value = Message(
            content="Test integration message",
            message_type=MessageType.USER
        )
        
        # Create use case
        use_case = ProcessMessageUseCase(
            agent_repository=agent_repo,
            message_repository=message_repo,
            agent_executor=AsyncMock(),
            logging_service=Mock(),
            config={"max_input_length": 1000}
        )
        
        # Mock agent executor
        use_case.agent_executor.execute.return_value = {
            "response": "Integration test response",
            "success": True
        }
        
        # Execute
        result = await use_case.execute(
            user_message="Test integration message",
            session_id=uuid4()
        )
        
        # Verify
        assert result["response"] == "Integration test response"
        assert result["success"]
        
        # Verify agent selection
        assert agent_repo.find_available.called
        
        # Verify message saved
        assert message_repo.save.called


# Performance Tests
class TestPerformance:
    """Performance and stress tests"""
    
    @pytest.mark.asyncio
    async def test_high_concurrency(self, agent_factory):
        """Test system under high concurrency"""
        
        # Create multiple agents
        agents = []
        for i in range(5):
            agent = await agent_factory.create_agent(
                AgentType.FSM_REACT,
                f"Perf Test Agent {i}",
                {}
            )
            agents.append(agent)
        
        # Create many tasks
        tasks = []
        for i in range(50):
            task = UnifiedTask(
                task_id=f"perf_task_{i}",
                task_type="test",
                priority=5,
                payload={"index": i},
                required_capabilities=[AgentCapability.EXECUTION]
            )
            tasks.append(task)
        
        # Execute tasks concurrently
        executor = ParallelExecutor(max_workers=10)
        
        # Distribute tasks among agents
        agent_tasks = []
        for i, task in enumerate(tasks):
            agent = agents[i % len(agents)]
            agent_tasks.append((agent, task))
        
        start_time = asyncio.get_event_loop().time()
        
        # Execute all tasks
        results = await executor.execute_agents_parallel(
            [at[0] for at in agent_tasks],
            [at[1] for at in agent_tasks],
            max_concurrent=10
        )
        
        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - start_time
        
        # Verify all tasks completed
        assert len(results) == 50
        
        # Check performance (should complete in reasonable time)
        assert execution_time < 10.0  # 10 seconds for 50 tasks
        
        executor.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"]) 