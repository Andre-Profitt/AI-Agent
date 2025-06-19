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
from src.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker, 
    CircuitBreakerConfig, 
    circuit_breaker,
    CircuitBreakerOpenError
)
from src.infrastructure.workflow.workflow_engine import (
    AgentOrchestrator, 
    WorkflowEngine, 
    WorkflowStatus,
    WorkflowDefinition,
    WorkflowStep,
    WorkflowType
)
from src.agents.advanced_agent_fsm import FSMReActAgent
from src.tools.base_tool import BaseTool


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


@pytest.mark.asyncio
async def test_circuit_breaker_integration():
    """Test complete circuit breaker integration"""
    config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60)
    breaker = CircuitBreaker("test_breaker", config)
    
    # Mock a failing database operation
    async def mock_db_op():
        raise Exception("Database connection failed")
    
    # Test that circuit opens after threshold
    for _ in range(3):
        with pytest.raises(Exception):
            await breaker.call(mock_db_op)
    
    # Verify circuit is now open
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call(mock_db_op)


@pytest.mark.asyncio
async def test_circuit_breaker_decorator():
    """Test circuit breaker decorator"""
    
    @circuit_breaker("test_decorator", CircuitBreakerConfig(failure_threshold=2, recovery_timeout=30))
    async def failing_function():
        raise Exception("Test failure")
    
    # Test that circuit opens after threshold
    for _ in range(2):
        with pytest.raises(Exception):
            await failing_function()
    
    # Verify circuit is now open
    with pytest.raises(CircuitBreakerOpenError):
        await failing_function()


@pytest.mark.asyncio
async def test_workflow_orchestration():
    """Test workflow engine state transitions"""
    engine = WorkflowEngine()
    
    # Create a simple workflow
    step1 = WorkflowStep(
        step_id="step_1",
        name="Test Step 1",
        description="First test step",
        tool_name="mock_tool"
    )
    
    workflow = WorkflowDefinition(
        workflow_id="test_workflow",
        name="Test Workflow",
        description="A test workflow",
        workflow_type=WorkflowType.SEQUENTIAL,
        steps=[step1],
        timeout=60
    )
    
    # Register workflow
    await engine.register_workflow(workflow)
    
    # Register mock tool
    mock_tool = MockTool()
    await engine.register_tool("mock_tool", mock_tool)
    
    # Execute workflow
    execution = await engine.execute_workflow("test_workflow", {"query": "test"})
    
    assert execution.status == WorkflowStatus.COMPLETED
    assert execution.workflow_id == "test_workflow"


@pytest.mark.asyncio
async def test_agent_orchestrator_integration():
    """Test agent orchestrator with FSM agent integration"""
    orchestrator = AgentOrchestrator()
    
    # Create mock FSM agent
    mock_tools = [MockTool()]
    fsm_agent = FSMReActAgent(tools=mock_tools)
    
    # Register FSM agent
    await orchestrator.register_fsm_agent("test_agent", fsm_agent)
    
    # Create workflow steps
    workflow_steps = [
        {
            "name": "Test Step",
            "description": "A test step",
            "agent_id": "test_agent",
            "type": "agent"
        }
    ]
    
    # Create workflow
    workflow_id = await orchestrator.create_workflow_from_fsm(
        "test_workflow", fsm_agent, workflow_steps
    )
    
    assert workflow_id == "test_workflow"
    assert "test_workflow" in orchestrator.get_available_workflows()
    assert "test_agent" in orchestrator.get_fsm_agents()


@pytest.mark.asyncio
async def test_fsm_agent_workflow_execution():
    """Test FSM agent's workflow execution capability"""
    mock_tools = [MockTool()]
    fsm_agent = FSMReActAgent(tools=mock_tools)
    
    # Wait for orchestrator initialization
    await asyncio.sleep(0.1)
    
    # Create workflow steps
    workflow_steps = [
        {
            "name": "Tool Step",
            "description": "Execute a tool",
            "tool_name": "mock_tool",
            "type": "tool",
            "params": {"query": "test query"}
        }
    ]
    
    # Execute workflow
    result = await fsm_agent.execute_workflow(workflow_steps, "test query")
    
    assert result['success'] is True
    assert result['workflow_id'] is not None
    assert result['execution_id'] is not None


@pytest.mark.asyncio
async def test_circuit_breaker_recovery():
    """Test circuit breaker recovery after timeout"""
    config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1)  # Short timeout for testing
    breaker = CircuitBreaker("recovery_test", config)
    
    # Mock a function that fails then succeeds
    call_count = 0
    async def mock_function():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise Exception("Temporary failure")
        return "success"
    
    # Fail twice to open circuit
    for _ in range(2):
        with pytest.raises(Exception):
            await breaker.call(mock_function)
    
    # Circuit should be open
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.call(mock_function)
    
    # Wait for recovery timeout
    await asyncio.sleep(0.2)
    
    # Circuit should be half-open and allow the call
    result = await breaker.call(mock_function)
    assert result == "success"


@pytest.mark.asyncio
async def test_workflow_engine_error_handling():
    """Test workflow engine error handling"""
    engine = WorkflowEngine()
    
    # Create a workflow with a non-existent tool
    step1 = WorkflowStep(
        step_id="step_1",
        name="Failing Step",
        description="A step that will fail",
        tool_name="non_existent_tool"
    )
    
    workflow = WorkflowDefinition(
        workflow_id="failing_workflow",
        name="Failing Workflow",
        description="A workflow that will fail",
        workflow_type=WorkflowType.SEQUENTIAL,
        steps=[step1],
        timeout=60
    )
    
    # Register workflow
    await engine.register_workflow(workflow)
    
    # Execute workflow - should fail but not crash
    execution = await engine.execute_workflow("failing_workflow", {"query": "test"})
    
    assert execution.status == WorkflowStatus.FAILED
    assert execution.error_message is not None


@pytest.mark.asyncio
async def test_parallel_workflow_execution():
    """Test parallel workflow execution"""
    engine = WorkflowEngine()
    
    # Create parallel steps
    step1 = WorkflowStep(
        step_id="step_1",
        name="Parallel Step 1",
        description="First parallel step",
        tool_name="mock_tool",
        parallel=True
    )
    
    step2 = WorkflowStep(
        step_id="step_2", 
        name="Parallel Step 2",
        description="Second parallel step",
        tool_name="mock_tool",
        parallel=True
    )
    
    workflow = WorkflowDefinition(
        workflow_id="parallel_workflow",
        name="Parallel Workflow",
        description="A workflow with parallel steps",
        workflow_type=WorkflowType.PARALLEL,
        steps=[step1, step2],
        timeout=60
    )
    
    # Register workflow and tool
    await engine.register_workflow(workflow)
    mock_tool = MockTool()
    await engine.register_tool("mock_tool", mock_tool)
    
    # Execute workflow
    execution = await engine.execute_workflow("parallel_workflow", {"query": "test"})
    
    assert execution.status == WorkflowStatus.COMPLETED


@pytest.mark.asyncio
async def test_workflow_cancellation():
    """Test workflow cancellation"""
    engine = WorkflowEngine()
    
    # Create a long-running workflow
    step1 = WorkflowStep(
        step_id="step_1",
        name="Long Step",
        description="A step that takes time",
        tool_name="mock_tool"
    )
    
    workflow = WorkflowDefinition(
        workflow_id="long_workflow",
        name="Long Workflow",
        description="A workflow that takes time",
        workflow_type=WorkflowType.SEQUENTIAL,
        steps=[step1],
        timeout=60
    )
    
    # Register workflow
    await engine.register_workflow(workflow)
    
    # Start execution
    execution_task = asyncio.create_task(
        engine.execute_workflow("long_workflow", {"query": "test"})
    )
    
    # Cancel execution
    await asyncio.sleep(0.1)  # Let it start
    cancelled = await engine.cancel_execution(execution_task.get_name())
    
    assert cancelled is True


class MockTool(BaseTool):
    """Mock tool for testing"""
    name = "mock_tool"
    description = "A mock tool for testing"
    
    def _run(self, query: str) -> str:
        return f"Mock result for: {query}"
    
    async def _arun(self, query: str) -> str:
        return f"Async mock result for: {query}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"]) 