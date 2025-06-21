from examples.enhanced_unified_example import health
from examples.enhanced_unified_example import metrics
from examples.enhanced_unified_example import registered_agents
from examples.enhanced_unified_example import task
from examples.integration.unified_architecture_example import analysis_agents
from examples.integration.unified_architecture_example import memory_entry
from examples.integration.unified_architecture_example import requirements
from examples.integration.unified_architecture_example import task_ids
from examples.parallel_execution_example import agents
from examples.parallel_execution_example import results
from migrations.env import config
from performance_dashboard import stats
from setup_environment import components
from setup_environment import value
from tests.load_test import data
from tests.load_test import success

from src.adapters.fsm_unified_adapter import capabilities
from src.api_server import conflict
from src.api_server import manager
from src.api_server import message
from src.core.entities.agent import Agent
from src.core.monitoring import key
from src.core.monitoring import memory
from src.database.models import agent_id
from src.database.models import metadata
from src.database.models import status
from src.gaia_components.multi_agent_orchestrator import broadcast_message
from src.gaia_components.multi_agent_orchestrator import task_id
from src.gaia_components.performance_optimization import entry
from src.tools_introspection import description
from src.tools_introspection import name
from src.unified_architecture.conflict_resolution import resolution
from src.unified_architecture.marketplace import listing
from src.unified_architecture.marketplace import listings_data
from src.unified_architecture.marketplace import ratings
from src.unified_architecture.resource_management import allocation
from src.unified_architecture.resource_management import utilization
from src.unified_architecture.task_distribution import selected_agent

from src.agents.advanced_agent_fsm import AgentCapability

from src.agents.advanced_agent_fsm import AgentStatus

from src.agents.advanced_agent_fsm import Agent

from src.agents.advanced_agent_fsm import MultiAgentPlatform

from src.agents.advanced_agent_fsm import IUnifiedAgent

from src.gaia_components.enhanced_memory_system import MemoryType

from src.agents.advanced_agent_fsm import AgentMetadata

from src.agents.advanced_agent_fsm import AgentMessage

from src.gaia_components.enhanced_memory_system import MemoryEntry
# TODO: Fix undefined variables: Any, List, TaskType, agent_id, agents, agents_data, allocation, analysis_agents, analysis_listings, broadcast_message, capabilities, comm, components, config, conflict, created_listing, data, datetime, description, distributor, e, engine, entry, experience_entry, experience_results, health, i, key, knowledge_entry, knowledge_results, listing, listings_data, manager, market, memory, memory_entry, message, metadata, metrics, ml_agents, name, notifications, rating, ratings, reg, registered_agents, requirements, resolution, resolver, result, results, retrieved_value, selected_agent, stats, status, success, tags, task, task_descriptions, task_id, task_ids, tracker, utilization, uuid4, value
from tests.test_gaia_agent import agent
import dash

from src.core.entities.agent import AgentCapability
from src.core.entities.agent import AgentMetadata


"""
import datetime
from typing import List
from datetime import datetime
from src.gaia_components.multi_agent_orchestrator import TaskPriority
from src.infrastructure.monitoring.metrics import PerformanceTracker
from src.unified_architecture.communication import AgentMessage
from src.unified_architecture.communication import CommunicationProtocol
from src.unified_architecture.communication import MessageType
from src.unified_architecture.conflict_resolution import Conflict
from src.unified_architecture.conflict_resolution import ConflictResolver
from src.unified_architecture.conflict_resolution import ConflictType
from src.unified_architecture.enhanced_platform import AgentCapability
from src.unified_architecture.enhanced_platform import AgentMetadata
from src.unified_architecture.enhanced_platform import AgentStatus
from src.unified_architecture.enhanced_platform import IUnifiedAgent
from src.unified_architecture.enhanced_platform import OrchestrationEngine
from src.unified_architecture.enhanced_platform import TaskResult
from src.unified_architecture.enhanced_platform import UnifiedTask
from src.unified_architecture.marketplace import AgentListing
from src.unified_architecture.marketplace import AgentMarketplace
from src.unified_architecture.marketplace import AgentRating
from src.unified_architecture.marketplace import ListingStatus
from src.unified_architecture.marketplace import RatingCategory
from src.unified_architecture.platform import PlatformConfig
from src.unified_architecture.resource_management import ResourceManager
from src.unified_architecture.resource_management import ResourceType
from src.unified_architecture.shared_memory import MemoryEntry
from src.unified_architecture.shared_memory import MemoryType
from src.unified_architecture.shared_memory import SharedMemorySystem
from src.unified_architecture.task_distribution import TaskDistributionStrategy
from src.unified_architecture.task_distribution import TaskDistributor
# TODO: Fix undefined variables: TaskType, agent, agent_id, agents, agents_data, allocation, analysis_agents, analysis_listings, broadcast_message, capabilities, comm, components, config, conflict, created_listing, dash, data, description, distributor, e, engine, entry, experience_entry, experience_results, health, i, key, knowledge_entry, knowledge_results, listing, listings_data, manager, market, memory, memory_entry, message, metadata, metrics, ml_agents, name, notifications, rating, ratings, reg, registered_agents, requirements, resolution, resolver, result, results, retrieved_value, selected_agent, self, stats, status, success, tags, task, task_descriptions, task_id, task_ids, tracker, utilization, value

from fastapi import status
Comprehensive Test Suite for Unified Architecture

This test suite covers all components of the unified architecture:
- Core interfaces and data structures
- Orchestration engine
- State management
- Communication protocol
- Resource management
- Agent registry
- Task distribution
- Shared memory system
- Conflict resolution
- Performance tracking
- Collaboration dashboard
- Agent marketplace
- Multi-agent platform
"""

from typing import Any

import asyncio
import pytest

from uuid import uuid4

from src.unified_architecture.core import (
    AgentCapability, AgentStatus, AgentMetadata,
    UnifiedTask, TaskResult, TaskPriority, TaskType,
    IUnifiedAgent
)
from src.unified_architecture.orchestration import OrchestrationEngine
from src.unified_architecture.state_management import StateManager
from src.unified_architecture.communication import (
    CommunicationProtocol, AgentMessage, MessageType
)
from src.unified_architecture.resource_management import (
    ResourceManager, ResourceType, ResourceAllocation
)
from src.unified_architecture.registry import AgentRegistry
from src.unified_architecture.task_distribution import (
    TaskDistributor, TaskDistributionStrategy
)
from src.unified_architecture.shared_memory import (
    SharedMemorySystem, MemoryType, MemoryEntry
)
from src.unified_architecture.conflict_resolution import (
    ConflictResolver, Conflict, ConflictType
)
from src.unified_architecture.performance import PerformanceTracker
from src.unified_architecture.dashboard import CollaborationDashboard
from src.unified_architecture.marketplace import (
    AgentMarketplace, AgentListing, ListingStatus, AgentRating, RatingCategory
)
from src.unified_architecture.platform import MultiAgentPlatform, PlatformConfig

class MockAgent(IUnifiedAgent):
    """Mock agent for testing"""

    def __init__(self, agent_id: str, name: str, capabilities: List[AgentCapability]):
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities
        self.status = AgentStatus.IDLE
        self.executed_tasks = []

    async def execute_task(self, task: UnifiedTask) -> TaskResult:
        """Execute a task"""
        try:
            self.status = AgentStatus.BUSY
            self.executed_tasks.append(task.id)

            # Simulate task execution
            await asyncio.sleep(0.1)

            result = TaskResult(
                task_id=task.id,
                success=True,
                data={"result": f"Task {task.id} completed by {self.name}"},
                metadata={"agent": self.name, "execution_time": 0.1}
            )

            self.status = AgentStatus.IDLE
            return result

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

class TestCore:
    """Test core interfaces and data structures"""

    def test_agent_capability_enum(self):
        """Test AgentCapability enum"""
        assert AgentCapability.DATA_ANALYSIS.value == "data_analysis"
        assert AgentCapability.MACHINE_LEARNING.value == "machine_learning"
        assert len(AgentCapability) > 0

    def test_agent_status_enum(self):
        """Test AgentStatus enum"""
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.BUSY.value == "busy"
        assert AgentStatus.ERROR.value == "error"

    def test_agent_metadata_creation(self):
        """Test AgentMetadata creation"""
        metadata = AgentMetadata(
            agent_id="test-agent-001",
            name="Test Agent",
            capabilities=[AgentCapability.DATA_ANALYSIS],
            status=AgentStatus.IDLE,
            version="1.0.0",
            description="Test agent",
            tags=["test"],
            created_at=datetime.utcnow()
        )

        assert metadata.agent_id == "test-agent-001"
        assert metadata.name == "Test Agent"
        assert len(metadata.capabilities) == 1
        assert metadata.capabilities[0] == AgentCapability.DATA_ANALYSIS

    def test_unified_task_creation(self):
        """Test UnifiedTask creation"""
        task = UnifiedTask(
            id=uuid4(),
            description="Test task",
            task_type=TaskType.ANALYSIS,
            priority=TaskPriority.HIGH,
            requirements={"capabilities": [AgentCapability.DATA_ANALYSIS]},
            dependencies=[],
            metadata={"test": True}
        )

        assert task.description == "Test task"
        assert task.task_type == TaskType.ANALYSIS
        assert task.priority == TaskPriority.HIGH

    def test_task_result_creation(self):
        """Test TaskResult creation"""
        task_id = uuid4()
        result = TaskResult(
            task_id=task_id,
            success=True,
            data={"result": "success"},
            metadata={"execution_time": 1.0}
        )

        assert result.task_id == task_id
        assert result.success is True
        assert result.data["result"] == "success"

class TestMockAgent:
    """Test mock agent implementation"""

    @pytest.mark.asyncio
    async def test_mock_agent_execution(self):
        """Test mock agent task execution"""
        agent = MockAgent(
            "test-agent-001",
            "Test Agent",
            [AgentCapability.DATA_ANALYSIS]
        )

        task = UnifiedTask(
            id=uuid4(),
            description="Test task",
            task_type=TaskType.ANALYSIS,
            priority=TaskPriority.MEDIUM,
            requirements={"capabilities": [AgentCapability.DATA_ANALYSIS]},
            dependencies=[],
            metadata={}
        )

        result = await agent.execute_task(task)

        assert result.success is True
        assert result.task_id == task.id
        assert len(agent.executed_tasks) == 1
        assert agent.executed_tasks[0] == task.id

    @pytest.mark.asyncio
    async def test_mock_agent_status(self):
        """Test mock agent status management"""
        agent = MockAgent(
            "test-agent-001",
            "Test Agent",
            [AgentCapability.DATA_ANALYSIS]
        )

        assert await agent.get_status() == AgentStatus.IDLE
        assert await agent.get_capabilities() == [AgentCapability.DATA_ANALYSIS]

class TestOrchestrationEngine:
    """Test orchestration engine"""

    @pytest.fixture
    async def orchestration_engine(self):
        """Create orchestration engine fixture"""
        engine = OrchestrationEngine()
        yield engine
        # Cleanup
        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_agent_registration(self, orchestration_engine):
        """Test agent registration"""
        agent = MockAgent(
            "test-agent-001",
            "Test Agent",
            [AgentCapability.DATA_ANALYSIS]
        )

        metadata = AgentMetadata(
            agent_id="test-agent-001",
            name="Test Agent",
            capabilities=[AgentCapability.DATA_ANALYSIS],
            status=AgentStatus.IDLE,
            version="1.0.0",
            description="Test agent",
            tags=["test"],
            created_at=datetime.utcnow()
        )

        success = await orchestration_engine.register_agent(agent, metadata)
        assert success is True

        # Verify registration
        registered_agents = await orchestration_engine.get_registered_agents()
        assert len(registered_agents) == 1
        assert registered_agents[0].agent_id == "test-agent-001"

    @pytest.mark.asyncio
    async def test_task_submission_and_execution(self, orchestration_engine):
        """Test task submission and execution"""
        # Register agent
        agent = MockAgent(
            "test-agent-001",
            "Test Agent",
            [AgentCapability.DATA_ANALYSIS]
        )

        metadata = AgentMetadata(
            agent_id="test-agent-001",
            name="Test Agent",
            capabilities=[AgentCapability.DATA_ANALYSIS],
            status=AgentStatus.IDLE,
            version="1.0.0",
            description="Test agent",
            tags=["test"],
            created_at=datetime.utcnow()
        )

        await orchestration_engine.register_agent(agent, metadata)

        # Submit task
        task = UnifiedTask(
            id=uuid4(),
            description="Test task",
            task_type=TaskType.ANALYSIS,
            priority=TaskPriority.HIGH,
            requirements={"capabilities": [AgentCapability.DATA_ANALYSIS]},
            dependencies=[],
            metadata={}
        )

        task_id = await orchestration_engine.submit_task(task)
        assert task_id is not None

        # Wait for execution
        await asyncio.sleep(0.2)

        # Check status
        status = await orchestration_engine.get_task_status(task_id)
        assert status is not None
        assert status.get("status") == "completed"

    @pytest.mark.asyncio
    async def test_task_statistics(self, orchestration_engine):
        """Test task statistics"""
        # Register agent and submit tasks
        agent = MockAgent(
            "test-agent-001",
            "Test Agent",
            [AgentCapability.DATA_ANALYSIS]
        )

        metadata = AgentMetadata(
            agent_id="test-agent-001",
            name="Test Agent",
            capabilities=[AgentCapability.DATA_ANALYSIS],
            status=AgentStatus.IDLE,
            version="1.0.0",
            description="Test agent",
            tags=["test"],
            created_at=datetime.utcnow()
        )

        await orchestration_engine.register_agent(agent, metadata)

        # Submit multiple tasks
        for i in range(3):
            task = UnifiedTask(
                id=uuid4(),
                description=f"Test task {i}",
                task_type=TaskType.ANALYSIS,
                priority=TaskPriority.MEDIUM,
                requirements={"capabilities": [AgentCapability.DATA_ANALYSIS]},
                dependencies=[],
                metadata={}
            )
            await orchestration_engine.submit_task(task)

        # Wait for execution
        await asyncio.sleep(0.5)

        # Get statistics
        stats = await orchestration_engine.get_task_statistics()
        assert stats["total"] >= 3
        assert stats["completed"] >= 3

class TestStateManagement:
    """Test state management"""

    @pytest.fixture
    async def state_manager(self):
        """Create state manager fixture"""
        manager = StateManager(storage_backend="memory")
        yield manager
        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_state_storage_and_retrieval(self, state_manager):
        """Test state storage and retrieval"""
        key = "test_key"
        value = {"data": "test_value", "timestamp": datetime.utcnow().isoformat()}

        # Store state
        success = await state_manager.set_state(key, value)
        assert success is True

        # Retrieve state
        retrieved_value = await state_manager.get_state(key)
        assert retrieved_value is not None
        assert retrieved_value["data"] == "test_value"

    @pytest.mark.asyncio
    async def test_state_notifications(self, state_manager):
        """Test state change notifications"""
        notifications = []

        async def notification_handler(key: str, value: Any):
            notifications.append((key, value))

        # Subscribe to notifications
        await state_manager.subscribe_to_changes(notification_handler)

        # Make state change
        key = "test_key"
        value = {"data": "new_value"}
        await state_manager.set_state(key, value)

        # Wait for notification
        await asyncio.sleep(0.1)

        assert len(notifications) == 1
        assert notifications[0][0] == key
        assert notifications[0][1]["data"] == "new_value"

class TestCommunication:
    """Test communication protocol"""

    @pytest.fixture
    async def communication(self):
        """Create communication protocol fixture"""
        comm = CommunicationProtocol()
        await comm.initialize()
        yield comm
        await comm.shutdown()

    @pytest.mark.asyncio
    async def test_message_sending(self, communication):
        """Test message sending"""
        message = AgentMessage(
            id=uuid4(),
            from_agent="agent-001",
            to_agent="agent-002",
            type=MessageType.COLLABORATION,
            content="Test message",
            timestamp=datetime.utcnow(),
            metadata={}
        )

        success = await communication.send_message(message)
        assert success is True

    @pytest.mark.asyncio
    async def test_message_broadcasting(self, communication):
        """Test message broadcasting"""
        message = AgentMessage(
            id=uuid4(),
            from_agent="agent-001",
            to_agent="all",
            type=MessageType.SYSTEM,
            content="System broadcast",
            timestamp=datetime.utcnow(),
            metadata={}
        )

        success = await communication.broadcast_message(message)
        assert success is True

class TestResourceManagement:
    """Test resource management"""

    @pytest.fixture
    async def resource_manager(self):
        """Create resource manager fixture"""
        manager = ResourceManager()
        await manager.start_monitoring()
        yield manager
        await manager.stop_monitoring()

    @pytest.mark.asyncio
    async def test_resource_allocation(self, resource_manager):
        """Test resource allocation"""
        requirements = {
            ResourceType.CPU: 1.0,
            ResourceType.MEMORY: 2.0
        }

        allocation = await resource_manager.allocate_resources(requirements)
        assert allocation is not None
        assert allocation.id is not None
        assert allocation.resources[ResourceType.CPU] == 1.0
        assert allocation.resources[ResourceType.MEMORY] == 2.0

    @pytest.mark.asyncio
    async def test_resource_release(self, resource_manager):
        """Test resource release"""
        requirements = {
            ResourceType.CPU: 1.0,
            ResourceType.MEMORY: 2.0
        }

        allocation = await resource_manager.allocate_resources(requirements)
        assert allocation is not None

        success = await resource_manager.release_resources(allocation.id)
        assert success is True

    @pytest.mark.asyncio
    async def test_resource_utilization(self, resource_manager):
        """Test resource utilization monitoring"""
        utilization = await resource_manager.get_utilization()

        assert ResourceType.CPU in utilization
        assert ResourceType.MEMORY in utilization
        assert 0.0 <= utilization[ResourceType.CPU] <= 1.0
        assert 0.0 <= utilization[ResourceType.MEMORY] <= 1.0

class TestAgentRegistry:
    """Test agent registry"""

    @pytest.fixture
    async def registry(self):
        """Create agent registry fixture"""
        reg = AgentRegistry()
        yield reg
        await reg.cleanup()

    @pytest.mark.asyncio
    async def test_agent_registration(self, registry):
        """Test agent registration"""
        metadata = AgentMetadata(
            agent_id="test-agent-001",
            name="Test Agent",
            capabilities=[AgentCapability.DATA_ANALYSIS],
            status=AgentStatus.IDLE,
            version="1.0.0",
            description="Test agent",
            tags=["test"],
            created_at=datetime.utcnow()
        )

        success = await registry.register_agent(metadata)
        assert success is True

        # Verify registration
        agents = await registry.get_agents()
        assert len(agents) == 1
        assert agents[0].agent_id == "test-agent-001"

    @pytest.mark.asyncio
    async def test_agent_discovery(self, registry):
        """Test agent discovery by capabilities"""
        # Register multiple agents
        agents_data = [
            ("agent-001", [AgentCapability.DATA_ANALYSIS]),
            ("agent-002", [AgentCapability.MACHINE_LEARNING]),
            ("agent-003", [AgentCapability.DATA_ANALYSIS, AgentCapability.MACHINE_LEARNING])
        ]

        for agent_id, capabilities in agents_data:
            metadata = AgentMetadata(
                agent_id=agent_id,
                name=f"Agent {agent_id}",
                capabilities=capabilities,
                status=AgentStatus.IDLE,
                version="1.0.0",
                description=f"Test agent {agent_id}",
                tags=["test"],
                created_at=datetime.utcnow()
            )
            await registry.register_agent(metadata)

        # Search by capability
        analysis_agents = await registry.get_agents([AgentCapability.DATA_ANALYSIS])
        assert len(analysis_agents) == 2  # agent-001 and agent-003

        ml_agents = await registry.get_agents([AgentCapability.MACHINE_LEARNING])
        assert len(ml_agents) == 2  # agent-002 and agent-003

class TestTaskDistribution:
    """Test task distribution"""

    @pytest.fixture
    async def task_distributor(self):
        """Create task distributor fixture"""
        distributor = TaskDistributor()
        yield distributor
        await distributor.cleanup()

    @pytest.mark.asyncio
    async def test_task_distribution_strategies(self, task_distributor):
        """Test different task distribution strategies"""
        # Create mock agents
        agents = [
            MockAgent("agent-001", "Agent 1", [AgentCapability.DATA_ANALYSIS]),
            MockAgent("agent-002", "Agent 2", [AgentCapability.DATA_ANALYSIS]),
            MockAgent("agent-003", "Agent 3", [AgentCapability.DATA_ANALYSIS])
        ]

        # Test round-robin distribution
        task = UnifiedTask(
            id=uuid4(),
            description="Test task",
            task_type=TaskType.ANALYSIS,
            priority=TaskPriority.MEDIUM,
            requirements={"capabilities": [AgentCapability.DATA_ANALYSIS]},
            dependencies=[],
            metadata={}
        )

        selected_agent = await task_distributor.select_agent(
            task, agents, TaskDistributionStrategy.ROUND_ROBIN
        )
        assert selected_agent is not None
        assert selected_agent in agents

class TestSharedMemory:
    """Test shared memory system"""

    @pytest.fixture
    async def shared_memory(self):
        """Create shared memory system fixture"""
        memory = SharedMemorySystem()
        yield memory
        await memory.cleanup()

    @pytest.mark.asyncio
    async def test_memory_storage_and_retrieval(self, shared_memory):
        """Test memory storage and retrieval"""
        entry = MemoryEntry(
            id=uuid4(),
            type=MemoryType.EXPERIENCE,
            content="Test memory content",
            source_agent="agent-001",
            tags=["test", "memory"],
            created_at=datetime.utcnow(),
            metadata={"test": True}
        )

        success = await shared_memory.store_memory(entry)
        assert success is True

        # Search memory
        results = await shared_memory.search_memory("test memory")
        assert len(results) == 1
        assert results[0].content == "Test memory content"

    @pytest.mark.asyncio
    async def test_memory_search_by_type(self, shared_memory):
        """Test memory search by type"""
        # Store different types of memory
        experience_entry = MemoryEntry(
            id=uuid4(),
            type=MemoryType.EXPERIENCE,
            content="Experience memory",
            source_agent="agent-001",
            tags=["experience"],
            created_at=datetime.utcnow()
        )

        knowledge_entry = MemoryEntry(
            id=uuid4(),
            type=MemoryType.KNOWLEDGE,
            content="Knowledge memory",
            source_agent="agent-001",
            tags=["knowledge"],
            created_at=datetime.utcnow()
        )

        await shared_memory.store_memory(experience_entry)
        await shared_memory.store_memory(knowledge_entry)

        # Search by type
        experience_results = await shared_memory.search_memory("", MemoryType.EXPERIENCE)
        assert len(experience_results) == 1
        assert experience_results[0].type == MemoryType.EXPERIENCE

        knowledge_results = await shared_memory.search_memory("", MemoryType.KNOWLEDGE)
        assert len(knowledge_results) == 1
        assert knowledge_results[0].type == MemoryType.KNOWLEDGE

class TestConflictResolution:
    """Test conflict resolution"""

    @pytest.fixture
    async def conflict_resolver(self):
        """Create conflict resolver fixture"""
        resolver = ConflictResolver()
        yield resolver
        await resolver.cleanup()

    @pytest.mark.asyncio
    async def test_conflict_resolution(self, conflict_resolver):
        """Test conflict resolution"""
        conflict = Conflict(
            id=uuid4(),
            conflict_type=ConflictType.RESOURCE_COMPETITION,
            agents_involved=["agent-001", "agent-002"],
            description="Both agents need GPU resources",
            severity=0.8,
            created_at=datetime.utcnow(),
            metadata={"resource": "gpu"}
        )

        resolution = await conflict_resolver.resolve_conflict(conflict)
        assert resolution is not None
        assert resolution["resolved"] is True

class TestPerformanceTracking:
    """Test performance tracking"""

    @pytest.fixture
    async def performance_tracker(self):
        """Create performance tracker fixture"""
        tracker = PerformanceTracker()
        await tracker.initialize()
        yield tracker
        await tracker.shutdown()

    @pytest.mark.asyncio
    async def test_agent_performance_tracking(self, performance_tracker):
        """Test agent performance tracking"""
        agent_id = "test-agent-001"

        # Register agent
        await performance_tracker.register_agent(agent_id)

        # Record performance metrics
        await performance_tracker.record_task_execution(
            agent_id, "task-001", True, 1.5
        )
        await performance_tracker.record_task_execution(
            agent_id, "task-002", True, 2.0
        )
        await performance_tracker.record_task_execution(
            agent_id, "task-003", False, 0.5
        )

        # Get metrics
        metrics = await performance_tracker.get_agent_metrics(agent_id)
        assert metrics is not None
        assert metrics["total_tasks"] == 3
        assert metrics["successful_tasks"] == 2
        assert metrics["success_rate"] == 2/3
        assert metrics["avg_execution_time"] == (1.5 + 2.0 + 0.5) / 3

class TestCollaborationDashboard:
    """Test collaboration dashboard"""

    @pytest.fixture
    async def dashboard(self):
        """Create collaboration dashboard fixture"""
        dash = CollaborationDashboard()
        await dash.initialize()
        yield dash
        await dash.shutdown()

    @pytest.mark.asyncio
    async def test_dashboard_metrics(self, dashboard):
        """Test dashboard metrics"""
        # Update metrics
        metrics = {
            "total_agents": 5,
            "active_agents": 3,
            "total_tasks": 10,
            "completed_tasks": 8,
            "collaboration_count": 15
        }

        await dashboard.update_metrics(metrics)

        # Get dashboard data
        data = await dashboard.get_dashboard_data()
        assert data is not None
        assert "metrics" in data
        assert data["metrics"]["total_agents"] == 5

class TestAgentMarketplace:
    """Test agent marketplace"""

    @pytest.fixture
    async def marketplace(self):
        """Create agent marketplace fixture"""
        market = AgentMarketplace()
        yield market
        await market.cleanup()

    @pytest.mark.asyncio
    async def test_listing_creation(self, marketplace):
        """Test agent listing creation"""
        listing = AgentListing(
            agent_id=uuid4(),
            name="Test Agent",
            description="Test agent for marketplace",
            version="1.0.0",
            author="Test Author",
            capabilities=[AgentCapability.DATA_ANALYSIS],
            tags=["test", "analysis"],
            status=ListingStatus.ACTIVE,
            pricing_model="free"
        )

        created_listing = await marketplace.create_listing(listing)
        assert created_listing is not None
        assert created_listing.name == "Test Agent"
        assert created_listing.status == ListingStatus.PENDING_REVIEW

    @pytest.mark.asyncio
    async def test_listing_search(self, marketplace):
        """Test listing search"""
        # Create multiple listings
        listings_data = [
            ("Analysis Agent", [AgentCapability.DATA_ANALYSIS], ["analysis"]),
            ("ML Agent", [AgentCapability.MACHINE_LEARNING], ["ml"]),
            ("Processing Agent", [AgentCapability.DATA_PROCESSING], ["processing"])
        ]

        for name, capabilities, tags in listings_data:
            listing = AgentListing(
                agent_id=uuid4(),
                name=name,
                description=f"Test {name}",
                version="1.0.0",
                author="Test Author",
                capabilities=capabilities,
                tags=tags,
                status=ListingStatus.ACTIVE,
                pricing_model="free"
            )
            await marketplace.create_listing(listing)

        # Search listings
        analysis_listings = await marketplace.search_listings(
            capabilities=[AgentCapability.DATA_ANALYSIS]
        )
        assert len(analysis_listings) == 1
        assert analysis_listings[0].name == "Analysis Agent"

    @pytest.mark.asyncio
    async def test_rating_system(self, marketplace):
        """Test rating system"""
        # Create listing
        listing = AgentListing(
            agent_id=uuid4(),
            name="Test Agent",
            description="Test agent",
            version="1.0.0",
            author="Test Author",
            capabilities=[AgentCapability.DATA_ANALYSIS],
            tags=["test"],
            status=ListingStatus.ACTIVE,
            pricing_model="free"
        )

        created_listing = await marketplace.create_listing(listing)

        # Add rating
        rating = AgentRating(
            agent_id=created_listing.agent_id,
            reviewer_id=uuid4(),
            category=RatingCategory.PERFORMANCE,
            score=4.5,
            review="Great performance!"
        )

        await marketplace.add_rating(rating)

        # Check rating
        ratings = await marketplace.get_ratings(created_listing.agent_id)
        assert len(ratings) == 1
        assert ratings[0].score == 4.5

class TestMultiAgentPlatform:
    """Test multi-agent platform integration"""

    @pytest.fixture
    async def platform(self):
        """Create multi-agent platform fixture"""
        config = PlatformConfig(
            max_concurrent_tasks=10,
            task_timeout=60,
            heartbeat_interval=5,
            cleanup_interval=30,
            enable_marketplace=True,
            enable_dashboard=True,
            enable_performance_tracking=True,
            enable_conflict_resolution=True,
            storage_backend="memory"
        )

        platform = MultiAgentPlatform(config)
        await platform.start()
        yield platform
        await platform.stop()

    @pytest.mark.asyncio
    async def test_platform_lifecycle(self, platform):
        """Test platform lifecycle"""
        assert platform.is_running() is True

        # Get platform stats
        stats = await platform.get_platform_stats()
        assert stats is not None
        assert stats.total_agents == 0
        assert stats.total_tasks == 0

    @pytest.mark.asyncio
    async def test_agent_registration_and_task_execution(self, platform):
        """Test agent registration and task execution"""
        # Create and register agent
        agent = MockAgent(
            "test-agent-001",
            "Test Agent",
            [AgentCapability.DATA_ANALYSIS]
        )

        metadata = AgentMetadata(
            agent_id="test-agent-001",
            name="Test Agent",
            capabilities=[AgentCapability.DATA_ANALYSIS],
            status=AgentStatus.IDLE,
            version="1.0.0",
            description="Test agent",
            tags=["test"],
            created_at=datetime.utcnow()
        )

        success = await platform.register_agent(agent, metadata)
        assert success is True

        # Submit task
        task = UnifiedTask(
            id=uuid4(),
            description="Test task",
            task_type=TaskType.ANALYSIS,
            priority=TaskPriority.HIGH,
            requirements={"capabilities": [AgentCapability.DATA_ANALYSIS]},
            dependencies=[],
            metadata={}
        )

        task_id = await platform.submit_task(task)
        assert task_id is not None

        # Wait for execution
        await asyncio.sleep(0.3)

        # Check task status
        status = await platform.get_task_status(task_id)
        assert status is not None
        assert status.get("status") == "completed"

    @pytest.mark.asyncio
    async def test_platform_health_check(self, platform):
        """Test platform health check"""
        health = await platform.health_check()

        assert health is not None
        assert "platform_status" in health
        assert "components" in health
        assert health["platform_status"] == "healthy"

        # Check component health
        components = health["components"]
        assert "orchestration" in components
        assert "state_manager" in components
        assert "communication" in components
        assert "resource_manager" in components
        assert "registry" in components

    @pytest.mark.asyncio
    async def test_platform_communication(self, platform):
        """Test platform communication"""
        # Send message
        message = AgentMessage(
            id=uuid4(),
            from_agent="agent-001",
            to_agent="agent-002",
            type=MessageType.COLLABORATION,
            content="Test message",
            timestamp=datetime.utcnow(),
            metadata={}
        )

        success = await platform.send_message(message)
        assert success is True

        # Broadcast message
        broadcast_message = AgentMessage(
            id=uuid4(),
            from_agent="platform",
            to_agent="all",
            type=MessageType.SYSTEM,
            content="System broadcast",
            timestamp=datetime.utcnow(),
            metadata={}
        )

        success = await platform.broadcast_message(broadcast_message)
        assert success is True

    @pytest.mark.asyncio
    async def test_platform_memory_sharing(self, platform):
        """Test platform memory sharing"""
        # Share memory
        memory_entry = MemoryEntry(
            id=uuid4(),
            type=MemoryType.EXPERIENCE,
            content="Test memory content",
            source_agent="agent-001",
            tags=["test", "memory"],
            created_at=datetime.utcnow(),
            metadata={"test": True}
        )

        success = await platform.share_memory(memory_entry)
        assert success is True

        # Search memory
        results = await platform.search_memory("test memory")
        assert len(results) == 1
        assert results[0].content == "Test memory content"

class TestIntegration:
    """Integration tests"""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow from task submission to completion"""
        # Create platform
        config = PlatformConfig(
            max_concurrent_tasks=5,
            task_timeout=30,
            heartbeat_interval=5,
            cleanup_interval=30,
            enable_marketplace=False,
            enable_dashboard=False,
            enable_performance_tracking=True,
            enable_conflict_resolution=False,
            storage_backend="memory"
        )

        platform = MultiAgentPlatform(config)

        async with platform.platform_context():
            # Register multiple agents
            agents_data = [
                ("analysis-agent", [AgentCapability.DATA_ANALYSIS]),
                ("processing-agent", [AgentCapability.DATA_PROCESSING]),
                ("ml-agent", [AgentCapability.MACHINE_LEARNING])
            ]

            for name, capabilities in agents_data:
                agent = MockAgent(f"{name}-001", name, capabilities)
                metadata = AgentMetadata(
                    agent_id=f"{name}-001",
                    name=name,
                    capabilities=capabilities,
                    status=AgentStatus.IDLE,
                    version="1.0.0",
                    description=f"Test {name}",
                    tags=["test"],
                    created_at=datetime.utcnow()
                )
                await platform.register_agent(agent, metadata)

            # Submit multiple tasks
            task_descriptions = [
                "Analyze customer data",
                "Process sensor data",
                "Train ML model"
            ]

            task_ids = []
            for description in task_descriptions:
                task = UnifiedTask(
                    id=uuid4(),
                    description=description,
                    task_type=TaskType.ANALYSIS,
                    priority=TaskPriority.MEDIUM,
                    requirements={"capabilities": [AgentCapability.DATA_ANALYSIS]},
                    dependencies=[],
                    metadata={}
                )
                task_id = await platform.submit_task(task)
                task_ids.append(task_id)

            # Wait for all tasks to complete
            await asyncio.sleep(1.0)

            # Verify all tasks completed
            for task_id in task_ids:
                status = await platform.get_task_status(task_id)
                assert status is not None
                assert status.get("status") == "completed"

            # Check platform statistics
            stats = await platform.get_platform_stats()
            assert stats.total_agents == 3
            assert stats.total_tasks == 3
            assert stats.completed_tasks == 3

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])