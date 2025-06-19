"""
Unit tests for concrete agent implementations
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.infrastructure.agents.concrete_agents import (
    FSMReactAgentImpl, NextGenAgentImpl, CrewAgentImpl, SpecializedAgentImpl,
    AgentConfig
)
from src.unified_architecture.core import (
    UnifiedTask, TaskStatus, AgentStatus, AgentCapability
)


class TestFSMReactAgentImpl:
    """Test FSM React Agent implementation"""
    
    @pytest.fixture
    def agent(self):
        """Create FSM React Agent instance"""
        return FSMReactAgentImpl()
    
    @pytest.fixture
    def task(self):
        """Create test task"""
        return UnifiedTask(
            task_id="test-task-1",
            title="Test Task",
            description="This is a test task",
            priority=1,
            required_capabilities=[],
            deadline=None
        )
    
    @pytest.mark.asyncio
    async def test_initialization(self, agent):
        """Test agent initialization"""
        with patch.object(agent._fsm_agent, 'initialize', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = True
            
            result = await agent.initialize()
            
            assert result is True
            assert agent.status == AgentStatus.IDLE
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialization_failure(self, agent):
        """Test agent initialization failure"""
        with patch.object(agent._fsm_agent, 'initialize', new_callable=AsyncMock) as mock_init:
            mock_init.side_effect = Exception("Initialization failed")
            
            result = await agent.initialize()
            
            assert result is False
            assert agent.status == AgentStatus.ERROR
    
    @pytest.mark.asyncio
    async def test_execute_task_success(self, agent, task):
        """Test successful task execution"""
        # Mock FSM agent run method
        mock_result = {
            "response": "Task completed successfully",
            "tools_used": ["tool1", "tool2"],
            "confidence": 0.9
        }
        
        with patch.object(agent._fsm_agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            
            result = await agent.execute_task(task)
            
            assert result.task_id == task.task_id
            assert result.agent_id == agent.agent_id
            assert result.status == TaskStatus.COMPLETED
            assert result.result == mock_result
            assert result.execution_time > 0
            assert result.metadata["agent_type"] == "FSM_REACT"
    
    @pytest.mark.asyncio
    async def test_execute_task_failure(self, agent, task):
        """Test task execution failure"""
        with patch.object(agent._fsm_agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = Exception("Task execution failed")
            
            result = await agent.execute_task(task)
            
            assert result.task_id == task.task_id
            assert result.agent_id == agent.agent_id
            assert result.status == TaskStatus.FAILED
            assert "Task execution failed" in result.error
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, agent):
        """Test getting agent capabilities"""
        capabilities = await agent.get_capabilities()
        
        expected_capabilities = [
            AgentCapability.REASONING,
            AgentCapability.TASK_EXECUTION,
            AgentCapability.DECISION_MAKING
        ]
        
        assert capabilities == expected_capabilities
    
    @pytest.mark.asyncio
    async def test_get_status(self, agent):
        """Test getting agent status"""
        status = await agent.get_status()
        assert status == AgentStatus.IDLE
    
    @pytest.mark.asyncio
    async def test_health_check(self, agent):
        """Test health check"""
        health = await agent.health_check()
        
        assert "status" in health
        assert "agent_id" in health
        assert "current_task" in health
        assert "task_history_size" in health
        assert "performance_metrics" in health
    
    @pytest.mark.asyncio
    async def test_collaboration(self, agent, task):
        """Test agent collaboration"""
        other_agent = Mock()
        other_agent.get_metadata.return_value = Mock(name="Test Agent")
        
        with patch.object(agent, 'execute_task', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = Mock()
            
            await agent.collaborate(other_agent, task)
            
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args[0][0]
            assert "Collaborative:" in call_args.title


class TestNextGenAgentImpl:
    """Test Next Gen Agent implementation"""
    
    @pytest.fixture
    def agent(self):
        """Create Next Gen Agent instance"""
        return NextGenAgentImpl()
    
    @pytest.fixture
    def task(self):
        """Create test task"""
        return UnifiedTask(
            task_id="test-task-2",
            title="Learning Task",
            description="This is a learning task",
            priority=2,
            required_capabilities=[],
            deadline=None
        )
    
    @pytest.mark.asyncio
    async def test_initialization(self, agent):
        """Test agent initialization"""
        with patch.object(agent._enhanced_agent, 'initialize', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = True
            
            result = await agent.initialize()
            
            assert result is True
            assert agent.status == AgentStatus.IDLE
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_task_with_learning(self, agent, task):
        """Test task execution with learning"""
        mock_result = {
            "response": "Learned from task",
            "tools_used": ["learning_tool"],
            "confidence": 0.95
        }
        
        with patch.object(agent._enhanced_agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            
            result = await agent.execute_task(task)
            
            assert result.status == TaskStatus.COMPLETED
            assert result.metadata["agent_type"] == "NEXT_GEN"
            assert result.metadata["learning_applied"] is True
            
            # Check that learning data was stored
            learning_key = f"task_{task.task_id}"
            assert learning_key in agent._learning_data
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, agent):
        """Test getting agent capabilities"""
        capabilities = await agent.get_capabilities()
        
        expected_capabilities = [
            AgentCapability.REASONING,
            AgentCapability.LEARNING,
            AgentCapability.ADAPTATION,
            AgentCapability.COLLABORATION
        ]
        
        assert capabilities == expected_capabilities


class TestCrewAgentImpl:
    """Test Crew Agent implementation"""
    
    @pytest.fixture
    def agent(self):
        """Create Crew Agent instance"""
        return CrewAgentImpl()
    
    @pytest.fixture
    def task(self):
        """Create test task"""
        return UnifiedTask(
            task_id="test-task-3",
            title="Team Task",
            description="This is a team collaboration task",
            priority=3,
            required_capabilities=[],
            deadline=None
        )
    
    @pytest.mark.asyncio
    async def test_initialization(self, agent):
        """Test agent initialization"""
        with patch.object(agent._crew_agent, 'initialize', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = True
            
            result = await agent.initialize()
            
            assert result is True
            assert agent.status == AgentStatus.IDLE
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_task_with_team(self, agent, task):
        """Test task execution with team collaboration"""
        mock_result = {
            "response": "Team completed task",
            "tools_used": ["team_tool"],
            "confidence": 0.88
        }
        
        with patch.object(agent._crew_agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            
            result = await agent.execute_task(task)
            
            assert result.status == TaskStatus.COMPLETED
            assert result.metadata["agent_type"] == "CREW"
            assert result.metadata["collaboration_level"] == "high"
            assert result.metadata["team_size"] == 0  # No team members yet
    
    @pytest.mark.asyncio
    async def test_collaboration_adds_team_member(self, agent, task):
        """Test that collaboration adds team members"""
        other_agent = Mock()
        other_agent.get_metadata.return_value = Mock(name="Team Member")
        other_agent.get_capabilities.return_value = [AgentCapability.REASONING]
        
        with patch.object(agent, 'execute_task', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = Mock()
            
            await agent.collaborate(other_agent, task)
            
            assert len(agent._team_members) == 1
            assert agent._team_members[0]["name"] == "Team Member"
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, agent):
        """Test getting agent capabilities"""
        capabilities = await agent.get_capabilities()
        
        expected_capabilities = [
            AgentCapability.COLLABORATION,
            AgentCapability.TASK_ORCHESTRATION,
            AgentCapability.TEAM_MANAGEMENT,
            AgentCapability.COORDINATION
        ]
        
        assert capabilities == expected_capabilities


class TestSpecializedAgentImpl:
    """Test Specialized Agent implementation"""
    
    @pytest.fixture
    def agent(self):
        """Create Specialized Agent instance"""
        return SpecializedAgentImpl("data_analysis")
    
    @pytest.fixture
    def task(self):
        """Create test task"""
        return UnifiedTask(
            task_id="test-task-4",
            title="Analysis Task",
            description="This is a data analysis task",
            priority=4,
            required_capabilities=[],
            deadline=None
        )
    
    @pytest.mark.asyncio
    async def test_initialization(self, agent):
        """Test agent initialization"""
        with patch.object(agent._hybrid_agent, 'initialize', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = True
            
            result = await agent.initialize()
            
            assert result is True
            assert agent.status == AgentStatus.IDLE
            assert agent.domain == "data_analysis"
            assert len(agent._domain_knowledge) > 0
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_task_with_domain_knowledge(self, agent, task):
        """Test task execution with domain knowledge"""
        mock_result = {
            "response": "Analysis completed",
            "tools_used": ["data_analyzer"],
            "confidence": 0.92
        }
        
        with patch.object(agent._hybrid_agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = mock_result
            
            result = await agent.execute_task(task)
            
            assert result.status == TaskStatus.COMPLETED
            assert result.metadata["agent_type"] == "SPECIALIZED"
            assert result.metadata["domain"] == "data_analysis"
            assert result.metadata["domain_knowledge_applied"] is True
            
            # Check that domain knowledge was applied
            call_args = mock_run.call_args[0][0]
            assert "[DATA_ANALYSIS]" in call_args
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, agent):
        """Test getting agent capabilities"""
        capabilities = await agent.get_capabilities()
        
        expected_capabilities = [
            AgentCapability.DOMAIN_EXPERTISE,
            AgentCapability.SPECIALIZED_REASONING,
            AgentCapability.OPTIMIZATION
        ]
        
        assert capabilities == expected_capabilities
    
    @pytest.mark.asyncio
    async def test_domain_knowledge_loading(self, agent):
        """Test domain knowledge loading"""
        await agent._load_domain_knowledge()
        
        assert agent._domain_knowledge["domain"] == "data_analysis"
        assert "expertise_areas" in agent._domain_knowledge
        assert "specialized_tools" in agent._domain_knowledge
        assert "loaded_at" in agent._domain_knowledge


class TestAgentConfig:
    """Test Agent Configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = AgentConfig()
        
        assert config.max_concurrent_tasks == 5
        assert config.task_timeout == 300
        assert config.heartbeat_interval == 30
        assert config.enable_learning is True
        assert config.enable_collaboration is True
        assert config.memory_size == 1000
        assert config.log_level == "INFO"
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = AgentConfig(
            max_concurrent_tasks=10,
            task_timeout=600,
            enable_learning=False,
            log_level="DEBUG"
        )
        
        assert config.max_concurrent_tasks == 10
        assert config.task_timeout == 600
        assert config.enable_learning is False
        assert config.log_level == "DEBUG"
        assert config.enable_collaboration is True  # Default value


class TestAgentIntegration:
    """Integration tests for agents"""
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle(self):
        """Test complete agent lifecycle"""
        # Create agent
        agent = FSMReactAgentImpl()
        
        # Initialize
        success = await agent.initialize()
        assert success is True
        assert agent.status == AgentStatus.IDLE
        
        # Create task
        task = UnifiedTask(
            task_id="lifecycle-test",
            title="Lifecycle Test",
            description="Testing complete lifecycle",
            priority=1,
            required_capabilities=[],
            deadline=None
        )
        
        # Execute task
        with patch.object(agent._fsm_agent, 'run', new_callable=AsyncMock) as mock_run:
            mock_run.return_value = {"response": "Lifecycle completed"}
            
            result = await agent.execute_task(task)
            
            assert result.status == TaskStatus.COMPLETED
            assert agent.status == AgentStatus.IDLE
        
        # Health check
        health = await agent.health_check()
        assert health["status"] == "idle"
        assert health["task_history_size"] == 1
    
    @pytest.mark.asyncio
    async def test_agent_collaboration_workflow(self):
        """Test agent collaboration workflow"""
        # Create two agents
        agent1 = FSMReactAgentImpl()
        agent2 = NextGenAgentImpl()
        
        await agent1.initialize()
        await agent2.initialize()
        
        # Create collaborative task
        task = UnifiedTask(
            task_id="collab-test",
            title="Collaboration Test",
            description="Testing agent collaboration",
            priority=1,
            required_capabilities=[],
            deadline=None
        )
        
        # Test collaboration
        with patch.object(agent1, 'execute_task', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = Mock()
            
            await agent1.collaborate(agent2, task)
            
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args[0][0]
            assert "Collaborative:" in call_args.title
            assert agent2.get_metadata().name in call_args.description 