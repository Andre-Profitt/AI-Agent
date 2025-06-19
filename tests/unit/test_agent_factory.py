"""
Unit tests for agent factory
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.infrastructure.agents.agent_factory import (
    AgentFactory, AgentType, get_agent_factory, create_default_agents
)
from src.infrastructure.agents.concrete_agents import (
    FSMReactAgentImpl, NextGenAgentImpl, CrewAgentImpl, SpecializedAgentImpl
)
from src.unified_architecture.core import IUnifiedAgent, AgentCapability


class TestAgentFactory:
    """Test Agent Factory"""
    
    @pytest.fixture
    def factory(self):
        """Create agent factory instance"""
        return AgentFactory()
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent"""
        agent = Mock(spec=IUnifiedAgent)
        agent.agent_id = "test-agent-id"
        agent.initialize = AsyncMock(return_value=True)
        agent.get_capabilities = AsyncMock(return_value=[AgentCapability.REASONING])
        agent.get_metadata = AsyncMock(return_value=Mock(name="Test Agent"))
        agent.status = Mock(value="idle")
        return agent
    
    def test_initialization(self, factory):
        """Test factory initialization"""
        assert len(factory._agent_registry) == 4
        assert AgentType.FSM_REACT in factory._agent_registry
        assert AgentType.NEXT_GEN in factory._agent_registry
        assert AgentType.CREW in factory._agent_registry
        assert AgentType.SPECIALIZED in factory._agent_registry
    
    def test_get_available_agent_types(self, factory):
        """Test getting available agent types"""
        types = factory.get_available_agent_types()
        
        expected_types = [
            AgentType.FSM_REACT,
            AgentType.NEXT_GEN,
            AgentType.CREW,
            AgentType.SPECIALIZED
        ]
        
        assert set(types) == set(expected_types)
    
    def test_register_agent_type(self, factory):
        """Test registering new agent type"""
        custom_agent_class = Mock()
        factory.register_agent_type("custom", custom_agent_class)
        
        assert "custom" in factory._agent_registry
        assert factory._agent_registry["custom"] == custom_agent_class
    
    @pytest.mark.asyncio
    async def test_create_fsm_agent(self, factory):
        """Test creating FSM React agent"""
        with patch('src.infrastructure.agents.concrete_agents.FSMReactAgentImpl') as mock_class:
            mock_agent = Mock()
            mock_agent.agent_id = "fsm-agent-id"
            mock_agent.initialize = AsyncMock(return_value=True)
            mock_class.return_value = mock_agent
            
            agent = await factory.create_agent(AgentType.FSM_REACT)
            
            assert agent == mock_agent
            assert "fsm-agent-id" in factory._created_agents
            mock_agent.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_next_gen_agent(self, factory):
        """Test creating Next Gen agent"""
        with patch('src.infrastructure.agents.concrete_agents.NextGenAgentImpl') as mock_class:
            mock_agent = Mock()
            mock_agent.agent_id = "next-gen-agent-id"
            mock_agent.initialize = AsyncMock(return_value=True)
            mock_class.return_value = mock_agent
            
            agent = await factory.create_agent(AgentType.NEXT_GEN)
            
            assert agent == mock_agent
            assert "next-gen-agent-id" in factory._created_agents
            mock_agent.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_crew_agent(self, factory):
        """Test creating Crew agent"""
        with patch('src.infrastructure.agents.concrete_agents.CrewAgentImpl') as mock_class:
            mock_agent = Mock()
            mock_agent.agent_id = "crew-agent-id"
            mock_agent.initialize = AsyncMock(return_value=True)
            mock_class.return_value = mock_agent
            
            agent = await factory.create_agent(AgentType.CREW)
            
            assert agent == mock_agent
            assert "crew-agent-id" in factory._created_agents
            mock_agent.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_specialized_agent(self, factory):
        """Test creating Specialized agent"""
        with patch('src.infrastructure.agents.concrete_agents.SpecializedAgentImpl') as mock_class:
            mock_agent = Mock()
            mock_agent.agent_id = "specialized-agent-id"
            mock_agent.initialize = AsyncMock(return_value=True)
            mock_class.return_value = mock_agent
            
            agent = await factory.create_agent(AgentType.SPECIALIZED, domain="data_analysis")
            
            assert agent == mock_agent
            assert "specialized-agent-id" in factory._created_agents
            mock_class.assert_called_once_with(domain="data_analysis", config=None)
            mock_agent.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_agent_with_config(self, factory):
        """Test creating agent with configuration"""
        from src.infrastructure.agents.concrete_agents import AgentConfig
        
        config = AgentConfig(max_concurrent_tasks=10)
        
        with patch('src.infrastructure.agents.concrete_agents.FSMReactAgentImpl') as mock_class:
            mock_agent = Mock()
            mock_agent.agent_id = "config-agent-id"
            mock_agent.initialize = AsyncMock(return_value=True)
            mock_class.return_value = mock_agent
            
            agent = await factory.create_agent(AgentType.FSM_REACT, config=config)
            
            mock_class.assert_called_once_with(config=config)
            assert agent == mock_agent
    
    @pytest.mark.asyncio
    async def test_create_unsupported_agent_type(self, factory):
        """Test creating unsupported agent type"""
        with pytest.raises(ValueError, match="Unsupported agent type"):
            await factory.create_agent("unsupported_type")
    
    @pytest.mark.asyncio
    async def test_create_agent_initialization_failure(self, factory):
        """Test agent creation with initialization failure"""
        with patch('src.infrastructure.agents.concrete_agents.FSMReactAgentImpl') as mock_class:
            mock_agent = Mock()
            mock_agent.agent_id = "failed-agent-id"
            mock_agent.initialize = AsyncMock(return_value=False)
            mock_class.return_value = mock_agent
            
            with pytest.raises(RuntimeError, match="Failed to initialize"):
                await factory.create_agent(AgentType.FSM_REACT)
    
    @pytest.mark.asyncio
    async def test_create_agent_for_capabilities(self, factory):
        """Test creating agent for specific capabilities"""
        required_capabilities = [AgentCapability.REASONING, AgentCapability.LEARNING]
        
        with patch.object(factory, 'create_agent', new_callable=AsyncMock) as mock_create:
            # Mock different agents with different capabilities
            agent1 = Mock()
            agent1.get_capabilities = AsyncMock(return_value=[AgentCapability.REASONING])
            
            agent2 = Mock()
            agent2.get_capabilities = AsyncMock(return_value=[AgentCapability.REASONING, AgentCapability.LEARNING])
            
            mock_create.side_effect = [agent1, agent2]
            
            result = await factory.create_agent_for_capabilities(required_capabilities)
            
            # Should return the agent with better capability match
            assert result == agent2
            assert mock_create.call_count == 2
    
    @pytest.mark.asyncio
    async def test_create_agent_team(self, factory):
        """Test creating agent team"""
        team_config = [
            {"type": AgentType.FSM_REACT, "kwargs": {}},
            {"type": AgentType.NEXT_GEN, "kwargs": {}},
            {"type": AgentType.SPECIALIZED, "kwargs": {"domain": "data_analysis"}}
        ]
        
        with patch.object(factory, 'create_agent', new_callable=AsyncMock) as mock_create:
            mock_agents = [Mock(), Mock(), Mock()]
            mock_create.side_effect = mock_agents
            
            agents = await factory.create_agent_team(team_config)
            
            assert len(agents) == 3
            assert agents == mock_agents
            assert mock_create.call_count == 3
    
    def test_get_agent(self, factory):
        """Test getting agent by ID"""
        agent_id = "test-agent-id"
        mock_agent = Mock()
        factory._created_agents[agent_id] = mock_agent
        
        result = factory.get_agent(agent_id)
        assert result == mock_agent
    
    def test_get_agent_not_found(self, factory):
        """Test getting non-existent agent"""
        result = factory.get_agent("non-existent-id")
        assert result is None
    
    def test_get_all_agents(self, factory):
        """Test getting all agents"""
        agent1 = Mock()
        agent2 = Mock()
        factory._created_agents = {"id1": agent1, "id2": agent2}
        
        result = factory.get_all_agents()
        
        assert result == {"id1": agent1, "id2": agent2}
        # Should return a copy, not the original
        assert result is not factory._created_agents
    
    @pytest.mark.asyncio
    async def test_destroy_agent(self, factory):
        """Test destroying agent"""
        agent_id = "test-agent-id"
        mock_agent = Mock()
        mock_agent.shutdown = AsyncMock()
        factory._created_agents[agent_id] = mock_agent
        
        result = await factory.destroy_agent(agent_id)
        
        assert result is True
        assert agent_id not in factory._created_agents
        mock_agent.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_destroy_agent_not_found(self, factory):
        """Test destroying non-existent agent"""
        result = await factory.destroy_agent("non-existent-id")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_destroy_agent_without_shutdown(self, factory):
        """Test destroying agent without shutdown method"""
        agent_id = "test-agent-id"
        mock_agent = Mock()
        # No shutdown method
        factory._created_agents[agent_id] = mock_agent
        
        result = await factory.destroy_agent(agent_id)
        
        assert result is True
        assert agent_id not in factory._created_agents
    
    @pytest.mark.asyncio
    async def test_destroy_all_agents(self, factory):
        """Test destroying all agents"""
        agent1 = Mock()
        agent1.shutdown = AsyncMock()
        agent2 = Mock()
        agent2.shutdown = AsyncMock()
        
        factory._created_agents = {"id1": agent1, "id2": agent2}
        
        await factory.destroy_all_agents()
        
        assert len(factory._created_agents) == 0
        agent1.shutdown.assert_called_once()
        agent2.shutdown.assert_called_once()
    
    def test_get_agent_stats(self, factory):
        """Test getting agent statistics"""
        # Create mock agents with different types and statuses
        agent1 = Mock()
        agent1.status = Mock(value="idle")
        
        agent2 = Mock()
        agent2.status = Mock(value="busy")
        
        agent3 = Mock()
        agent3.status = Mock(value="idle")
        
        factory._created_agents = {
            "id1": agent1,
            "id2": agent2,
            "id3": agent3
        }
        
        stats = factory.get_agent_stats()
        
        assert stats["total_agents"] == 3
        assert "Mock" in stats["agent_types"]
        assert stats["agent_types"]["Mock"] == 3
        assert stats["status_counts"]["idle"] == 2
        assert stats["status_counts"]["busy"] == 1


class TestAgentFactoryGlobal:
    """Test global agent factory functions"""
    
    def test_get_agent_factory_singleton(self):
        """Test that get_agent_factory returns singleton"""
        factory1 = get_agent_factory()
        factory2 = get_agent_factory()
        
        assert factory1 is factory2
    
    @pytest.mark.asyncio
    async def test_create_default_agents(self):
        """Test creating default agents"""
        with patch('src.infrastructure.agents.agent_factory.get_agent_factory') as mock_get_factory:
            mock_factory = Mock()
            mock_factory.create_agent = AsyncMock()
            mock_get_factory.return_value = mock_factory
            
            # Mock created agents
            mock_agents = {
                "fsm_react": Mock(),
                "next_gen": Mock(),
                "crew": Mock(),
                "specialized_data_analysis": Mock(),
                "specialized_code_generation": Mock(),
                "specialized_research": Mock(),
                "specialized_creative": Mock()
            }
            mock_factory.create_agent.side_effect = list(mock_agents.values())
            
            result = await create_default_agents()
            
            assert len(result) == 7
            assert "fsm_react" in result
            assert "next_gen" in result
            assert "crew" in result
            assert "specialized_data_analysis" in result
            assert "specialized_code_generation" in result
            assert "specialized_research" in result
            assert "specialized_creative" in result
            
            # Verify create_agent was called for each agent type
            assert mock_factory.create_agent.call_count == 7


class TestAgentFactoryIntegration:
    """Integration tests for agent factory"""
    
    @pytest.mark.asyncio
    async def test_complete_agent_lifecycle(self):
        """Test complete agent lifecycle with factory"""
        factory = AgentFactory()
        
        # Create agent
        with patch('src.infrastructure.agents.concrete_agents.FSMReactAgentImpl') as mock_class:
            mock_agent = Mock()
            mock_agent.agent_id = "lifecycle-agent-id"
            mock_agent.initialize = AsyncMock(return_value=True)
            mock_agent.shutdown = AsyncMock()
            mock_class.return_value = mock_agent
            
            agent = await factory.create_agent(AgentType.FSM_REACT)
            
            # Verify agent was created and registered
            assert agent == mock_agent
            assert "lifecycle-agent-id" in factory._created_agents
            
            # Get agent
            retrieved_agent = factory.get_agent("lifecycle-agent-id")
            assert retrieved_agent == agent
            
            # Get stats
            stats = factory.get_agent_stats()
            assert stats["total_agents"] == 1
            
            # Destroy agent
            success = await factory.destroy_agent("lifecycle-agent-id")
            assert success is True
            assert "lifecycle-agent-id" not in factory._created_agents
            mock_agent.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_agent_team_workflow(self):
        """Test agent team workflow"""
        factory = AgentFactory()
        
        team_config = [
            {"type": AgentType.FSM_REACT, "kwargs": {}},
            {"type": AgentType.NEXT_GEN, "kwargs": {}},
            {"type": AgentType.CREW, "kwargs": {}}
        ]
        
        with patch.object(factory, 'create_agent', new_callable=AsyncMock) as mock_create:
            mock_agents = [Mock(), Mock(), Mock()]
            mock_create.side_effect = mock_agents
            
            # Create team
            team = await factory.create_agent_team(team_config)
            
            assert len(team) == 3
            assert len(factory._created_agents) == 3
            
            # Get team stats
            stats = factory.get_agent_stats()
            assert stats["total_agents"] == 3
            
            # Destroy all agents
            await factory.destroy_all_agents()
            assert len(factory._created_agents) == 0 