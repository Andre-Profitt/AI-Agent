"""
Tests for Unified Agent
Comprehensive unit and integration tests
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.agents.unified_agent import (
    UnifiedAgent, AgentState, AgentCapability, 
    AgentContext, create_agent
)
from src.core.entities.message import Message
from src.core.entities.tool import Tool, ToolResult
from src.infrastructure.config import AgentConfig
from src.core.exceptions import AgentError


class TestUnifiedAgent:
    """Test unified agent functionality"""
    
    @pytest.fixture
    def agent_config(self):
        """Create test agent configuration"""
        return AgentConfig(
            model_name="gpt-4",
            temperature=0.7,
            max_iterations=5,
            timeout=30.0,
            enable_memory=True,
            enable_monitoring=True,
            error_threshold=3,
            recovery_timeout=1.0
        )
        
    @pytest.fixture
    def mock_tools(self):
        """Create mock tools"""
        tool1 = Mock(spec=Tool)
        tool1.name = "search"
        tool1.description = "Search the web"
        
        tool2 = Mock(spec=Tool)
        tool2.name = "calculate"
        tool2.description = "Perform calculations"
        
        return [tool1, tool2]
        
    @pytest.fixture
    def agent(self, agent_config, mock_tools):
        """Create test agent"""
        return UnifiedAgent(
            agent_id="test-agent",
            name="Test Agent",
            config=agent_config,
            tools=mock_tools,
            capabilities=[
                AgentCapability.REASONING,
                AgentCapability.TOOL_USE,
                AgentCapability.MEMORY
            ]
        )
        
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent is properly initialized"""
        assert agent.agent_id == "test-agent"
        assert agent.name == "Test Agent"
        assert agent.state == AgentState.IDLE
        assert len(agent.tools) == 2
        assert AgentCapability.MEMORY in agent.capabilities
        
    @pytest.mark.asyncio
    async def test_state_transitions(self, agent):
        """Test state transition logic"""
        # Test transition to thinking
        await agent._transition_state(AgentState.THINKING)
        assert agent.state == AgentState.THINKING
        
        # Test transition to executing
        await agent._transition_state(AgentState.EXECUTING)
        assert agent.state == AgentState.EXECUTING
        
        # Test error state recovery
        await agent._transition_state(AgentState.ERROR)
        assert agent.state == AgentState.ERROR
        await asyncio.sleep(1.1)  # Wait for recovery
        
    @pytest.mark.asyncio
    async def test_message_processing(self, agent):
        """Test message processing pipeline"""
        message = Message(content="What is the weather?", role="user")
        context = AgentContext(session_id="test-session")
        
        # Mock reasoning engine
        with patch.object(agent, '_reason', new_callable=AsyncMock) as mock_reason:
            mock_reason.return_value = {
                "response": "Let me check the weather.",
                "requires_tools": False,
                "confidence": 0.9
            }
            
            response = await agent.process(message, context)
            
            assert response.role == "assistant"
            assert "Let me check the weather" in response.content
            assert response.metadata["confidence"] == 0.9
            
    @pytest.mark.asyncio
    async def test_tool_execution(self, agent, mock_tools):
        """Test tool execution with circuit breaker"""
        message = Message(content="Search for AI news", role="user")
        context = AgentContext(session_id="test-session")
        
        # Mock reasoning and tool execution
        with patch.object(agent, '_reason', new_callable=AsyncMock) as mock_reason:
            mock_reason.return_value = {
                "response": "I'll search for AI news.",
                "requires_tools": True,
                "tool_calls": [
                    {"tool": "search", "parameters": {"query": "AI news"}}
                ]
            }
            
            with patch.object(agent, '_tool_executor') as mock_executor:
                mock_executor.execute = AsyncMock(return_value=ToolResult(
                    tool_name="search",
                    success=True,
                    data="Latest AI breakthroughs..."
                ))
                
                response = await agent.process(message, context)
                
                assert "search" in response.metadata["tools_used"]
                assert "Latest AI breakthroughs" in response.content
                
    @pytest.mark.asyncio
    async def test_memory_system(self, agent):
        """Test memory system integration"""
        message = Message(content="Remember this: API key is 12345", role="user")
        context = AgentContext(session_id="test-session")
        
        # Mock memory system
        with patch('src.gaia_components.enhanced_memory_system.EnhancedMemorySystem') as MockMemory:
            mock_memory = AsyncMock()
            MockMemory.return_value = mock_memory
            
            with patch.object(agent, '_reason', new_callable=AsyncMock) as mock_reason:
                mock_reason.return_value = {
                    "response": "I'll remember that.",
                    "requires_tools": False
                }
                
                response = await agent.process(message, context)
                
                # Verify memory was updated
                mock_memory.add_interaction.assert_called_once()
                
    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling and recovery"""
        message = Message(content="Cause an error", role="user")
        context = AgentContext(session_id="test-session")
        
        # Mock reasoning to raise error
        with patch.object(agent, '_reason', new_callable=AsyncMock) as mock_reason:
            mock_reason.side_effect = Exception("Test error")
            
            response = await agent.process(message, context)
            
            assert response.metadata["error"] is True
            assert "encountered an error" in response.content
            assert agent.state == AgentState.ERROR
            
    @pytest.mark.asyncio
    async def test_collaboration(self, agent, agent_config, mock_tools):
        """Test agent collaboration"""
        # Create second agent
        other_agent = UnifiedAgent(
            agent_id="helper-agent",
            name="Helper Agent",
            config=agent_config,
            tools=mock_tools,
            capabilities=[AgentCapability.COLLABORATION]
        )
        
        # Add collaboration capability to first agent
        agent.capabilities.append(AgentCapability.COLLABORATION)
        
        task = {"type": "research", "topic": "quantum computing"}
        
        with patch.object(other_agent, 'process', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = Message(
                content="Research completed",
                role="assistant"
            )
            
            result = await agent.collaborate(other_agent, task)
            
            assert result.content == "Research completed"
            mock_process.assert_called_once()
            
    def test_factory_function(self, agent_config, mock_tools):
        """Test agent factory function"""
        # Test default creation
        agent = create_agent()
        assert agent.name.startswith("unified_agent_")
        
        # Test with parameters
        agent = create_agent(
            agent_type="unified",
            name="Custom Agent",
            config=agent_config,
            tools=mock_tools
        )
        assert agent.name == "Custom Agent"
        assert len(agent.tools) == 2
        
    def test_metrics_collection(self, agent):
        """Test metrics collection"""
        metrics = agent.get_metrics()
        
        assert metrics["agent_id"] == "test-agent"
        assert metrics["name"] == "Test Agent"
        assert metrics["state"] == "idle"
        assert metrics["tools_count"] == 2
        assert "circuit_breaker_state" in metrics


class TestAgentIntegration:
    """Integration tests for agent system"""
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """Test complete conversation flow"""
        agent = create_agent(
            name="Integration Test Agent",
            capabilities=[
                AgentCapability.REASONING,
                AgentCapability.TOOL_USE,
                AgentCapability.MEMORY
            ]
        )
        
        context = AgentContext(
            session_id="integration-test",
            user_id="test-user"
        )
        
        # Simulate conversation
        messages = [
            "Hello, what can you do?",
            "Can you remember my name? It's Alice.",
            "What's my name?"
        ]
        
        for msg_content in messages:
            message = Message(content=msg_content, role="user")
            
            with patch.object(agent, '_reason', new_callable=AsyncMock) as mock_reason:
                if "remember" in msg_content:
                    mock_reason.return_value = {
                        "response": "I'll remember your name is Alice.",
                        "requires_tools": False
                    }
                elif "What's my name" in msg_content:
                    mock_reason.return_value = {
                        "response": "Your name is Alice.",
                        "requires_tools": False
                    }
                else:
                    mock_reason.return_value = {
                        "response": "I can help with various tasks.",
                        "requires_tools": False
                    }
                    
                response = await agent.process(message, context)
                assert response.role == "assistant"
                context.history.append(message)
                context.history.append(response)
