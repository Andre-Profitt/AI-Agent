"""
End-to-End Integration Tests
Tests complete system functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from src.agents.unified_agent import create_agent, AgentCapability
from src.core.entities.message import Message
from src.infrastructure.config import AgentConfig
from src.unified_architecture.platform import UnifiedPlatform
from src.services.integration_hub import IntegrationHub


class TestEndToEndIntegration:
    """Test complete system integration"""
    
    @pytest.fixture
    async def platform(self):
        """Create test platform"""
        platform = UnifiedPlatform()
        await platform.initialize()
        yield platform
        await platform.shutdown()
        
    @pytest.fixture
    def integration_hub(self):
        """Create integration hub"""
        return IntegrationHub()
        
    @pytest.mark.asyncio
    async def test_multi_agent_collaboration(self, platform):
        """Test multiple agents working together"""
        # Create specialized agents
        research_agent = create_agent(
            name="Research Agent",
            capabilities=[
                AgentCapability.REASONING,
                AgentCapability.TOOL_USE,
                AgentCapability.COLLABORATION
            ]
        )
        
        analysis_agent = create_agent(
            name="Analysis Agent",
            capabilities=[
                AgentCapability.REASONING,
                AgentCapability.MEMORY,
                AgentCapability.COLLABORATION
            ]
        )
        
        # Register agents
        await platform.register_agent(research_agent)
        await platform.register_agent(analysis_agent)
        
        # Create collaborative task
        task = {
            "type": "research_and_analyze",
            "topic": "AI trends 2024",
            "steps": [
                {"agent": "Research Agent", "action": "gather_data"},
                {"agent": "Analysis Agent", "action": "analyze_findings"}
            ]
        }
        
        # Mock agent responses
        with patch.object(research_agent, 'process', new_callable=AsyncMock) as mock_research:
            mock_research.return_value = Message(
                content="Found 10 key AI trends for 2024...",
                role="assistant",
                metadata={"data_points": 10}
            )
            
            with patch.object(analysis_agent, 'process', new_callable=AsyncMock) as mock_analysis:
                mock_analysis.return_value = Message(
                    content="Analysis complete: Top trends are...",
                    role="assistant",
                    metadata={"confidence": 0.85}
                )
                
                # Execute task
                result = await platform.execute_task(task)
                
                assert result["status"] == "completed"
                assert len(result["steps"]) == 2
                
    @pytest.mark.asyncio
    async def test_tool_integration_flow(self):
        """Test agent with multiple tool integrations"""
        # Create agent with tools
        tools = [
            Mock(name="web_search", execute=AsyncMock()),
            Mock(name="calculator", execute=AsyncMock()),
            Mock(name="file_reader", execute=AsyncMock())
        ]
        
        agent = create_agent(
            name="Tool User",
            tools=tools,
            capabilities=[AgentCapability.TOOL_USE]
        )
        
        # Test complex query requiring multiple tools
        query = "Search for the latest GDP data and calculate the growth rate"
        
        with patch.object(agent, '_reason', new_callable=AsyncMock) as mock_reason:
            mock_reason.return_value = {
                "response": "I'll search for GDP data and calculate the growth rate.",
                "requires_tools": True,
                "tool_calls": [
                    {"tool": "web_search", "parameters": {"query": "latest GDP data"}},
                    {"tool": "calculator", "parameters": {"expression": "(new-old)/old*100"}}
                ]
            }
            
            # Mock tool results
            tools[0].execute.return_value = {"data": "GDP: 2023: $25T, 2024: $26T"}
            tools[1].execute.return_value = {"result": 4.0}
            
            response = await agent.process(
                Message(content=query, role="user"),
                AgentContext(session_id="test")
            )
            
            assert "4" in response.content or "growth" in response.content.lower()
            
    @pytest.mark.asyncio
    async def test_error_recovery_flow(self):
        """Test system error recovery"""
        agent = create_agent(name="Resilient Agent")
        
        # Simulate intermittent failures
        call_count = 0
        
        async def flaky_reasoning(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count < 3:
                raise Exception("Temporary failure")
                
            return {
                "response": "Success after retries",
                "requires_tools": False
            }
            
        with patch.object(agent, '_reason', new=flaky_reasoning):
            # First attempt should fail but recover
            response = await agent.process(
                Message(content="Test resilience", role="user"),
                AgentContext(session_id="test")
            )
            
            # Should eventually succeed
            assert "error" in response.content.lower()
            
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance with concurrent requests"""
        agent = create_agent(
            name="Performance Test Agent",
            config=AgentConfig(max_concurrent_requests=5)
        )
        
        # Create multiple concurrent requests
        async def make_request(i: int):
            message = Message(content=f"Request {i}", role="user")
            context = AgentContext(session_id=f"session-{i}")
            
            with patch.object(agent, '_reason', new_callable=AsyncMock) as mock:
                mock.return_value = {
                    "response": f"Response to request {i}",
                    "requires_tools": False
                }
                
                return await agent.process(message, context)
                
        # Run 20 concurrent requests
        start_time = asyncio.get_event_loop().time()
        responses = await asyncio.gather(*[
            make_request(i) for i in range(20)
        ])
        end_time = asyncio.get_event_loop().time()
        
        # Verify all succeeded
        assert len(responses) == 20
        assert all(r.role == "assistant" for r in responses)
        
        # Should have rate limited (not all concurrent)
        duration = end_time - start_time
        assert duration > 0.5  # Some throttling occurred
        
    @pytest.mark.asyncio
    async def test_integration_hub_connections(self, integration_hub):
        """Test external service integrations"""
        # Test service registration
        mock_service = Mock(
            name="test_service",
            connect=AsyncMock(return_value=True),
            disconnect=AsyncMock(return_value=True),
            execute=AsyncMock(return_value={"status": "success"})
        )
        
        integration_hub.register_service(mock_service)
        
        # Test connection
        await integration_hub.connect("test_service")
        mock_service.connect.assert_called_once()
        
        # Test execution
        result = await integration_hub.execute(
            "test_service",
            "test_action",
            {"param": "value"}
        )
        
        assert result["status"] == "success"
        
        # Test disconnection
        await integration_hub.disconnect("test_service")
        mock_service.disconnect.assert_called_once()


class TestSystemReliability:
    """Test system reliability and fault tolerance"""
    
    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self):
        """Test that system doesn't leak memory"""
        import gc
        import weakref
        
        # Create and destroy many agents
        refs = []
        
        for i in range(100):
            agent = create_agent(name=f"Agent-{i}")
            refs.append(weakref.ref(agent))
            del agent
            
        # Force garbage collection
        gc.collect()
        
        # Check that agents were cleaned up
        alive_count = sum(1 for ref in refs if ref() is not None)
        assert alive_count == 0
        
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, platform):
        """Test graceful shutdown procedures"""
        # Start some tasks
        async def long_running_task():
            await asyncio.sleep(10)
            
        task = asyncio.create_task(long_running_task())
        
        # Shutdown should cancel tasks gracefully
        await platform.shutdown()
        
        assert task.cancelled()
