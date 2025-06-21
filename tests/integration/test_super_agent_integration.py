"""
Integration tests for the Super AI Agent subsystems
Tests the full integration of reasoning, memory, and tool execution
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

from src.agents.unified_agent import UnifiedAgent, AgentContext, create_agent
from src.infrastructure.agent_config import AgentConfig
from src.core.entities.base_message import Message
from src.gaia_components.advanced_reasoning_engine import AdvancedReasoningEngine, ReasoningType
from src.gaia_components.enhanced_memory_system import EnhancedMemorySystem, MemoryPriority
from src.gaia_components.tool_executor import create_production_tool_executor, ToolExecutionStatus


class TestReasoningEngineIntegration:
    """Integration tests for the Advanced Reasoning Engine"""
    
    @pytest.fixture
    def reasoning_engine(self):
        """Create a reasoning engine instance"""
        return AdvancedReasoningEngine()
    
    @pytest.mark.asyncio
    async def test_reasoning_with_context(self, reasoning_engine):
        """Test reasoning with rich context"""
        query = "What causes rain?"
        context = {
            "user_level": "beginner",
            "previous_topics": ["weather", "clouds"],
            "preferences": {"explanation_style": "simple"}
        }
        
        result = await reasoning_engine.reason(query, context)
        
        assert result is not None
        assert "response" in result
        assert "confidence" in result
        assert "reasoning_type" in result
        assert result["confidence"] > 0.5
        
    @pytest.mark.asyncio
    async def test_adaptive_reasoning_selection(self, reasoning_engine):
        """Test that reasoning type adapts to query"""
        test_cases = [
            {
                "query": "If all birds can fly and penguins are birds, can penguins fly?",
                "expected_type": ReasoningType.DEDUCTIVE
            },
            {
                "query": "I've seen 5 black ravens. What color are ravens?",
                "expected_type": ReasoningType.INDUCTIVE
            },
            {
                "query": "The grass is wet but it didn't rain. What happened?",
                "expected_type": ReasoningType.ABDUCTIVE
            }
        ]
        
        for case in test_cases:
            result = await reasoning_engine.reason(
                case["query"], 
                {"test": True}
            )
            
            # The reasoning engine should select appropriate type
            assert result["reasoning_type"] == case["expected_type"].value
            
    @pytest.mark.asyncio
    async def test_reasoning_path_generation(self, reasoning_engine):
        """Test that reasoning generates proper paths"""
        query = "Explain step by step how photosynthesis works"
        
        result = await reasoning_engine.reason(
            query,
            {"detailed": True},
            reasoning_type=ReasoningType.CHAIN_OF_THOUGHT
        )
        
        assert "reasoning_path" in result
        assert len(result["reasoning_path"]) >= 3
        assert all("step" in step for step in result["reasoning_path"])
        assert all("confidence" in step for step in result["reasoning_path"])


class TestMemorySystemIntegration:
    """Integration tests for the Enhanced Memory System"""
    
    @pytest.fixture
    def memory_system(self, tmp_path):
        """Create a memory system instance with temp storage"""
        return EnhancedMemorySystem(persist_directory=str(tmp_path))
    
    def test_memory_storage_and_retrieval(self, memory_system):
        """Test storing and retrieving different memory types"""
        # Store episodic memory
        episode_id = memory_system.store_episodic(
            content="User asked about machine learning",
            event_type="question",
            metadata={"topic": "ML", "timestamp": datetime.now()}
        )
        
        # Store semantic memory
        semantic_id = memory_system.store_semantic(
            content="Machine learning is a subset of AI",
            concepts=["machine", "learning", "AI"],
            metadata={"source": "knowledge"}
        )
        
        # Store working memory
        working_id = memory_system.store_working(
            content="Current task: Explain ML concepts",
            priority=MemoryPriority.HIGH
        )
        
        # Test retrieval
        ml_memories = memory_system.search_memories("machine learning")
        assert len(ml_memories) >= 2
        
        # Test concept retrieval
        concept_memories = memory_system.retrieve_semantic_by_concept("machine")
        assert len(concept_memories) >= 1
        
        # Test statistics
        stats = memory_system.get_memory_statistics()
        assert stats["episodic_count"] >= 1
        assert stats["semantic_count"] >= 1
        assert stats["working_count"] >= 1
        
    def test_memory_consolidation(self, memory_system):
        """Test memory consolidation from working to semantic"""
        # Store high-strength working memory
        memory_id = memory_system.store_working(
            content="Important fact: Python is a programming language",
            priority=MemoryPriority.CRITICAL,
            metadata={"importance": "high"}
        )
        
        # Access it multiple times to increase strength
        for _ in range(5):
            memory = memory_system.retrieve_working(memory_id)
            if memory:
                memory.access()
        
        # Force consolidation
        memory_system.consolidation_threshold = 0.3  # Lower threshold for testing
        memory_system.consolidation_interval = 0  # Immediate consolidation
        memory_system.consolidate_memories()
        
        # Check if memory was moved to semantic
        semantic_memories = memory_system.search_memories("Python programming")
        assert any("Python" in m.content for m in semantic_memories)
        
    def test_memory_persistence(self, memory_system):
        """Test saving and loading memories"""
        # Store some memories
        memory_system.store_episodic(
            "Test episodic memory",
            "test_event"
        )
        memory_system.store_semantic(
            "Test semantic memory",
            ["test", "semantic"]
        )
        
        # Save memories
        memory_system.save_memories()
        
        # Create new memory system with same directory
        new_memory_system = EnhancedMemorySystem(
            persist_directory=memory_system.persist_directory
        )
        
        # Check if memories were loaded
        stats = new_memory_system.get_memory_statistics()
        assert stats["episodic_count"] >= 1
        assert stats["semantic_count"] >= 1


class TestToolExecutorIntegration:
    """Integration tests for the Production Tool Executor"""
    
    @pytest.fixture
    async def tool_executor(self):
        """Create a tool executor instance"""
        executor = create_production_tool_executor(max_workers=5)
        async with executor:
            yield executor
    
    @pytest.mark.asyncio
    async def test_calculator_tool(self, tool_executor):
        """Test calculator tool execution"""
        result = await tool_executor.execute(
            "calculator",
            expression="2 + 2 * 3"
        )
        
        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["result"] == 8.0
        assert result.execution_time > 0
        
    @pytest.mark.asyncio
    async def test_text_processor_tool(self, tool_executor):
        """Test text processing tool"""
        text = "The quick brown fox jumps over the lazy dog. This sentence contains every letter."
        
        # Test keyword extraction
        result = await tool_executor.execute(
            "text_processor",
            text=text,
            operation="extract_keywords"
        )
        
        assert result.status == ToolExecutionStatus.SUCCESS
        assert "keywords" in result.result
        assert len(result.result["keywords"]) > 0
        
    @pytest.mark.asyncio
    async def test_data_analyzer_tool(self, tool_executor):
        """Test data analysis tool"""
        sample_data = [
            {"x": 1, "y": 2},
            {"x": 2, "y": 4},
            {"x": 3, "y": 6},
            {"x": 4, "y": 8}
        ]
        
        result = await tool_executor.execute(
            "data_analyzer",
            data=sample_data,
            analysis_type="basic"
        )
        
        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.result["shape"] == (4, 2)
        assert "summary" in result.result
        
    @pytest.mark.asyncio
    async def test_tool_timeout(self, tool_executor):
        """Test tool execution timeout"""
        # Set a very short timeout
        tool_executor.execution_timeout = 0.001
        
        # Execute a tool that takes time
        result = await tool_executor.execute(
            "calculator",
            expression="sum(range(1000000))"  # Time-consuming calculation
        )
        
        assert result.status == ToolExecutionStatus.TIMEOUT
        assert "timeout" in result.error.lower()
        
    @pytest.mark.asyncio
    async def test_tool_statistics(self, tool_executor):
        """Test execution statistics tracking"""
        # Execute multiple tools
        await tool_executor.execute("calculator", expression="1+1")
        await tool_executor.execute("calculator", expression="2+2")
        await tool_executor.execute("text_processor", text="test", operation="analyze")
        
        stats = tool_executor.get_execution_stats()
        
        assert "calculator" in stats
        assert stats["calculator"]["total_calls"] == 2
        assert stats["calculator"]["successful_calls"] == 2
        assert "text_processor" in stats


class TestUnifiedAgentIntegration:
    """Integration tests for the Unified Agent with all subsystems"""
    
    @pytest.fixture
    async def agent(self):
        """Create a unified agent with all capabilities"""
        config = AgentConfig(
            model_name="gpt-4",
            temperature=0.5,
            max_iterations=10,
            enable_memory=True
        )
        
        agent = create_agent(
            agent_type="unified",
            name="Test Agent",
            config=config,
            capabilities=['reasoning', 'tool_use', 'memory', 'collaboration']
        )
        
        return agent
    
    @pytest.fixture
    def context(self):
        """Create agent context"""
        return AgentContext(
            session_id="test_session",
            user_id="test_user",
            metadata={"test": True}
        )
    
    @pytest.mark.asyncio
    async def test_agent_with_reasoning(self, agent, context):
        """Test agent processes messages with reasoning"""
        message = Message(
            content="What is the capital of France?",
            role="user"
        )
        
        response = await agent.process(message, context)
        
        assert response is not None
        assert response.role == "assistant"
        assert len(response.content) > 0
        assert "confidence" in response.metadata
        
    @pytest.mark.asyncio
    async def test_agent_memory_integration(self, agent, context):
        """Test agent memory integration"""
        # First interaction
        message1 = Message(
            content="My name is Alice and I like quantum physics",
            role="user"
        )
        response1 = await agent.process(message1, context)
        
        # Second interaction referencing first
        message2 = Message(
            content="What did I tell you my name was?",
            role="user"
        )
        response2 = await agent.process(message2, context)
        
        # Agent should remember the name if memory is working
        # Note: This is a mock test since we're not using real LLM
        assert response2 is not None
        
    @pytest.mark.asyncio
    async def test_agent_state_transitions(self, agent, context):
        """Test agent state transitions during processing"""
        from src.agents.unified_agent import AgentState
        
        # Check initial state
        assert agent.get_state() == AgentState.IDLE
        
        # Start processing
        message = Message(content="Test message", role="user")
        
        # We can't easily test intermediate states in async,
        # but we can verify final state
        response = await agent.process(message, context)
        
        # Should return to IDLE after processing
        assert agent.get_state() == AgentState.IDLE
        
    @pytest.mark.asyncio
    async def test_agent_error_recovery(self, agent, context):
        """Test agent error handling and recovery"""
        # Send a message that might cause an error
        message = Message(
            content="Process this: " + "x" * 10000,  # Very long message
            role="user",
            metadata={"force_error": True}
        )
        
        response = await agent.process(message, context)
        
        # Agent should handle error gracefully
        assert response is not None
        assert response.role == "assistant"
        # Error response should be generated
        if response.metadata.get("error"):
            assert "error" in response.content.lower()
            
    @pytest.mark.asyncio
    async def test_multi_agent_collaboration(self):
        """Test multiple agents collaborating"""
        # Create two agents
        config = AgentConfig(enable_memory=True)
        
        agent1 = create_agent(
            agent_type="unified",
            name="Agent 1",
            config=config,
            capabilities=['reasoning', 'memory', 'collaboration']
        )
        
        agent2 = create_agent(
            agent_type="unified", 
            name="Agent 2",
            config=config,
            capabilities=['reasoning', 'memory', 'collaboration']
        )
        
        # Test collaboration
        task = {
            "action": "analyze",
            "data": "Test collaboration data"
        }
        
        result = await agent1.collaborate(agent2, task)
        
        assert result is not None
        assert isinstance(result, Message)


class TestEndToEndIntegration:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_complete_agent_pipeline(self):
        """Test the complete agent pipeline with all subsystems"""
        # Create all components
        memory_system = EnhancedMemorySystem()
        reasoning_engine = AdvancedReasoningEngine()
        tool_executor = create_production_tool_executor()
        
        # Create agent
        config = AgentConfig(
            model_name="gpt-4",
            temperature=0.5,
            enable_memory=True
        )
        
        agent = create_agent(
            agent_type="unified",
            name="Integration Test Agent",
            config=config,
            capabilities=['reasoning', 'tool_use', 'memory']
        )
        
        context = AgentContext(
            session_id="integration_test",
            metadata={"test_mode": True}
        )
        
        async with tool_executor:
            # Test 1: Simple query
            msg1 = Message(content="What is 2+2?", role="user")
            response1 = await agent.process(msg1, context)
            assert response1 is not None
            
            # Test 2: Complex reasoning
            msg2 = Message(
                content="Explain why water freezes at 0°C",
                role="user"
            )
            response2 = await agent.process(msg2, context)
            assert response2 is not None
            assert response2.metadata.get("reasoning_path") is not None
            
            # Test 3: Tool usage
            msg3 = Message(
                content="Calculate the square root of 144",
                role="user"
            )
            response3 = await agent.process(msg3, context)
            assert response3 is not None
            
            # Verify agent metrics
            metrics = agent.get_metrics()
            assert metrics["state"] == "idle"
            assert "capabilities" in metrics
            
        # Verify memory was used
        if hasattr(agent, '_memory_system') and agent._memory_system:
            stats = agent._memory_system.get_memory_statistics()
            assert stats["total_memories"] > 0


if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__, "-v", "-k", "test_complete_agent_pipeline"])