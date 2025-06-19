"""
Basic tests for GAIA-Enhanced FSMReActAgent
"""

import pytest
import asyncio
from examples.gaia_usage_example import GAIAEnhancedAgent

@pytest.fixture
def agent():
    """Create a test agent"""
    return GAIAEnhancedAgent()

@pytest.mark.asyncio
async def test_basic_query(agent):
    """Test basic query functionality"""
    result = await agent.solve("What is 2 + 2?")
    assert result["success"] is True
    assert "answer" in result

@pytest.mark.asyncio
async def test_memory_system(agent):
    """Test memory system functionality"""
    stats = agent.get_memory_insights()
    assert isinstance(stats, dict)

@pytest.mark.asyncio
async def test_tool_system(agent):
    """Test adaptive tool system"""
    insights = agent.get_tool_insights()
    assert isinstance(insights, dict)
