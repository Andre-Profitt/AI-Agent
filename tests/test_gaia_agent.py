from benchmarks.cot_performance import insights
from performance_dashboard import stats

from src.agents.advanced_agent_fsm import Agent

from src.agents.advanced_agent_fsm import FSMReActAgent
# TODO: Fix undefined variables: insights, result, stats
from src.agents.gaiaenhancedagent import GAIAEnhancedAgent

# TODO: Fix undefined variables: GAIAEnhancedAgent, insights, result, stats

"""
Basic tests for GAIA-Enhanced FSMReActAgent
"""

import pytest

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