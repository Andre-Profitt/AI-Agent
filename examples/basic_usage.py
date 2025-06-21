from src.core.entities.agent import Agent

from src.agents.advanced_agent_fsm import Agent

from src.agents.advanced_agent_fsm import FSMReActAgent
# TODO: Fix undefined variables: result
from tests.test_gaia_agent import agent

from src.agents.gaiaenhancedagent import GAIAEnhancedAgent

# TODO: Fix undefined variables: GAIAEnhancedAgent, agent, result

"""
Basic usage example for GAIA-Enhanced FSMReActAgent
"""

import asyncio
from examples.gaia_usage_example import GAIAEnhancedAgent

async def main():
    # Initialize the agent
    agent = GAIAEnhancedAgent()

    # Simple query
    result = await agent.solve("What is 2 + 2?")
    print(f"Answer: {result['answer']}")

    # Complex query
    result = await agent.solve("Calculate the population density of New York City")
    print(f"Answer: {result['answer']}")

if __name__ == "__main__":
    asyncio.run(main())