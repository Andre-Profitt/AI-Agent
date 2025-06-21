#!/usr/bin/env python3
from agent import query
from agent import tools
from benchmarks.cot_performance import duration
from examples.enhanced_unified_example import start_time
from examples.enhanced_unified_example import task
from examples.enhanced_unified_example import tasks
from examples.parallel_execution_example import agents
from examples.parallel_execution_example import end_time
from examples.parallel_execution_example import executor
from examples.parallel_execution_example import final_result
from examples.parallel_execution_example import inputs
from examples.parallel_execution_example import items
from examples.parallel_execution_example import operations
from examples.parallel_execution_example import results
from examples.parallel_execution_example import summary
from examples.parallel_execution_example import tool_calls
from examples.parallel_execution_example import tool_name
from examples.parallel_execution_example import total_time
from setup_environment import value
from tests.load_test import success
from tests.unit.simple_test import func

from src.application.tools.tool_executor import expression
from src.core.entities.agent import Agent
from src.core.monitoring import key
from src.database.models import agent_id
from src.database.models import text
from src.gaia_components.monitoring import reset_metrics
from src.infrastructure.monitoring.metrics_integration import get_metrics_summary
from src.tools_introspection import name
from src.utils.knowledge_utils import word

from src.tools.base_tool import Tool

from src.agents.advanced_agent_fsm import Agent
from src.application.executors.parallel_executor import ParallelFSMReactAgent
# TODO: Fix undefined variables: agent_id, agents, city, duration, e, end_time, executor, expression, final_result, func, get_metrics_summary, i, inputs, items, key, logging, name, num, operations, os, query, reset_metrics, result, results, start_time, success, summary, sys, target_language, task, tasks, text, tool_calls, tool_name, tools, total_time, value, word
from tests.test_gaia_agent import agent

# TODO: Fix undefined variables: agent, agent_id, agents, city, duration, e, end_time, executor, expression, final_result, func, get_metrics_summary, i, inputs, items, key, name, num, operations, query, reset_metrics, result, results, self, start_time, success, summary, target_language, task, tasks, text, tool_calls, tool_name, tools, total_time, traceback, value, word

"""

from sqlalchemy import func
Example demonstrating parallel execution capabilities.

This script shows how to use the ParallelExecutor for:
1. Parallel tool execution
2. Parallel agent execution
3. Map-reduce operations
4. Performance monitoring
"""

import asyncio
import sys
import os
import logging

logger = logging.getLogger(__name__)

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.application.executors.parallel_executor import ParallelExecutor, ParallelFSMReactAgent
from src.infrastructure.monitoring.decorators import get_metrics_summary, reset_metrics

async def demo_parallel_tool_execution():
    """Demonstrate parallel tool execution"""
    logger.info("\n=== Parallel Tool Execution Demo ===")

    # Create parallel executor
    executor = ParallelExecutor(max_workers=5)

    # Define mock tools that simulate real operations
    async def web_search(query: str) -> str:
        await asyncio.sleep(1)  # Simulate API call
        return f"Search results for: {query}"

    async def calculate(expression: str) -> float:
        await asyncio.sleep(0.5)  # Simulate calculation
        return eval(expression)  # Note: unsafe in production

    async def analyze_text(text: str) -> dict:
        await asyncio.sleep(2)  # Simulate analysis
        return {
            "length": len(text),
            "words": len(text.split()),
            "sentences": len(text.split('.')),
            "avg_word_length": sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0
        }

    async def fetch_weather(city: str) -> dict:
        await asyncio.sleep(1.5)  # Simulate API call
        return {
            "city": city,
            "temperature": 22.5,
            "condition": "sunny",
            "humidity": 65
        }

    async def translate_text(text: str, target_language: str) -> str:
        await asyncio.sleep(1)  # Simulate translation
        return f"Translated '{text}' to {target_language}"

    # Execute tools in parallel
    tools = [web_search, calculate, analyze_text, fetch_weather, translate_text]
    inputs = [
        {"query": "parallel execution python"},
        {"expression": "2 + 2 * 3"},
        {"text": "This is a sample text for analysis. It contains multiple sentences."},
        {"city": "New York"},
        {"text": "Hello world", "target_language": "Spanish"}
    ]

    logger.info("Executing 5 tools in parallel...")
    start_time = asyncio.get_event_loop().time()

    results = await executor.execute_tools_parallel(tools, inputs, timeout=10.0)

    end_time = asyncio.get_event_loop().time()
    total_time = end_time - start_time

    logger.info("Completed in {} seconds", extra={"total_time": total_time})
    logger.info("Results:")

    for i, (success, result) in enumerate(results):
        tool_name = tools[i].__name__
        if success:
            logger.info("  ✓ {}: {}", extra={"tool_name": tool_name, "result": result})
        else:
            logger.info("  ✗ {}: Error - {}", extra={"tool_name": tool_name, "result": result})

    # Cleanup
    executor.shutdown()

async def demo_map_reduce():
    """Demonstrate map-reduce operations"""
    logger.info("\n=== Map-Reduce Demo ===")

    executor = ParallelExecutor(max_workers=8)

    # Define map and reduce functions
    async def process_number(num: int) -> int:
        await asyncio.sleep(0.1)  # Simulate processing
        return num * num

    def sum_results(self, results: list) -> int:
        return sum(results)

    # Process a large dataset
    items = list(range(100))
    logger.info("Processing {} items with map-reduce...", extra={"len_items_": len(items)})

    start_time = asyncio.get_event_loop().time()

    final_result = await executor.map_reduce(
        process_number, sum_results, items, chunk_size=10
    )

    end_time = asyncio.get_event_loop().time()
    total_time = end_time - start_time

    logger.info("Sum of squares: {}", extra={"final_result": final_result})
    logger.info("Completed in {} seconds", extra={"total_time": total_time})

    # Cleanup
    executor.shutdown()

async def demo_parallel_agent_execution():
    """Demonstrate parallel agent execution"""
    logger.info("\n=== Parallel Agent Execution Demo ===")

    executor = ParallelExecutor(max_workers=3)

    # Mock agents
    class MockAgent:
        def __init__(self, agent_id: str, name: str):
            self.agent_id = agent_id
            self.name = name

        async def execute(self, task: dict) -> dict:
            await asyncio.sleep(1)  # Simulate agent processing
            return {
                "agent_id": self.agent_id,
                "agent_name": self.name,
                "task": task["description"],
                "result": f"Processed by {self.name}",
                "status": "completed"
            }

    # Create mock agents
    agents = [
        MockAgent("agent_1", "Research Agent"),
        MockAgent("agent_2", "Analysis Agent"),
        MockAgent("agent_3", "Synthesis Agent")
    ]

    # Define tasks
    tasks = [
        {"description": "Research market trends"},
        {"description": "Analyze competitor data"},
        {"description": "Synthesize findings"}
    ]

    logger.info("Executing 3 agents in parallel...")
    start_time = asyncio.get_event_loop().time()

    results = await executor.execute_agents_parallel(agents, tasks, max_concurrent=2)

    end_time = asyncio.get_event_loop().time()
    total_time = end_time - start_time

    logger.info("Completed in {} seconds", extra={"total_time": total_time})
    logger.info("Results:")

    for agent_id, result in results:
        if "error" not in result:
            logger.info("  ✓ {}: {}", extra={"agent_id": agent_id, "result__result_": result['result']})
        else:
            logger.info("  ✗ {}: Error - {}", extra={"agent_id": agent_id, "result__error_": result['error']})

    # Cleanup
    executor.shutdown()

async def demo_performance_monitoring():
    """Demonstrate performance monitoring"""
    logger.info("\n=== Performance Monitoring Demo ===")

    # Reset metrics
    reset_metrics()

    # Run some operations to generate metrics
    executor = ParallelExecutor(max_workers=4)

    async def monitored_operation(name: str, duration: float):
        await asyncio.sleep(duration)
        return f"Operation {name} completed"

    # Execute multiple monitored operations
    operations = [
        ("A", 0.5),
        ("B", 1.0),
        ("C", 0.3),
        ("D", 0.8)
    ]

    tasks = [monitored_operation(name, duration) for name, duration in operations]
    await asyncio.gather(*tasks)

    # Get metrics summary
    summary = get_metrics_summary()

    logger.info("Performance Metrics Summary:")
    for key, value in summary.items():
        if key != "timestamp":
            logger.info("  {}: {}", extra={"key": key, "value": value})

    # Cleanup
    executor.shutdown()

async def demo_parallel_fsm_agent():
    """Demonstrate parallel FSM agent"""
    logger.info("\n=== Parallel FSM Agent Demo ===")

    # Mock tools for the FSM agent
    class MockTool:
        def __init__(self, name: str, func):
            self.name = name
            self.func = func

    async def search_tool(query: str) -> str:
        await asyncio.sleep(1)
        return f"Search results for: {query}"

    async def calculate_tool(expression: str) -> float:
        await asyncio.sleep(0.5)
        return eval(expression)

    async def analyze_tool(text: str) -> dict:
        await asyncio.sleep(1.5)
        return {"word_count": len(text.split()), "char_count": len(text)}

    # Create tools
    tools = [
        MockTool("search", search_tool),
        MockTool("calculate", calculate_tool),
        MockTool("analyze", analyze_tool)
    ]

    # Create parallel FSM agent
    agent = ParallelFSMReactAgent(tools, max_parallel_tools=3)

    # Define tool calls
    tool_calls = [
        {"tool_name": "search", "arguments": {"query": "parallel processing"}},
        {"tool_name": "calculate", "arguments": {"expression": "10 * 5 + 2"}},
        {"tool_name": "analyze", "arguments": {"text": "This is a sample text for analysis."}}
    ]

    logger.info("Executing tool calls in parallel with FSM agent...")
    start_time = asyncio.get_event_loop().time()

    results = await agent.execute_tools_parallel(tool_calls)

    end_time = asyncio.get_event_loop().time()
    total_time = end_time - start_time

    logger.info("Completed in {} seconds", extra={"total_time": total_time})
    logger.info("Results:")

    for result in results:
        tool_name = result["tool_name"]
        if result["success"]:
            logger.info("  ✓ {}: {}", extra={"tool_name": tool_name, "result__result_": result['result']})
        else:
            logger.info("  ✗ {}: Error - {}", extra={"tool_name": tool_name, "result__error_": result['error']})

async def main():
    """Run all demos"""
    logger.info("🚀 Parallel Execution Demo Suite")
    print("=" * 50)

    try:
        await demo_parallel_tool_execution()
        await demo_map_reduce()
        await demo_parallel_agent_execution()
        await demo_performance_monitoring()
        await demo_parallel_fsm_agent()

        logger.info("\n✅ All demos completed successfully!")

    except Exception as e:
        logger.info("\n❌ Demo failed: {}", extra={"e": e})
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())