from agent import query
from agent import tools
from benchmarks.cot_performance import semaphore
from examples.enhanced_unified_example import execution_time
from examples.enhanced_unified_example import start_time
from examples.enhanced_unified_example import task
from examples.enhanced_unified_example import tasks
from examples.parallel_execution_example import agents
from examples.parallel_execution_example import executor
from examples.parallel_execution_example import final_result
from examples.parallel_execution_example import inputs
from examples.parallel_execution_example import items
from examples.parallel_execution_example import results
from examples.parallel_execution_example import tool_calls
from examples.parallel_execution_example import tool_name
from tests.load_test import args
from tests.load_test import success
from tests.unit.simple_test import func

from src.application.executors.parallel_executor import arguments
from src.application.executors.parallel_executor import map_results
from src.application.executors.parallel_executor import map_tasks
from src.application.executors.parallel_executor import result_dict
from src.application.tools.tool_executor import expression
from src.application.tools.tool_executor import processed_results
from src.core.entities.agent import Agent
from src.core.langgraph_compatibility import batch
from src.core.langgraph_compatibility import loop
from src.database.models import input_data
from src.database.models import text
from src.database.models import tool
from src.gaia_components.enhanced_memory_system import item
from src.gaia_components.performance_optimization import batch_results
from src.gaia_components.performance_optimization import batch_size
from src.gaia_components.performance_optimization import max_workers
from src.gaia_components.performance_optimization import valid_results
from src.gaia_components.production_vector_store import formatted_results
from src.utils.error_category import call
from src.workflow.workflow_automation import timeout

from src.tools.base_tool import Tool

from src.agents.advanced_agent_fsm import Agent

from src.agents.advanced_agent_fsm import IUnifiedAgent
from src.unified_architecture.enhanced_platform import IUnifiedAgent
from src.unified_architecture.enhanced_platform import UnifiedTask
# TODO: Fix undefined variables: Any, Callable, Dict, List, Optional, Tuple, agents, args, arguments, batch, batch_results, batch_size, call, dataclass, e, execution_time, executor, expression, final_result, formatted_results, func, i, input_data, inputs, item, items, kwargs, logging, loop, map_func, map_results, map_tasks, max_concurrent, max_parallel_tools, max_workers, processed_results, query, reduce_func, result, result_dict, results, semaphore, start_time, success, t, task, tasks, text, time, timeout, tool_calls, tool_name, tools, valid_results, wait
from concurrent.futures import ThreadPoolExecutor
from tests.test_gaia_agent import agent

from src.tools.base_tool import tool

# TODO: Fix undefined variables: ThreadPoolExecutor, agent, agents, args, arguments, batch, batch_results, batch_size, call, e, execution_time, executor, expression, final_result, formatted_results, func, i, input_data, inputs, item, items, kwargs, loop, map_func, map_results, map_tasks, max_concurrent, max_parallel_tools, max_workers, processed_results, query, reduce_func, result, result_dict, results, self, semaphore, start_time, success, t, task, tasks, text, timeout, tool, tool_calls, tool_name, tools, valid_results, wait

"""

from sqlalchemy import func
from unittest.mock import call
Parallel Executor for concurrent task execution

This module provides parallel execution capabilities for tools and agents,
enabling efficient concurrent processing of multiple tasks.
"""

import logging
logger = logging.getLogger(__name__)
from typing import Tuple
from typing import Optional
from typing import Any
from typing import List
from typing import Callable

import asyncio
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass

import time

@dataclass
class ExecutionResult:
    """Result of parallel execution"""
    success: bool
    result: Any
    execution_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

class ParallelExecutor:
    """
    Parallel execution engine for tools and agents.

    This class provides efficient concurrent execution of multiple tasks,
    with support for both async and sync operations, resource management,
    and error handling.
    """

    def __init__(self, max_workers: int = 4, max_concurrent: int = 10):
        self.max_workers = max_workers
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.logger = logging.getLogger(__name__)

    async def execute_tools_parallel(
        self,
        tools: List[Callable],
        inputs: List[Dict[str, Any]]
    ) -> List[Tuple[bool, Any]]:
        """
        Execute tools in parallel.

        Args:
            tools: List of tool functions to execute
            inputs: List of input dictionaries for each tool

        Returns:
            List of (success, result) tuples
        """
        if len(tools) != len(inputs):
            raise ValueError("Number of tools must match number of inputs")

        async def execute_single_tool(tool: Callable, input_data: Dict[str, Any]) -> Tuple[bool, Any]:
            async with self.semaphore:
                start_time = time.time()
                try:
                    if asyncio.iscoroutinefunction(tool):
                        result = await tool(**input_data)
                    else:
                        # Run sync function in thread pool
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(self.thread_pool, tool, **input_data)

                    execution_time = time.time() - start_time
                    self.logger.debug("Tool executed successfully in {}s", extra={"execution_time": execution_time})
                    return True, result

                except Exception as e:
                    execution_time = time.time() - start_time
                    self.logger.error("Tool execution failed: {}", extra={"e": e})
                    return False, str(e)

        # Create tasks for all tools
        tasks = [execute_single_tool(tool, input_data) for tool, input_data in zip(tools, inputs)]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append((False, str(result)))
            else:
                processed_results.append(result)

        return processed_results

    async def execute_agents_parallel(
        self,
        agents: List[IUnifiedAgent],
        tasks: List[UnifiedTask],
        max_concurrent: Optional[int] = None
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Execute agents in parallel.

        Args:
            agents: List of agents to execute
            tasks: List of tasks to execute
            max_concurrent: Maximum concurrent executions (overrides default)

        Returns:
            List of (agent_id, result) tuples
        """
        if len(agents) != len(tasks):
            raise ValueError("Number of agents must match number of tasks")

        # Use provided max_concurrent or default
        semaphore = asyncio.Semaphore(max_concurrent or self.max_concurrent)

        async def execute_single_agent(agent: IUnifiedAgent, task: UnifiedTask) -> Tuple[str, Dict[str, Any]]:
            async with semaphore:
                start_time = time.time()
                try:
                    result = await agent.execute(task)
                    execution_time = time.time() - start_time

                    # Convert result to dict if it's a TaskResult
                    if hasattr(result, '__dict__'):
                        result_dict = result.__dict__
                    else:
                        result_dict = {"result": result}

                    result_dict["execution_time"] = execution_time
                    result_dict["agent_id"] = agent.agent_id

                    self.logger.debug("Agent {} executed task {} in {}s", extra={"agent_agent_id": agent.agent_id, "task_task_id": task.task_id, "execution_time": execution_time})
                    return agent.agent_id, result_dict

                except Exception as e:
                    execution_time = time.time() - start_time
                    self.logger.error("Agent {} failed to execute task {}: {}", extra={"agent_agent_id": agent.agent_id, "task_task_id": task.task_id, "e": e})
                    return agent.agent_id, {
                        "error": str(e),
                        "execution_time": execution_time,
                        "agent_id": agent.agent_id
                    }

        # Create tasks for all agents
        tasks = [execute_single_agent(agent, task) for agent, task in zip(agents, tasks)]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(("unknown", {"error": str(result)}))
            else:
                processed_results.append(result)

        return processed_results

    async def map_reduce(
        self,
        map_func: Callable,
        reduce_func: Callable,
        items: List[Any]
    ) -> Any:
        """
        Execute map-reduce pattern.

        Args:
            map_func: Function to apply to each item
            reduce_func: Function to combine results
            items: List of items to process

        Returns:
            Reduced result
        """
        async def map_item(item: Any) -> Any:
            async with self.semaphore:
                try:
                    if asyncio.iscoroutinefunction(map_func):
                        return await map_func(item)
                    else:
                        # Run sync function in thread pool
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(self.thread_pool, map_func, item)
                except Exception as e:
                    self.logger.error("Map function failed for item {}: {}", extra={"item": item, "e": e})
                    raise

        # Map phase - execute map function on all items
        map_tasks = [map_item(item) for item in items]
        map_results = await asyncio.gather(*map_tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for result in map_results:
            if isinstance(result, Exception):
                self.logger.warning("Map operation failed: {}", extra={"result": result})
            else:
                valid_results.append(result)

        # Reduce phase - combine results
        if not valid_results:
            raise ValueError("No valid results from map phase")

        return reduce_func(valid_results)

    async def execute_with_timeout(
        self,
        func: Callable,
        timeout: float,
        *args,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute a function with timeout.

        Args:
            func: Function to execute
            timeout: Timeout in seconds
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            ExecutionResult with success status and result
        """
        start_time = time.time()

        try:
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
            else:
                # Run sync function in thread pool with timeout
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(self.thread_pool, func, *args, **kwargs),
                    timeout=timeout
                )

            execution_time = time.time() - start_time
            return ExecutionResult(
                success=True,
                result=result,
                execution_time=execution_time
            )

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                result=None,
                execution_time=execution_time,
                error=f"Execution timed out after {timeout}s"
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                result=None,
                execution_time=execution_time,
                error=str(e)
            )

    async def batch_execute(
        self,
        func: Callable,
        items: List[Any],
        batch_size: int = 10,
        timeout: Optional[float] = None
    ) -> List[ExecutionResult]:
        """
        Execute function on items in batches.

        Args:
            func: Function to execute
            items: List of items to process
            batch_size: Number of items to process concurrently
            timeout: Timeout per execution

        Returns:
            List of ExecutionResult objects
        """
        results = []

        # Process items in batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]

            # Create tasks for batch
            tasks = []
            for item in batch:
                if timeout:
                    task = self.execute_with_timeout(func, timeout, item)
                else:
                    task = self.execute_single_item(func, item)
                tasks.append(task)

            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process batch results
            for result in batch_results:
                if isinstance(result, Exception):
                    results.append(ExecutionResult(
                        success=False,
                        result=None,
                        execution_time=0.0,
                        error=str(result)
                    ))
                else:
                    results.append(result)

        return results

    async def execute_single_item(self, func: Callable, item: Any) -> ExecutionResult:
        """Execute function on a single item."""
        start_time = time.time()

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(item)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self.thread_pool, func, item)

            execution_time = time.time() - start_time
            return ExecutionResult(
                success=True,
                result=result,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                result=None,
                execution_time=execution_time,
                error=str(e)
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "max_workers": self.max_workers,
            "max_concurrent": self.max_concurrent,
            "active_tasks": len(self.active_tasks),
            "semaphore_value": self.semaphore._value,
            "thread_pool_active": len(self.thread_pool._threads)
        }

    def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        # Cancel any remaining tasks
        for task in self.active_tasks.values():
            if not task.done():
                task.cancel()

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=wait)

        self.logger.info("ParallelExecutor shutdown complete")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.shutdown()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()

# Enhanced FSM Agent with parallel tool execution
class ParallelFSMReactAgent:
    """FSM React Agent with parallel tool execution capabilities"""

    def __init__(self, tools: List[Any], max_parallel_tools: int = 5):
        self.tools = tools
        self.parallel_executor = ParallelExecutor(max_workers=max_parallel_tools)
        self.logger = logging.getLogger(__name__)

    async def execute_tools_parallel(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Execute multiple tool calls in parallel

        Args:
            tool_calls: List of dicts with 'tool_name' and 'arguments'

        Returns:
            List of results
        """

        # Group tools and inputs
        tools = []
        inputs = []

        for call in tool_calls:
            tool_name = call['tool_name']
            arguments = call.get('arguments', {})

            # Find tool by name
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                self.logger.warning("Tool {} not found", extra={"tool_name": tool_name})
                continue

            tools.append(tool.func)
            inputs.append(arguments)

        if not tools:
            return []

        # Execute in parallel
        results = await self.parallel_executor.execute_tools_parallel(
            tools, inputs, timeout=30.0
        )

        # Format results
        formatted_results = []
        for i, (success, result) in enumerate(results):
            formatted_results.append({
                "tool_name": tool_calls[i]['tool_name'],
                "success": success,
                "result": result if success else None,
                "error": result if not success else None
            })

        return formatted_results

# Example usage
async def example_parallel_execution():
    """Example of parallel tool execution"""

    # Create parallel executor
    executor = ParallelExecutor(max_workers=10)

    # Define some mock tools
    async def web_search(query: str) -> str:
        await asyncio.sleep(1)  # Simulate API call
        return f"Search results for: {query}"

    async def calculate(expression: str) -> float:
        await asyncio.sleep(0.5)  # Simulate calculation
        return eval(expression)  # Note: unsafe in production

    async def analyze_text(text: str) -> Dict[str, Any]:
        await asyncio.sleep(2)  # Simulate analysis
        return {"length": len(text), "words": len(text.split())}

    # Execute tools in parallel
    tools = [web_search, calculate, analyze_text]
    inputs = [
        {"query": "parallel execution python"},
        {"expression": "2 + 2 * 3"},
        {"text": "This is a sample text for analysis"}
    ]

    results = await executor.execute_tools_parallel(tools, inputs)

    for (success, result) in results:
        if success:
            logger.info("Result: {}", extra={"result": result})
        else:
            logger.info("Error: {}", extra={"result": result})

    # Map-reduce example
    async def process_item(item: int) -> int:
        await asyncio.sleep(0.1)
        return item * item

    def sum_results(self, results: List[int]) -> int:
        return sum(results)

    items = list(range(100))
    final_result = await executor.map_reduce(
        process_item, sum_results, items
    )
    logger.info("Sum of squares: {}", extra={"final_result": final_result})

    # Cleanup
    executor.shutdown()