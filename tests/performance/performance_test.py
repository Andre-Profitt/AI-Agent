# performance_test.py
from agent import response
from benchmarks.cot_performance import semaphore
from examples.enhanced_unified_example import start_time
from examples.enhanced_unified_example import task
from examples.enhanced_unified_example import tasks
from examples.parallel_execution_example import results
from tests.load_test import data
from tests.performance.performance_test import agent_data
from tests.performance.performance_test import all_latencies
from tests.performance.performance_test import latencies
from tests.performance.performance_test import p50_latency
from tests.performance.performance_test import p95_latency
from tests.performance.performance_test import p99_latency
from tests.performance.performance_test import responses
from tests.performance.performance_test import scenarios
from tests.performance.performance_test import task_data
from tests.performance.performance_test import tester

from src.api_server import token
from src.collaboration.realtime_collaboration import session
from src.database_extended import avg_latency
from src.gaia_components.production_vector_store import count
from src.services.integration_hub import latency
from src.unified_architecture.registry import agent_ids

import asyncio
import aiohttp
import time
import statistics

import logging

import concurrent
from typing import List

from src.agents.advanced_agent_fsm import Agent
# TODO: Fix undefined variables: List, agent_count, agent_data, agent_ids, all_latencies, avg_latency, base_url, concurrent, concurrent_tasks, count, data, i, lat, latencies, latency, logging, p50_latency, p95_latency, p99_latency, response, result, results, scenarios, semaphore, session, start_time, task, task_count, task_data, tasks, tester, time, token
import responses

# TODO: Fix undefined variables: agent_count, agent_data, agent_ids, aiohttp, all_latencies, avg_latency, base_url, concurrent, concurrent_tasks, count, data, i, lat, latencies, latency, p50_latency, p95_latency, p99_latency, response, responses, result, results, scenarios, self, semaphore, session, start_time, statistics, task, task_count, task_data, tasks, tester, token

logger = logging.getLogger(__name__)

class PerformanceTester:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}

    async def register_test_agents(self, count: int) -> List[str]:
        '''Register multiple test agents'''
        agent_ids = []

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(count):
                agent_data = {
                    "name": f"TestAgent_{i}",
                    "version": "1.0.0",
                    "capabilities": ["REASONING", "COLLABORATION"],
                    "tags": ["test", "performance"],
                    "resources": {"cpu_cores": 0.5, "memory_mb": 256}
                }

                task = session.post(
                    f"{self.base_url}/api/v2/agents/register",
                    json=agent_data,
                    headers=self.headers
                )
                tasks.append(task)

            responses = await asyncio.gather(*tasks)

            for response in responses:
                data = await response.json()
                agent_ids.append(data["agent_id"])

        return agent_ids

    async def submit_test_tasks(self, count: int) -> List[float]:
        '''Submit multiple tasks and measure latency'''
        latencies = []

        async with aiohttp.ClientSession() as session:
            for i in range(count):
                task_data = {
                    "task_type": "test_task",
                    "priority": 5,
                    "payload": {"test_id": i},
                    "required_capabilities": ["REASONING"]
                }

                start_time = time.time()

                response = await session.post(
                    f"{self.base_url}/api/v2/tasks/submit",
                    json=task_data,
                    headers=self.headers
                )

                latency = time.time() - start_time
                latencies.append(latency)

                if response.status != 200:
                    logger.info("Task submission failed: {}", extra={"await_response_text__": await response.text()})

        return latencies

    async def run_load_test(self, agent_count: int, task_count: int,
                          concurrent_tasks: int = 10):
        '''Run a load test'''
        logger.info("Starting load test with {} agents and {} tasks", extra={"agent_count": agent_count, "task_count": task_count})

        # Register agents
        logger.info("Registering agents...")
        agent_ids = await self.register_test_agents(agent_count)
        logger.info("Registered {} agents", extra={"len_agent_ids_": len(agent_ids)})

        # Submit tasks with concurrency control
        logger.info("Submitting {} tasks...", extra={"task_count": task_count})

        semaphore = asyncio.Semaphore(concurrent_tasks)

        async def submit_with_limit():
            async with semaphore:
                return await self.submit_test_tasks(1)

        tasks = [submit_with_limit() for _ in range(task_count)]
        results = await asyncio.gather(*tasks)

        # Flatten results
        all_latencies = [lat for result in results for lat in result]

        # Calculate statistics
        avg_latency = statistics.mean(all_latencies)
        p50_latency = statistics.median(all_latencies)
        p95_latency = statistics.quantiles(all_latencies, n=20)[18]  # 95th percentile
        p99_latency = statistics.quantiles(all_latencies, n=100)[98]  # 99th percentile

        logger.info("\nPerformance Results:")
        logger.info("Total requests: {}", extra={"len_all_latencies_": len(all_latencies)})
        logger.info("Average latency: {}s", extra={"avg_latency": avg_latency})
        logger.info("P50 latency: {}s", extra={"p50_latency": p50_latency})
        logger.info("P95 latency: {}s", extra={"p95_latency": p95_latency})
        logger.info("P99 latency: {}s", extra={"p99_latency": p99_latency})

        return {
            "total_requests": len(all_latencies),
            "avg_latency": avg_latency,
            "p50_latency": p50_latency,
            "p95_latency": p95_latency,
            "p99_latency": p99_latency
        }

async def main():
    tester = PerformanceTester("http://localhost:8080", "valid_token")

    # Run different load scenarios
    scenarios = [
        (10, 100, 10),    # 10 agents, 100 tasks, 10 concurrent
        (50, 500, 20),    # 50 agents, 500 tasks, 20 concurrent
        (100, 1000, 50),  # 100 agents, 1000 tasks, 50 concurrent
    ]

    for agent_count, task_count, concurrent in scenarios:
        logger.info("\n{}", extra={"____50": '='*50})
        results = await tester.run_load_test(agent_count, task_count, concurrent)
        logger.info("{}\n", extra={"____50": '='*50})

if __name__ == "__main__":
    asyncio.run(main())