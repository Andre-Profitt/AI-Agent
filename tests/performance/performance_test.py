# performance_test.py
import asyncio
import aiohttp
import time
import statistics
from typing import List, Dict

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
                    print(f"Task submission failed: {await response.text()}")
        
        return latencies
    
    async def run_load_test(self, agent_count: int, task_count: int, 
                          concurrent_tasks: int = 10):
        '''Run a load test'''
        print(f"Starting load test with {agent_count} agents and {task_count} tasks")
        
        # Register agents
        print("Registering agents...")
        agent_ids = await self.register_test_agents(agent_count)
        print(f"Registered {len(agent_ids)} agents")
        
        # Submit tasks with concurrency control
        print(f"Submitting {task_count} tasks...")
        
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
        
        print(f"\nPerformance Results:")
        print(f"Total requests: {len(all_latencies)}")
        print(f"Average latency: {avg_latency:.3f}s")
        print(f"P50 latency: {p50_latency:.3f}s")
        print(f"P95 latency: {p95_latency:.3f}s")
        print(f"P99 latency: {p99_latency:.3f}s")
        
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
        print(f"\n{'='*50}")
        results = await tester.run_load_test(agent_count, task_count, concurrent)
        print(f"{'='*50}\n")

if __name__ == "__main__":
    asyncio.run(main()) 