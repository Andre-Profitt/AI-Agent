#!/usr/bin/env python3
from src.agents.advanced_agent_fsm import Agent
# TODO: Fix undefined variables: Any, Dict, List, Optional, agent_data, args, auth_token, base_url, client_id, code, data, dataclass, duration, e, endpoint, endpoint_results, endpoints, f, i, json, logging, mean, median, message_count, method, num_agents, num_connections, num_requests, num_tasks, parser, r, random, report, response, response_time, response_times, result, results, rt_stats, session, start_time, stats, status_codes, stdev, summary, task, task_data, tasks, tester, time, url, ws
# TODO: Fix undefined variables: agent_data, aiohttp, argparse, args, auth_token, base_url, client_id, code, data, duration, e, endpoint, endpoint_results, endpoints, f, i, mean, median, message_count, method, num_agents, num_connections, num_requests, num_tasks, parser, r, report, response, response_time, response_times, result, results, rt_stats, self, session, start_time, stats, status_codes, stdev, summary, task, task_data, tasks, tester, url, ws

"""
Performance Testing Script for Multi-Agent Platform API
Simulates load testing for various endpoints and WebSocket connections
"""

from typing import Optional
from typing import Any
from typing import List

import asyncio
import aiohttp
import json
import time
import random

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from statistics import mean, median, stdev
import argparse
import logging

logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    endpoint: str
    method: str
    status_code: int
    response_time: float
    timestamp: float

class PerformanceTester:
    def __init__(self, base_url: str, auth_token: str):
        self.base_url = base_url.rstrip('/')
        self.auth_token = auth_token
        self.results: List[TestResult] = []
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={'Authorization': f'Bearer {self.auth_token}'}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def make_request(self, method: str, endpoint: str, data: Dict = None) -> TestResult:
        """Make a single request and record metrics"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()

        try:
            if method.upper() == 'GET':
                async with self.session.get(url) as response:
                    response_time = time.time() - start_time
                    return TestResult(
                        endpoint=endpoint,
                        method=method,
                        status_code=response.status,
                        response_time=response_time,
                        timestamp=start_time
                    )
            elif method.upper() == 'POST':
                async with self.session.post(url, json=data) as response:
                    response_time = time.time() - start_time
                    return TestResult(
                        endpoint=endpoint,
                        method=method,
                        status_code=response.status,
                        response_time=response_time,
                        timestamp=start_time
                    )
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time=response_time,
                timestamp=start_time
            )

    async def test_agent_registration(self, num_agents: int = 10) -> List[TestResult]:
        """Test agent registration endpoint"""
        logger.info("Testing agent registration with {} agents...", extra={"num_agents": num_agents})

        tasks = []
        for i in range(num_agents):
            agent_data = {
                "name": f"test_agent_{i}",
                "capabilities": ["task_execution", "data_processing"],
                "resources": {"cpu": 1, "memory": "512Mi"},
                "metadata": {"test": True, "id": i}
            }
            task = self.make_request('POST', '/api/v1/agents', agent_data)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        self.results.extend(results)
        return results

    async def test_task_submission(self, num_tasks: int = 20) -> List[TestResult]:
        """Test task submission endpoint"""
        logger.info("Testing task submission with {} tasks...", extra={"num_tasks": num_tasks})

        tasks = []
        for i in range(num_tasks):
            task_data = {
                "title": f"Test Task {i}",
                "description": f"This is test task number {i}",
                "priority": random.randint(1, 10),
                "required_capabilities": ["task_execution"],
                "resources": {"cpu": 0.5, "memory": "256Mi"},
                "metadata": {"test": True, "id": i}
            }
            task = self.make_request('POST', '/api/v1/tasks', task_data)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        self.results.extend(results)
        return results

    async def test_list_endpoints(self, num_requests: int = 50) -> List[TestResult]:
        """Test list endpoints (agents, tasks, resources, conflicts)"""
        logger.info("Testing list endpoints with {} requests...", extra={"num_requests": num_requests})

        endpoints = [
            '/api/v1/agents',
            '/api/v1/tasks',
            '/api/v1/resources',
            '/api/v1/conflicts'
        ]

        tasks = []
        for _ in range(num_requests):
            endpoint = random.choice(endpoints)
            task = self.make_request('GET', endpoint)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        self.results.extend(results)
        return results

    async def test_health_endpoint(self, num_requests: int = 100) -> List[TestResult]:
        """Test health endpoint"""
        logger.info("Testing health endpoint with {} requests...", extra={"num_requests": num_requests})

        tasks = []
        for _ in range(num_requests):
            task = self.make_request('GET', '/api/v1/health')
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        self.results.extend(results)
        return results

    async def test_dashboard_endpoints(self, num_requests: int = 30) -> List[TestResult]:
        """Test dashboard endpoints"""
        logger.info("Testing dashboard endpoints with {} requests...", extra={"num_requests": num_requests})

        endpoints = [
            '/api/v1/dashboard/summary',
            '/api/v1/dashboard/activity'
        ]

        tasks = []
        for _ in range(num_requests):
            endpoint = random.choice(endpoints)
            task = self.make_request('GET', endpoint)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        self.results.extend(results)
        return results

    async def test_websocket_connections(self, num_connections: int = 10, duration: int = 30):
        """Test WebSocket connections"""
        logger.info("Testing WebSocket connections with {} connections for {}s...", extra={"num_connections": num_connections, "duration": duration})

        async def websocket_client(client_id: str):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        f"{self.base_url.replace('http', 'ws')}/api/v1/ws/{client_id}"
                    ) as ws:
                        # Send subscription message
                        await ws.send_json({
                            "type": "subscribe",
                            "subscriptions": ["agent_activity", "task_activity"]
                        })

                        # Keep connection alive and receive messages
                        start_time = time.time()
                        message_count = 0

                        while time.time() - start_time < duration:
                            try:
                                msg = await asyncio.wait_for(ws.receive_json(), timeout=1.0)
                                message_count += 1
                            except asyncio.TimeoutError:
                                continue

                        logger.info("Client {}: Received {} messages", extra={"client_id": client_id, "message_count": message_count})

            except Exception as e:
                logger.info("WebSocket client {} error: {}", extra={"client_id": client_id, "e": e})

        # Create multiple WebSocket connections
        tasks = []
        for i in range(num_connections):
            client_id = f"test_client_{i}"
            task = websocket_client(client_id)
            tasks.append(task)

        await asyncio.gather(*tasks)

    def generate_report(self) -> Dict[str, Any]:
        """Generate performance test report"""
        if not self.results:
            return {"error": "No test results available"}

        # Group results by endpoint
        endpoint_results = {}
        for result in self.results:
            if result.endpoint not in endpoint_results:
                endpoint_results[result.endpoint] = []
            endpoint_results[result.endpoint].append(result)

        # Calculate statistics for each endpoint
        report = {
            "summary": {
                "total_requests": len(self.results),
                "successful_requests": len([r for r in self.results if r.status_code == 200]),
                "failed_requests": len([r for r in self.results if r.status_code != 200]),
                "total_duration": max(r.timestamp for r in self.results) - min(r.timestamp for r in self.results)
            },
            "endpoints": {}
        }

        for endpoint, results in endpoint_results.items():
            response_times = [r.response_time for r in results]
            status_codes = [r.status_code for r in results]

            report["endpoints"][endpoint] = {
                "total_requests": len(results),
                "successful_requests": len([r for r in results if r.status_code == 200]),
                "failed_requests": len([r for r in results if r.status_code != 200]),
                "response_time_stats": {
                    "mean": mean(response_times),
                    "median": median(response_times),
                    "min": min(response_times),
                    "max": max(response_times),
                    "std_dev": stdev(response_times) if len(response_times) > 1 else 0
                },
                "status_code_distribution": {
                    str(code): status_codes.count(code) for code in set(status_codes)
                }
            }

        return report

    def print_report(self, report: Dict[str, Any]):
        """Print formatted performance report"""
        print("\n" + "="*60)
        logger.info("PERFORMANCE TEST REPORT")
        print("="*60)

        summary = report["summary"]
        logger.info("\nSUMMARY:")
        logger.info("  Total Requests: {}", extra={"summary__total_requests_": summary['total_requests']})
        logger.info("  Successful: {}", extra={"summary__successful_requests_": summary['successful_requests']})
        logger.info("  Failed: {}", extra={"summary__failed_requests_": summary['failed_requests']})
        logger.info("  Success Rate: {}%", extra={"summary__successful_requests__summary__total_requests__100": summary['successful_requests']/summary['total_requests']*100})
        logger.info("  Total Duration: {}s", extra={"summary__total_duration_": summary['total_duration']})

        logger.info("\nENDPOINT DETAILS:")
        for endpoint, stats in report["endpoints"].items():
            logger.info("\n  {}:", extra={"endpoint": endpoint})
            logger.info("    Requests: {}", extra={"stats__total_requests_": stats['total_requests']})
            logger.info("    Success Rate: {}%", extra={"stats__successful_requests__stats__total_requests__100": stats['successful_requests']/stats['total_requests']*100})

            rt_stats = stats['response_time_stats']
            logger.info("    Response Time (s):")
            logger.info("      Mean: {}", extra={"rt_stats__mean_": rt_stats['mean']})
            logger.info("      Median: {}", extra={"rt_stats__median_": rt_stats['median']})
            logger.info("      Min: {}", extra={"rt_stats__min_": rt_stats['min']})
            logger.info("      Max: {}", extra={"rt_stats__max_": rt_stats['max']})
            logger.info("      Std Dev: {}", extra={"rt_stats__std_dev_": rt_stats['std_dev']})

            logger.info("    Status Codes: {}", extra={"stats__status_code_distribution_": stats['status_code_distribution']})

async def main():
    parser = argparse.ArgumentParser(description='Performance test for Multi-Agent Platform API')
    parser.add_argument('--base-url', default='http://localhost:8000', help='Base URL of the API')
    parser.add_argument('--auth-token', default='test-token', help='Authentication token')
    parser.add_argument('--agents', type=int, default=10, help='Number of agents to register')
    parser.add_argument('--tasks', type=int, default=20, help='Number of tasks to submit')
    parser.add_argument('--list-requests', type=int, default=50, help='Number of list requests')
    parser.add_argument('--health-requests', type=int, default=100, help='Number of health requests')
    parser.add_argument('--dashboard-requests', type=int, default=30, help='Number of dashboard requests')
    parser.add_argument('--websocket-connections', type=int, default=10, help='Number of WebSocket connections')
    parser.add_argument('--websocket-duration', type=int, default=30, help='WebSocket test duration in seconds')
    parser.add_argument('--output', help='Output file for JSON report')

    args = parser.parse_args()

    logger.info("Starting Performance Test...")
    logger.info("Target URL: {}", extra={"args_base_url": args.base_url})
    logger.info("Test Configuration:")
    logger.info("  Agents: {}", extra={"args_agents": args.agents})
    logger.info("  Tasks: {}", extra={"args_tasks": args.tasks})
    logger.info("  List Requests: {}", extra={"args_list_requests": args.list_requests})
    logger.info("  Health Requests: {}", extra={"args_health_requests": args.health_requests})
    logger.info("  Dashboard Requests: {}", extra={"args_dashboard_requests": args.dashboard_requests})
    logger.info("  WebSocket Connections: {}", extra={"args_websocket_connections": args.websocket_connections})
    logger.info("  WebSocket Duration: {}s", extra={"args_websocket_duration": args.websocket_duration})

    async with PerformanceTester(args.base_url, args.auth_token) as tester:
        # Run all tests
        await tester.test_health_endpoint(args.health_requests)
        await tester.test_agent_registration(args.agents)
        await tester.test_task_submission(args.tasks)
        await tester.test_list_endpoints(args.list_requests)
        await tester.test_dashboard_endpoints(args.dashboard_requests)

        # Test WebSocket connections
        await tester.test_websocket_connections(args.websocket_connections, args.websocket_duration)

        # Generate and print report
        report = tester.generate_report()
        tester.print_report(report)

        # Save report to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info("\nReport saved to: {}", extra={"args_output": args.output})

if __name__ == "__main__":
    asyncio.run(main())