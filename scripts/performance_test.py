#!/usr/bin/env python3
"""
Performance Testing Script for Multi-Agent Platform API
Simulates load testing for various endpoints and WebSocket connections
"""

import asyncio
import aiohttp
import json
import time
import random
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from statistics import mean, median, stdev
import argparse

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
        print(f"Testing agent registration with {num_agents} agents...")
        
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
        print(f"Testing task submission with {num_tasks} tasks...")
        
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
        print(f"Testing list endpoints with {num_requests} requests...")
        
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
        print(f"Testing health endpoint with {num_requests} requests...")
        
        tasks = []
        for _ in range(num_requests):
            task = self.make_request('GET', '/api/v1/health')
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        self.results.extend(results)
        return results
    
    async def test_dashboard_endpoints(self, num_requests: int = 30) -> List[TestResult]:
        """Test dashboard endpoints"""
        print(f"Testing dashboard endpoints with {num_requests} requests...")
        
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
        print(f"Testing WebSocket connections with {num_connections} connections for {duration}s...")
        
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
                        
                        print(f"Client {client_id}: Received {message_count} messages")
                        
            except Exception as e:
                print(f"WebSocket client {client_id} error: {e}")
        
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
        print("PERFORMANCE TEST REPORT")
        print("="*60)
        
        summary = report["summary"]
        print(f"\nSUMMARY:")
        print(f"  Total Requests: {summary['total_requests']}")
        print(f"  Successful: {summary['successful_requests']}")
        print(f"  Failed: {summary['failed_requests']}")
        print(f"  Success Rate: {summary['successful_requests']/summary['total_requests']*100:.2f}%")
        print(f"  Total Duration: {summary['total_duration']:.2f}s")
        
        print(f"\nENDPOINT DETAILS:")
        for endpoint, stats in report["endpoints"].items():
            print(f"\n  {endpoint}:")
            print(f"    Requests: {stats['total_requests']}")
            print(f"    Success Rate: {stats['successful_requests']/stats['total_requests']*100:.2f}%")
            
            rt_stats = stats['response_time_stats']
            print(f"    Response Time (s):")
            print(f"      Mean: {rt_stats['mean']:.3f}")
            print(f"      Median: {rt_stats['median']:.3f}")
            print(f"      Min: {rt_stats['min']:.3f}")
            print(f"      Max: {rt_stats['max']:.3f}")
            print(f"      Std Dev: {rt_stats['std_dev']:.3f}")
            
            print(f"    Status Codes: {stats['status_code_distribution']}")

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
    
    print("Starting Performance Test...")
    print(f"Target URL: {args.base_url}")
    print(f"Test Configuration:")
    print(f"  Agents: {args.agents}")
    print(f"  Tasks: {args.tasks}")
    print(f"  List Requests: {args.list_requests}")
    print(f"  Health Requests: {args.health_requests}")
    print(f"  Dashboard Requests: {args.dashboard_requests}")
    print(f"  WebSocket Connections: {args.websocket_connections}")
    print(f"  WebSocket Duration: {args.websocket_duration}s")
    
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
            print(f"\nReport saved to: {args.output}")

if __name__ == "__main__":
    asyncio.run(main()) 