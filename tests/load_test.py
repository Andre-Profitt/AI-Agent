from agent import query
from agent import response
from examples.enhanced_unified_example import task
from performance_dashboard import response_time
from performance_dashboard import stats
from tests.load_test import calculations
from tests.load_test import credentials
from tests.load_test import data
from tests.load_test import headers
from tests.load_test import queries
from tests.load_test import registration
from tests.load_test import simple_queries
from tests.load_test import success
from tests.load_test import token_response
from tests.load_test import tool_queries

from src.api.auth import auth_service
from src.gaia_components.production_vector_store import environment
from src.infrastructure.events.event_bus import events
from src.tools_introspection import name

from src.tools.base_tool import Tool
# TODO: Fix undefined variables: Any, Dict, HttpUser, auth_service, between, calculations, credentials, data, e, environment, events, exception, headers, name, os, queries, query, random, registration, response, response_time, simple_queries, stats, success, sys, task, token_response, tool_queries

"""
from typing import Dict
from src.api.auth import UserCredentials
from src.api.auth import UserRegistration
from src.api.auth import UserRole
# TODO: Fix undefined variables: HttpUser, argparse, auth_service, between, calculations, credentials, data, e, environment, events, exception, headers, name, queries, query, registration, response, response_time, self, simple_queries, stats, success, task, token_response, tool_queries

Load Testing Script for GAIA API
Uses Locust for comprehensive load testing
"""

from typing import Any

import os
import sys

import random

from locust import HttpUser, task, between, events

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api.auth import auth_service, UserCredentials

class GAIAUser(HttpUser):
    """Load testing user for GAIA API"""
    wait_time = between(1, 3)

    def on_start(self):
        """Initialize user session"""
        self.user_id = None
        self.access_token = None
        self.session_data = {}

        # Login to get access token
        try:
            self._login()
        except Exception as e:
            self.environment.runner.quit()
            raise e

    def _login(self):
        """Login and get access token"""
        # Use test credentials
        credentials = UserCredentials(
            username="loadtest_user",
            password="loadtest_pass"
        )

        try:
            # Try to login with test user
            token_response = auth_service.login_user(credentials)
            self.access_token = token_response.access_token
            self.user_id = token_response.user_id
        except ValueError:
            # Create test user if it doesn't exist
            from api.auth import UserRegistration, UserRole
            registration = UserRegistration(
                username="loadtest_user",
                email="loadtest@example.com",
                password="loadtest_pass",
                role=UserRole.PREMIUM
            )
            user = auth_service.register_user(registration)

            # Login with new user
            token_response = auth_service.login_user(credentials)
            self.access_token = token_response.access_token
            self.user_id = token_response.user_id

    @task(3)
    def simple_query(self):
        """Simple query task (most common)"""
        headers = {"Authorization": f"Bearer {self.access_token}"}

        queries = [
            "What is 2+2?",
            "What is the capital of France?",
            "How many planets are in our solar system?",
            "What is the largest ocean on Earth?",
            "Who wrote Romeo and Juliet?"
        ]

        query = random.choice(queries)

        with self.client.post(
            "/api/v1/query",
            json={"query": query},
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    response.success()
                else:
                    response.failure(f"Query failed: {data.get('error')}")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def complex_query(self):
        """Complex query task"""
        headers = {"Authorization": f"Bearer {self.access_token}"}

        queries = [
            "Analyze the impact of artificial intelligence on modern society",
            "Compare and contrast renewable energy sources",
            "Explain the principles of quantum computing",
            "What are the economic implications of climate change?",
            "How does machine learning work in practice?"
        ]

        query = random.choice(queries)

        with self.client.post(
            "/api/v1/query",
            json={"query": query, "verification_level": "thorough"},
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    response.success()
                else:
                    response.failure(f"Complex query failed: {data.get('error')}")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def calculation_query(self):
        """Mathematical calculation task"""
        headers = {"Authorization": f"Bearer {self.access_token}"}

        calculations = [
            "Calculate 15 * 23 + 7",
            "What is the square root of 144?",
            "Solve: 2x + 5 = 13",
            "Calculate the area of a circle with radius 5",
            "What is 3 to the power of 4?"
        ]

        query = random.choice(calculations)

        with self.client.post(
            "/api/v1/query",
            json={"query": query, "answer_format": "numeric"},
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    response.success()
                else:
                    response.failure(f"Calculation failed: {data.get('error')}")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def tool_execution_query(self):
        """Tool execution task"""
        headers = {"Authorization": f"Bearer {self.access_token}"}

        tool_queries = [
            "Search for the latest news about AI",
            "Get the current weather in New York",
            "Calculate the distance between Paris and London",
            "Find information about Python programming",
            "Analyze this data: [1, 2, 3, 4, 5]"
        ]

        query = random.choice(tool_queries)

        with self.client.post(
            "/api/v1/query",
            json={"query": query, "use_tools": True},
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    response.success()
                else:
                    response.failure(f"Tool execution failed: {data.get('error')}")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def health_check(self):
        """Health check endpoint"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: HTTP {response.status_code}")

    @task(1)
    def metrics_endpoint(self):
        """Metrics endpoint (for monitoring)"""
        headers = {"Authorization": f"Bearer {self.access_token}"}

        with self.client.get("/metrics", headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Metrics failed: HTTP {response.status_code}")

class AdminUser(HttpUser):
    """Admin user for administrative tasks"""
    wait_time = between(5, 10)
    weight = 1  # Lower weight for admin users

    def on_start(self):
        """Initialize admin session"""
        self.access_token = None
        self._login()

    def _login(self):
        """Login as admin"""
        credentials = UserCredentials(
            username="admin",
            password="admin123"
        )

        try:
            token_response = auth_service.login_user(credentials)
            self.access_token = token_response.access_token
        except Exception as e:
            self.environment.runner.quit()
            raise e

    @task(1)
    def view_analytics(self):
        """View analytics dashboard"""
        headers = {"Authorization": f"Bearer {self.access_token}"}

        with self.client.get("/api/v1/admin/analytics", headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Analytics failed: HTTP {response.status_code}")

    @task(1)
    def system_health(self):
        """Check system health"""
        headers = {"Authorization": f"Bearer {self.access_token}"}

        with self.client.get("/api/v1/admin/health", headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"System health failed: HTTP {response.status_code}")

class StressTestUser(HttpUser):
    """User for stress testing"""
    wait_time = between(0.1, 0.5)  # Very fast requests
    weight = 2

    def on_start(self):
        """Initialize stress test user"""
        self.access_token = None
        self._login()

    def _login(self):
        """Login for stress testing"""
        credentials = UserCredentials(
            username="stresstest_user",
            password="stresstest_pass"
        )

        try:
            token_response = auth_service.login_user(credentials)
            self.access_token = token_response.access_token
        except ValueError:
            # Create stress test user
            registration = UserRegistration(
                username="stresstest_user",
                email="stresstest@example.com",
                password="stresstest_pass",
                role=UserRole.USER
            )
            user = auth_service.register_user(registration)

            token_response = auth_service.login_user(credentials)
            self.access_token = token_response.access_token

    @task(1)
    def rapid_queries(self):
        """Send rapid queries for stress testing"""
        headers = {"Authorization": f"Bearer {self.access_token}"}

        simple_queries = [
            "Hello",
            "Test",
            "Quick",
            "Fast",
            "Simple"
        ]

        query = random.choice(simple_queries)

        with self.client.post(
            "/api/v1/query",
            json={"query": query},
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Rapid query failed: HTTP {response.status_code}")

# Custom event handlers for monitoring
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts"""
    print("🚀 GAIA Load Test Starting...")
    print(f"Target: {environment.host}")
    print(f"Users: {environment.runner.user_count}")
    print(f"Spawn Rate: {environment.runner.spawn_rate}")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops"""
    print("✅ GAIA Load Test Completed!")

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, start_time, url, **kwargs):
    """Called for every request"""
    if exception:
        print(f"❌ Request failed: {name} - {exception}")
    elif response.status_code >= 400:
        print(f"⚠️  Request error: {name} - HTTP {response.status_code}")

# Performance thresholds
@events.request_failure.add_listener
def on_request_failure(request_type, name, response_time, response_length, exception, **kwargs):
    """Called when request fails"""
    if response_time > 5000:  # 5 seconds
        print(f"🐌 Slow request: {name} took {response_time}ms")

# Custom metrics collection
class CustomMetrics:
    """Custom metrics collection"""

    def __init__(self):
        self.success_count = 0
        self.failure_count = 0
        self.total_response_time = 0
        self.request_count = 0

    def record_request(self, success: bool, response_time: int):
        """Record request metrics"""
        self.request_count += 1
        self.total_response_time += response_time

        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        if self.request_count == 0:
            return {
                "success_rate": 0.0,
                "avg_response_time": 0.0,
                "total_requests": 0
            }

        return {
            "success_rate": self.success_count / self.request_count,
            "avg_response_time": self.total_response_time / self.request_count,
            "total_requests": self.request_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count
        }

# Global metrics instance
custom_metrics = CustomMetrics()

@events.request.add_listener
def record_metrics(request_type, name, response_time, response_length, response, context, exception, start_time, url, **kwargs):
    """Record custom metrics"""
    success = exception is None and response.status_code < 400
    custom_metrics.record_request(success, response_time)

@events.test_stop.add_listener
def print_final_stats(environment, **kwargs):
    """Print final statistics"""
    stats = custom_metrics.get_stats()
    print("\n📊 Final Statistics:")
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Success Rate: {stats['success_rate']:.2%}")
    print(f"Average Response Time: {stats['avg_response_time']:.2f}ms")
    print(f"Success Count: {stats['success_count']}")
    print(f"Failure Count: {stats['failure_count']}")

# Command line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GAIA Load Testing")
    parser.add_argument("--host", default="http://localhost:8080", help="Target host")
    parser.add_argument("--users", type=int, default=10, help="Number of users")
    parser.add_argument("--spawn-rate", type=int, default=2, help="Users per second")
    parser.add_argument("--run-time", default="60s", help="Test duration")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")

    args = parser.parse_args()

    # Set environment variables for locust
    os.environ["LOCUST_HOST"] = args.host

    # Build locust command
    cmd = [
        "locust",
        "-f", __file__,
        "--host", args.host,
        "--users", str(args.users),
        "--spawn-rate", str(args.spawn_rate),
        "--run-time", args.run_time
    ]

    if args.headless:
        cmd.append("--headless")

    # Run locust
    os.execvp("locust", cmd)