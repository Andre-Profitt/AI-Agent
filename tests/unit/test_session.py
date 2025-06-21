from benchmarks.cot_performance import start
from examples.enhanced_unified_example import metrics
from performance_dashboard import stats

from src.api.rate_limiter import limiter
from src.api_server import manager
from src.collaboration.realtime_collaboration import session_id
from src.tools_introspection import error

from src.agents.advanced_agent_fsm import RateLimiter
from src.infrastructure.session import AsyncResponseCache
from src.infrastructure.session import SessionManager
from src.infrastructure.session import SessionMetrics
# TODO: Fix undefined variables: analytics, context, custom_id, error, i, limiter, manager, metrics, new_session, old_session, result_id, session1, session2, session_id, start, stats, analytics, context, custom_id, error, i, limiter, manager, metrics, new_session, old_session, result_id, session1, session2, session_id, start, stats
# TODO: Fix undefined variables: analytics, context, custom_id, error, i, limiter, manager, metrics, new_session, old_session, result_id, session1, session2, session_id, start, stats, analytics, context, custom_id, error, i, limiter, manager, metrics, new_session, old_session, result_id, session1, session2, session_id, start, stats
"""

from fastapi import status
Unit tests for session management module
"""

import time

from session import SessionManager, SessionMetrics, AsyncResponseCache, RateLimiter
from functools import cache
import time

from fastapi import status

class TestSessionMetrics:
    """Test session metrics functionality"""

    def test_session_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = SessionMetrics(session_id="test-123")

        assert metrics.session_id == "test-123"
        assert metrics.total_queries == 0
        assert metrics.cache_hits == 0
        assert metrics.total_response_time == 0.0
        assert metrics.tool_usage == {}
        assert metrics.errors == []
        assert metrics.parallel_executions == 0

    def test_average_response_time(self):
        """Test average response time calculation"""
        metrics = SessionMetrics(session_id="test-123")

        # No queries yet
        assert metrics.average_response_time == 0.0

        # Add some queries
        metrics.total_queries = 3
        metrics.total_response_time = 6.0
        assert metrics.average_response_time == 2.0

    def test_cache_hit_rate(self):
        """Test cache hit rate calculation"""
        metrics = SessionMetrics(session_id="test-123")

        # No queries yet
        assert metrics.cache_hit_rate == 0.0

        # Add queries and cache hits
        metrics.total_queries = 10
        metrics.cache_hits = 3
        assert metrics.cache_hit_rate == 30.0

    def test_uptime_hours(self):
        """Test uptime calculation"""
        metrics = SessionMetrics(session_id="test-123")

        # Mock created_at to be 1 hour ago
        metrics.created_at = time.time() - 3600
        assert 0.99 < metrics.uptime_hours < 1.01  # Allow small variance

class TestAsyncResponseCache:
    """Test async response cache functionality"""

    def test_cache_initialization(self):
        """Test cache initialization"""
        cache = AsyncResponseCache(max_size=100, ttl_seconds=60)

        assert cache.max_size == 100
        assert cache.ttl_seconds == 60
        assert len(cache.cache) == 0
        assert len(cache.timestamps) == 0

    def test_cache_set_and_get(self):
        """Test cache set and get operations"""
        cache = AsyncResponseCache(max_size=100, ttl_seconds=60)

        # Set a value
        cache.set("key1", "value1")

        # Get the value
        assert cache.get("key1") == "value1"

        # Non-existent key
        assert cache.get("key2") is None

    def test_cache_expiration(self):
        """Test cache TTL expiration"""
        cache = AsyncResponseCache(max_size=100, ttl_seconds=1)  # 1 second TTL

        # Set a value
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_cache_size_limit(self):
        """Test cache size limiting"""
        cache = AsyncResponseCache(max_size=5, ttl_seconds=60)

        # Fill cache beyond limit
        for i in range(10):
            cache.set(f"key{i}", f"value{i}")

        # Should not exceed max size
        assert len(cache.cache) <= 5

    def test_cache_stats(self):
        """Test cache statistics"""
        cache = AsyncResponseCache(max_size=100, ttl_seconds=60)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        stats = cache.get_stats()
        assert stats["size"] == 2
        assert stats["max_size"] == 100
        assert stats["ttl_seconds"] == 60

class TestRateLimiter:
    """Test rate limiter functionality"""

    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization"""
        limiter = RateLimiter(max_requests_per_minute=60)

        assert limiter.max_requests_per_minute == 60
        assert limiter.total_requests == 0
        assert len(limiter.requests) == 0

    def test_rate_limiting_enforcement(self):
        """Test that rate limiting is enforced"""
        limiter = RateLimiter(max_requests_per_minute=2)  # Very low limit for testing

        # First two requests should go through quickly
        start = time.time()
        limiter.wait_if_needed()
        limiter.wait_if_needed()

        # Third request should be delayed
        limiter.wait_if_needed()
        elapsed = time.time() - start

        # Should have some delay due to rate limiting
        assert limiter.total_requests == 3

    def test_rate_limiter_status(self):
        """Test rate limiter status reporting"""
        limiter = RateLimiter(max_requests_per_minute=60)

        limiter.wait_if_needed()
        status = limiter.get_status()

        assert status["max_rpm"] == 60
        assert status["total_requests"] == 1
        assert status["current_rpm"] == 1
        assert "last_request_time" in status

class TestSessionManager:
    """Test session manager functionality"""

    def test_session_creation(self):
        """Test session creation"""
        manager = SessionManager()

        # Create session with auto-generated ID
        session_id = manager.create_session()
        assert session_id in manager.sessions
        assert isinstance(manager.sessions[session_id], SessionMetrics)

        # Create session with specific ID
        custom_id = "custom-123"
        result_id = manager.create_session(custom_id)
        assert result_id == custom_id
        assert custom_id in manager.sessions

    def test_update_query_metrics(self):
        """Test updating query metrics"""
        manager = SessionManager()
        session_id = manager.create_session()

        # Update metrics
        manager.update_query_metrics(session_id, response_time=2.5, cache_hit=True, parallel=True)

        metrics = manager.get_session_metrics(session_id)
        assert metrics.total_queries == 1
        assert metrics.total_response_time == 2.5
        assert metrics.cache_hits == 1
        assert metrics.parallel_executions == 1

    def test_update_tool_usage(self):
        """Test updating tool usage"""
        manager = SessionManager()
        session_id = manager.create_session()

        # Update tool usage
        manager.update_tool_usage(session_id, "web_search")
        manager.update_tool_usage(session_id, "web_search")
        manager.update_tool_usage(session_id, "calculator")

        metrics = manager.get_session_metrics(session_id)
        assert metrics.tool_usage["web_search"] == 2
        assert metrics.tool_usage["calculator"] == 1

    def test_add_error(self):
        """Test error logging"""
        manager = SessionManager()
        session_id = manager.create_session()

        # Add an error
        error = ValueError("Test error")
        context = {"step": "test_step"}
        manager.add_error(session_id, error, context)

        metrics = manager.get_session_metrics(session_id)
        assert len(metrics.errors) == 1
        assert metrics.errors[0]["error_type"] == "ValueError"
        assert metrics.errors[0]["error_message"] == "Test error"
        assert metrics.errors[0]["context"]["step"] == "test_step"

    def test_global_analytics(self):
        """Test global analytics aggregation"""
        manager = SessionManager()

        # Create multiple sessions with activity
        session1 = manager.create_session()
        session2 = manager.create_session()

        manager.update_query_metrics(session1, 1.0, cache_hit=True)
        manager.update_query_metrics(session1, 2.0)
        manager.update_query_metrics(session2, 3.0, parallel=True)

        manager.update_tool_usage(session1, "tool1")
        manager.update_tool_usage(session2, "tool1")
        manager.update_tool_usage(session2, "tool2")

        # Get global analytics
        analytics = manager.get_global_analytics()

        assert analytics["total_sessions"] == 2
        assert analytics["total_queries"] == 3
        assert analytics["avg_response_time"] == 2.0  # (1+2+3)/3
        assert analytics["cache_hit_rate"] == 33.33333333333333  # 1/3
        assert analytics["parallel_executions"] == 1
        assert analytics["tool_usage"]["tool1"] == 2
        assert analytics["tool_usage"]["tool2"] == 1

    def test_cleanup_old_sessions(self):
        """Test cleaning up old sessions"""
        manager = SessionManager()

        # Create sessions
        old_session = manager.create_session()
        new_session = manager.create_session()

        # Make one session old
        manager.sessions[old_session].created_at = time.time() - (25 * 3600)  # 25 hours ago

        # Cleanup sessions older than 24 hours
        manager.cleanup_old_sessions(max_age_hours=24)

        assert old_session not in manager.sessions
        assert new_session in manager.sessions