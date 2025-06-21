from performance_dashboard import metric
from tests.conftest import mock_db

from unittest.mock import Mock
# TODO: Fix undefined variables: Path, REGISTRY, metric, mock_db, os, reset_all_circuit_breakers, sys
from unittest.mock import AsyncMock

# TODO: Fix undefined variables: AsyncMock, REGISTRY, metric, mock_db, reset_all_circuit_breakers
# tests/conftest.py
"""
Enhanced test configuration with fixtures and setup
"""

import pytest

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

@pytest.fixture(autouse=True)
async def reset_circuit_breakers():
    """Reset all circuit breakers between tests"""
    yield
    try:
        from src.infrastructure.resilience.circuit_breaker import reset_all_circuit_breakers
        await reset_all_circuit_breakers()
    except ImportError:
        pass  # Circuit breaker module might not be available

@pytest.fixture(autouse=True)
def clean_metrics():
    """Clean metrics between tests"""
    yield
    try:
        from prometheus_client import REGISTRY
        # Reset Prometheus metrics
        for metric in list(REGISTRY._collector_to_names.keys()):
            REGISTRY.unregister(metric)
    except ImportError:
        pass  # Prometheus might not be available

@pytest.fixture(autouse=True)
def set_test_environment():
    """Set test environment variables"""
    import os
    os.environ['ENVIRONMENT'] = 'test'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    os.environ['TESTING'] = 'true'
    yield
    # Clean up
    os.environ.pop('ENVIRONMENT', None)
    os.environ.pop('LOG_LEVEL', None)
    os.environ.pop('TESTING', None)

@pytest.fixture
def mock_repositories():
    """Create mock repositories for testing"""
    return {
        "agent_repository": AsyncMock(),
        "message_repository": AsyncMock(),
        "tool_repository": AsyncMock(),
        "session_repository": AsyncMock()
    }

@pytest.fixture
def mock_services():
    """Create mock services for testing"""
    return {
        "agent_executor": AsyncMock(),
        "tool_executor": AsyncMock(),
        "logging_service": Mock()
    }

@pytest.fixture
def test_config():
    """Create test configuration"""
    return {
        "max_input_length": 1000,
        "timeout": 30,
        "max_retries": 3,
        "test_mode": True
    }

@pytest.fixture
async def db_transaction():
    """Database transaction fixture for test isolation"""
    # Mock database transaction
    mock_db = AsyncMock()
    await mock_db.begin()
    yield mock_db
    await mock_db.rollback()

# Performance test baselines
PERFORMANCE_BASELINES = {
    "parallel_tool_execution": 1.0,  # Should complete in <1s
    "circuit_breaker_trigger": 0.01,  # Should trigger in <10ms
    "workflow_orchestration": 2.0,  # Complex workflows <2s
    "concurrent_requests": 100,  # Handle 100 concurrent requests
    "api_response_time": 0.5,  # API calls <500ms
    "database_operation": 0.1,  # DB operations <100ms
}


"""
Pytest Configuration
Shared fixtures and test setup
"""

import pytest
import asyncio
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# Set test environment
os.environ["TESTING"] = "true"
os.environ["LOG_LEVEL"] = "ERROR"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests"""
    # Reset any singleton instances
    from src.infrastructure.di.container import Container
    Container._instance = None
    
    from src.tools.registry import ToolRegistry
    ToolRegistry._instance = None


@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    llm = Mock()
    llm.agenerate = AsyncMock(return_value=Mock(
        generations=[[Mock(text="Test response")]]
    ))
    return llm


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for tests"""
    return tmp_path


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    redis = Mock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=True)
    redis.expire = AsyncMock(return_value=True)
    return redis


@pytest.fixture
def mock_database():
    """Mock database connection"""
    db = Mock()
    db.execute = AsyncMock()
    db.fetch_one = AsyncMock()
    db.fetch_all = AsyncMock()
    return db


# Markers
pytest.mark.slow = pytest.mark.slow
pytest.mark.integration = pytest.mark.integration
pytest.mark.unit = pytest.mark.unit
