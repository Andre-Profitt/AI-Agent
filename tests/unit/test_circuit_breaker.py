from examples.enhanced_unified_example import tasks
from examples.gaia_usage_example import result1
from examples.gaia_usage_example import result2
from examples.parallel_execution_example import results
from migrations.env import config
from performance_dashboard import circuit_breaker_registry
from performance_dashboard import stats

from src.gaia_components.adaptive_tool_system import breaker
from src.infrastructure.resilience.circuit_breaker import API_CIRCUIT_BREAKER_CONFIG
from src.infrastructure.resilience.circuit_breaker import DB_CIRCUIT_BREAKER_CONFIG
from src.infrastructure.resilience.circuit_breaker import REDIS_CIRCUIT_BREAKER_CONFIG
from src.infrastructure.resilience.circuit_breaker import circuit_breaker_context_sync
from src.infrastructure.resilience.circuit_breaker import get_all_circuit_breaker_stats
from src.tools_introspection import error

from src.infrastructure.resilience.circuit_breaker import CircuitBreakerOpenError
from src.infrastructure.resilience.circuit_breaker import CircuitBreakerRegistry
from src.infrastructure.resilience.circuit_breaker import CircuitBreakerStats
from src.services.circuit_breaker import CircuitBreaker
from src.services.circuit_breaker import CircuitBreakerConfig
from src.services.circuit_breaker import CircuitState
# TODO: Fix undefined variables: API_CIRCUIT_BREAKER_CONFIG, DB_CIRCUIT_BREAKER_CONFIG, REDIS_CIRCUIT_BREAKER_CONFIG, _, all_breakers, api_breaker, breaker, breaker1, breaker2, circuit_breaker_context, circuit_breaker_registry, config, db_breaker, error, get_api_circuit_breaker, get_db_circuit_breaker, get_redis_circuit_breaker, reset_all_circuit_breakers, result, result1, result2, results, stats, tasks
from src.infrastructure.resilience.circuit_breaker import circuit_breaker_context_sync
from src.infrastructure.resilience.circuit_breaker import get_all_circuit_breaker_stats

# TODO: Fix undefined variables: API_CIRCUIT_BREAKER_CONFIG, DB_CIRCUIT_BREAKER_CONFIG, REDIS_CIRCUIT_BREAKER_CONFIG, _, all_breakers, api_breaker, breaker, breaker1, breaker2, circuit_breaker_context, circuit_breaker_context_sync, circuit_breaker_registry, config, db_breaker, error, get_all_circuit_breaker_stats, get_api_circuit_breaker, get_db_circuit_breaker, get_redis_circuit_breaker, reset_all_circuit_breakers, result, result1, result2, results, stats, tasks
"""
Unit tests for circuit breaker pattern implementation
"""

import pytest
import asyncio

from src.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerStats,
    CircuitState,
    CircuitBreakerOpenError,
    CircuitBreakerRegistry,
    circuit_breaker,
    circuit_breaker_context,
    circuit_breaker_context_sync,
    DB_CIRCUIT_BREAKER_CONFIG,
    API_CIRCUIT_BREAKER_CONFIG,
    REDIS_CIRCUIT_BREAKER_CONFIG,
    get_db_circuit_breaker,
    get_api_circuit_breaker,
    get_redis_circuit_breaker,
    get_all_circuit_breaker_stats,
    reset_all_circuit_breakers,
    circuit_breaker_registry
)

class TestCircuitBreakerConfig:
    """Test circuit breaker configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.expected_exception == Exception
        assert config.monitor_interval == 10.0
        assert config.success_threshold == 2

    def test_custom_config(self):
        """Test custom configuration values"""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exception=ValueError,
            monitor_interval=5.0,
            success_threshold=1
        )

        assert config.failure_threshold == 3
        assert config.recovery_timeout == 30.0
        assert config.expected_exception == ValueError
        assert config.monitor_interval == 5.0
        assert config.success_threshold == 1

class TestCircuitBreakerStats:
    """Test circuit breaker statistics"""

    def test_default_stats(self):
        """Test default statistics values"""
        stats = CircuitBreakerStats()

        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0
        assert stats.last_failure_time is None
        assert stats.last_success_time is None
        assert stats.current_state == CircuitState.CLOSED
        assert stats.consecutive_failures == 0
        assert stats.consecutive_successes == 0

class TestCircuitBreaker:
    """Test circuit breaker functionality"""

    @pytest.fixture
    def circuit_breaker(self):
        """Create a circuit breaker for testing"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=1.0,
            success_threshold=1
        )
        return CircuitBreaker("test_breaker", config)

    @pytest.mark.asyncio
    async def test_initial_state(self, circuit_breaker):
        """Test initial circuit breaker state"""
        assert circuit_breaker.stats.current_state == CircuitState.CLOSED
        assert circuit_breaker.stats.total_requests == 0

    @pytest.mark.asyncio
    async def test_successful_call(self, circuit_breaker):
        """Test successful function call"""
        async def success_func():
            return "success"

        result = await circuit_breaker.call(success_func)

        assert result == "success"
        assert circuit_breaker.stats.current_state == CircuitState.CLOSED
        assert circuit_breaker.stats.total_requests == 1
        assert circuit_breaker.stats.successful_requests == 1
        assert circuit_breaker.stats.failed_requests == 0
        assert circuit_breaker.stats.consecutive_successes == 1
        assert circuit_breaker.stats.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_failed_call(self, circuit_breaker):
        """Test failed function call"""
        async def failure_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await circuit_breaker.call(failure_func)

        assert circuit_breaker.stats.current_state == CircuitState.CLOSED
        assert circuit_breaker.stats.total_requests == 1
        assert circuit_breaker.stats.successful_requests == 0
        assert circuit_breaker.stats.failed_requests == 1
        assert circuit_breaker.stats.consecutive_successes == 0
        assert circuit_breaker.stats.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self, circuit_breaker):
        """Test circuit opens after failure threshold"""
        async def failure_func():
            raise ValueError("Test error")

        # First failure
        with pytest.raises(ValueError):
            await circuit_breaker.call(failure_func)
        assert circuit_breaker.stats.current_state == CircuitState.CLOSED

        # Second failure - should open circuit
        with pytest.raises(ValueError):
            await circuit_breaker.call(failure_func)
        assert circuit_breaker.stats.current_state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_blocks_when_open(self, circuit_breaker):
        """Test circuit blocks calls when open"""
        # Open the circuit
        async def failure_func():
            raise ValueError("Test error")

        for _ in range(2):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failure_func)

        # Circuit should be open now
        assert circuit_breaker.stats.current_state == CircuitState.OPEN

        # Try to call a function - should be blocked
        async def success_func():
            return "success"

        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(success_func)

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open(self, circuit_breaker):
        """Test circuit transitions to half-open after timeout"""
        # Open the circuit
        async def failure_func():
            raise ValueError("Test error")

        for _ in range(2):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failure_func)

        assert circuit_breaker.stats.current_state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Check state - should be half-open
        await circuit_breaker._check_state()
        assert circuit_breaker.stats.current_state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_closes_after_success_threshold(self, circuit_breaker):
        """Test circuit closes after success threshold in half-open state"""
        # Open the circuit
        async def failure_func():
            raise ValueError("Test error")

        for _ in range(2):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failure_func)

        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        await circuit_breaker._check_state()

        # Should be half-open now
        assert circuit_breaker.stats.current_state == CircuitState.HALF_OPEN

        # Successful call should close the circuit
        async def success_func():
            return "success"

        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.stats.current_state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_reopens_after_failure_in_half_open(self, circuit_breaker):
        """Test circuit reopens after failure in half-open state"""
        # Open the circuit
        async def failure_func():
            raise ValueError("Test error")

        for _ in range(2):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failure_func)

        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        await circuit_breaker._check_state()

        # Should be half-open now
        assert circuit_breaker.stats.current_state == CircuitState.HALF_OPEN

        # Another failure should reopen the circuit
        with pytest.raises(ValueError):
            await circuit_breaker.call(failure_func)

        assert circuit_breaker.stats.current_state == CircuitState.OPEN

    def test_sync_call_success(self, circuit_breaker):
        """Test successful synchronous function call"""
        def success_func():
            return "success"

        result = circuit_breaker.call_sync(success_func)

        assert result == "success"
        assert circuit_breaker.stats.current_state == CircuitState.CLOSED
        assert circuit_breaker.stats.total_requests == 1
        assert circuit_breaker.stats.successful_requests == 1

    def test_sync_call_failure(self, circuit_breaker):
        """Test failed synchronous function call"""
        def failure_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            circuit_breaker.call_sync(failure_func)

        assert circuit_breaker.stats.current_state == CircuitState.CLOSED
        assert circuit_breaker.stats.total_requests == 1
        assert circuit_breaker.stats.failed_requests == 1

    def test_get_stats(self, circuit_breaker):
        """Test getting circuit breaker statistics"""
        stats = circuit_breaker.get_stats()

        assert isinstance(stats, CircuitBreakerStats)
        assert stats.current_state == CircuitState.CLOSED

    def test_reset(self, circuit_breaker):
        """Test resetting circuit breaker"""
        # Open the circuit first
        def failure_func():
            raise ValueError("Test error")

        for _ in range(2):
            with pytest.raises(ValueError):
                circuit_breaker.call_sync(failure_func)

        assert circuit_breaker.stats.current_state == CircuitState.OPEN

        # Reset the circuit
        circuit_breaker.reset()

        assert circuit_breaker.stats.current_state == CircuitState.CLOSED
        assert circuit_breaker.stats.consecutive_failures == 0
        assert circuit_breaker.stats.consecutive_successes == 0

class TestCircuitBreakerRegistry:
    """Test circuit breaker registry"""

    @pytest.fixture
    def registry(self):
        """Create a circuit breaker registry for testing"""
        return CircuitBreakerRegistry()

    @pytest.mark.asyncio
    async def test_get_breaker_creates_new(self, registry):
        """Test getting a breaker creates a new one"""
        breaker = await registry.get_breaker("test_breaker")

        assert isinstance(breaker, CircuitBreaker)
        assert breaker.name == "test_breaker"

    @pytest.mark.asyncio
    async def test_get_breaker_returns_existing(self, registry):
        """Test getting a breaker returns existing one"""
        breaker1 = await registry.get_breaker("test_breaker")
        breaker2 = await registry.get_breaker("test_breaker")

        assert breaker1 is breaker2

    @pytest.mark.asyncio
    async def test_get_breaker_with_config(self, registry):
        """Test getting a breaker with custom config"""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = await registry.get_breaker("test_breaker", config)

        assert breaker.config.failure_threshold == 3

    def test_get_breaker_sync(self, registry):
        """Test getting a breaker synchronously"""
        breaker = registry.get_breaker_sync("test_breaker")

        assert isinstance(breaker, CircuitBreaker)
        assert breaker.name == "test_breaker"

    def test_get_all_breakers(self, registry):
        """Test getting all breakers"""
        registry.get_breaker_sync("breaker1")
        registry.get_breaker_sync("breaker2")

        all_breakers = registry.get_all_breakers()
        assert len(all_breakers) == 2
        assert "breaker1" in all_breakers
        assert "breaker2" in all_breakers

    @pytest.mark.asyncio
    async def test_reset_all(self, registry):
        """Test resetting all breakers"""
        breaker1 = await registry.get_breaker("breaker1")
        breaker2 = await registry.get_breaker("breaker2")

        # Open both circuits
        for breaker in [breaker1, breaker2]:
            for _ in range(2):
                with pytest.raises(ValueError):
                    await breaker.call(lambda: (_ for _ in ()).throw(ValueError("Test")))

        assert breaker1.stats.current_state == CircuitState.OPEN
        assert breaker2.stats.current_state == CircuitState.OPEN

        # Reset all
        await registry.reset_all()

        assert breaker1.stats.current_state == CircuitState.CLOSED
        assert breaker2.stats.current_state == CircuitState.CLOSED

    def test_get_stats(self, registry):
        """Test getting statistics for all breakers"""
        registry.get_breaker_sync("breaker1")
        registry.get_breaker_sync("breaker2")

        stats = registry.get_stats()
        assert len(stats) == 2
        assert "breaker1" in stats
        assert "breaker2" in stats
        assert isinstance(stats["breaker1"], CircuitBreakerStats)

class TestDecorators:
    """Test circuit breaker decorators"""

    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator_async(self):
        """Test circuit_breaker decorator with async function"""
        @circuit_breaker("test_breaker")
        async def test_async_func():
            return "success"

        result = await test_async_func()
        assert result == "success"

    def test_circuit_breaker_decorator_sync(self):
        """Test circuit_breaker decorator with sync function"""
        @circuit_breaker("test_breaker")
        def test_sync_func():
            return "success"

        result = test_sync_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator_with_failure(self):
        """Test circuit_breaker decorator with failure"""
        @circuit_breaker("test_breaker", CircuitBreakerConfig(failure_threshold=1))
        async def test_failure_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await test_failure_func()

        # Second call should be blocked
        with pytest.raises(CircuitBreakerOpenError):
            await test_failure_func()

class TestContextManagers:
    """Test circuit breaker context managers"""

    @pytest.mark.asyncio
    async def test_circuit_breaker_context_async(self):
        """Test async circuit breaker context manager"""
        async with circuit_breaker_context("test_breaker") as breaker:
            assert isinstance(breaker, CircuitBreaker)
            assert breaker.name == "test_breaker"

    def test_circuit_breaker_context_sync(self):
        """Test sync circuit breaker context manager"""
        with circuit_breaker_context_sync("test_breaker") as breaker:
            assert isinstance(breaker, CircuitBreaker)
            assert breaker.name == "test_breaker"

    @pytest.mark.asyncio
    async def test_circuit_breaker_context_with_open_circuit(self):
        """Test context manager with open circuit"""
        # Open a circuit first
        breaker = await circuit_breaker_registry.get_breaker("test_breaker",
                                                           CircuitBreakerConfig(failure_threshold=1))

        async def failure_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await breaker.call(failure_func)

        # Context manager should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            async with circuit_breaker_context("test_breaker"):
                pass

class TestPreConfiguredConfigs:
    """Test pre-configured circuit breaker configurations"""

    def test_db_circuit_breaker_config(self):
        """Test database circuit breaker configuration"""
        assert DB_CIRCUIT_BREAKER_CONFIG.failure_threshold == 3
        assert DB_CIRCUIT_BREAKER_CONFIG.recovery_timeout == 30.0
        assert DB_CIRCUIT_BREAKER_CONFIG.success_threshold == 2

    def test_api_circuit_breaker_config(self):
        """Test API circuit breaker configuration"""
        assert API_CIRCUIT_BREAKER_CONFIG.failure_threshold == 5
        assert API_CIRCUIT_BREAKER_CONFIG.recovery_timeout == 60.0
        assert API_CIRCUIT_BREAKER_CONFIG.success_threshold == 3

    def test_redis_circuit_breaker_config(self):
        """Test Redis circuit breaker configuration"""
        assert REDIS_CIRCUIT_BREAKER_CONFIG.failure_threshold == 2
        assert REDIS_CIRCUIT_BREAKER_CONFIG.recovery_timeout == 15.0
        assert REDIS_CIRCUIT_BREAKER_CONFIG.success_threshold == 1

class TestUtilityFunctions:
    """Test utility functions"""

    @pytest.mark.asyncio
    async def test_get_db_circuit_breaker(self):
        """Test getting database circuit breaker"""
        breaker = await get_db_circuit_breaker()

        assert isinstance(breaker, CircuitBreaker)
        assert breaker.name == "database"
        assert breaker.config.failure_threshold == 3

    @pytest.mark.asyncio
    async def test_get_api_circuit_breaker(self):
        """Test getting API circuit breaker"""
        breaker = await get_api_circuit_breaker()

        assert isinstance(breaker, CircuitBreaker)
        assert breaker.name == "api"
        assert breaker.config.failure_threshold == 5

    @pytest.mark.asyncio
    async def test_get_redis_circuit_breaker(self):
        """Test getting Redis circuit breaker"""
        breaker = await get_redis_circuit_breaker()

        assert isinstance(breaker, CircuitBreaker)
        assert breaker.name == "redis"
        assert breaker.config.failure_threshold == 2

    def test_get_all_circuit_breaker_stats(self):
        """Test getting all circuit breaker statistics"""
        stats = get_all_circuit_breaker_stats()

        assert isinstance(stats, dict)
        # Should include the pre-configured breakers
        assert "database" in stats
        assert "api" in stats
        assert "redis" in stats

    @pytest.mark.asyncio
    async def test_reset_all_circuit_breakers(self):
        """Test resetting all circuit breakers"""
        # This should not raise any exceptions
        await reset_all_circuit_breakers()

class TestErrorHandling:
    """Test error handling"""

    def test_circuit_breaker_open_error(self):
        """Test CircuitBreakerOpenError"""
        error = CircuitBreakerOpenError("Test circuit is open")

        assert str(error) == "Test circuit is open"
        assert isinstance(error, Exception)

    @pytest.mark.asyncio
    async def test_unexpected_exception_does_not_trigger_circuit(self, circuit_breaker):
        """Test that unexpected exceptions don't trigger circuit breaker"""
        async def unexpected_error_func():
            raise RuntimeError("Unexpected error")

        # Should not trigger circuit breaker for unexpected exception
        with pytest.raises(RuntimeError):
            await circuit_breaker.call(unexpected_error_func)

        assert circuit_breaker.stats.current_state == CircuitState.CLOSED
        assert circuit_breaker.stats.failed_requests == 0

class TestConcurrency:
    """Test concurrency handling"""

    @pytest.mark.asyncio
    async def test_concurrent_calls(self, circuit_breaker):
        """Test concurrent calls to circuit breaker"""
        async def success_func():
            await asyncio.sleep(0.1)
            return "success"

        # Make multiple concurrent calls
        tasks = [circuit_breaker.call(success_func) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        assert all(result == "success" for result in results)
        assert circuit_breaker.stats.total_requests == 5
        assert circuit_breaker.stats.successful_requests == 5

    @pytest.mark.asyncio
    async def test_concurrent_failures(self, circuit_breaker):
        """Test concurrent failures"""
        async def failure_func():
            await asyncio.sleep(0.1)
            raise ValueError("Test error")

        # Make multiple concurrent calls that fail
        tasks = [circuit_breaker.call(failure_func) for _ in range(3)]

        with pytest.raises(ValueError):
            await asyncio.gather(*tasks)

        # Circuit should be open after threshold failures
        assert circuit_breaker.stats.current_state == CircuitState.OPEN

class TestIntegration:
    """Integration tests"""

    @pytest.mark.asyncio
    async def test_full_circuit_breaker_lifecycle(self):
        """Test complete circuit breaker lifecycle"""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.5,
            success_threshold=1
        )
        breaker = CircuitBreaker("lifecycle_test", config)

        # Start in closed state
        assert breaker.stats.current_state == CircuitState.CLOSED

        # Fail twice to open circuit
        async def failure_func():
            raise ValueError("Test error")

        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(failure_func)

        assert breaker.stats.current_state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.6)
        await breaker._check_state()

        # Should be half-open
        assert breaker.stats.current_state == CircuitState.HALF_OPEN

        # Success should close circuit
        async def success_func():
            return "success"

        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.stats.current_state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_registry_integration(self):
        """Test integration with registry"""
        # Get breakers from registry
        db_breaker = await get_db_circuit_breaker()
        api_breaker = await get_api_circuit_breaker()

        # Use them
        async def success_func():
            return "success"

        result1 = await db_breaker.call(success_func)
        result2 = await api_breaker.call(success_func)

        assert result1 == "success"
        assert result2 == "success"

        # Check stats
        stats = get_all_circuit_breaker_stats()
        assert "database" in stats
        assert "api" in stats
