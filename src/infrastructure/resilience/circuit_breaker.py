"""
Circuit Breaker Pattern Implementation
Provides resilience for external service calls and database operations
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable, TypeVar, Awaitable
from enum import Enum
from dataclasses import dataclass, field
from contextlib import asynccontextmanager, contextmanager
import functools

logger = logging.getLogger(__name__)

T = TypeVar('T')

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    expected_exception: type = Exception
    monitor_interval: float = 10.0  # seconds
    success_threshold: int = 2  # Successes needed to close circuit

@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    current_state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    consecutive_successes: int = 0

class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
        self._last_state_change = time.time()
        
    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            await self._check_state()
            
        try:
            if self.stats.current_state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is OPEN")
                
            # Execute the function
            result = await func(*args, **kwargs)
            
            # Record success
            await self._record_success()
            return result
            
        except self.config.expected_exception as e:
            # Record failure
            await self._record_failure()
            raise
            
    def call_sync(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute synchronous function with circuit breaker protection"""
        if self.stats.current_state == CircuitState.OPEN:
            raise CircuitBreakerOpenError(f"Circuit breaker '{self.name}' is OPEN")
            
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Record success
            self._record_success_sync()
            return result
            
        except self.config.expected_exception as e:
            # Record failure
            self._record_failure_sync()
            raise
            
    async def _check_state(self):
        """Check and update circuit breaker state"""
        current_time = time.time()
        
        if self.stats.current_state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (current_time - self._last_state_change) >= self.config.recovery_timeout:
                logger.info("Circuit breaker '{}' transitioning to HALF_OPEN", extra={"self_name": self.name})
                self.stats.current_state = CircuitState.HALF_OPEN
                self._last_state_change = current_time
                
    async def _record_success(self):
        """Record successful operation"""
        self.stats.total_requests += 1
        self.stats.successful_requests += 1
        self.stats.last_success_time = time.time()
        self.stats.consecutive_successes += 1
        self.stats.consecutive_failures = 0
        
        if self.stats.current_state == CircuitState.HALF_OPEN:
            if self.stats.consecutive_successes >= self.config.success_threshold:
                logger.info("Circuit breaker '{}' transitioning to CLOSED", extra={"self_name": self.name})
                self.stats.current_state = CircuitState.CLOSED
                self._last_state_change = time.time()
                
    async def _record_failure(self):
        """Record failed operation"""
        self.stats.total_requests += 1
        self.stats.failed_requests += 1
        self.stats.last_failure_time = time.time()
        self.stats.consecutive_failures += 1
        self.stats.consecutive_successes = 0
        
        if self.stats.current_state == CircuitState.CLOSED:
            if self.stats.consecutive_failures >= self.config.failure_threshold:
                logger.warning("Circuit breaker '{}' transitioning to OPEN", extra={"self_name": self.name})
                self.stats.current_state = CircuitState.OPEN
                self._last_state_change = time.time()
        elif self.stats.current_state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker '{}' transitioning back to OPEN", extra={"self_name": self.name})
            self.stats.current_state = CircuitState.OPEN
            self._last_state_change = time.time()
            
    def _record_success_sync(self):
        """Record successful operation (synchronous)"""
        self.stats.total_requests += 1
        self.stats.successful_requests += 1
        self.stats.last_success_time = time.time()
        self.stats.consecutive_successes += 1
        self.stats.consecutive_failures = 0
        
        if self.stats.current_state == CircuitState.HALF_OPEN:
            if self.stats.consecutive_successes >= self.config.success_threshold:
                logger.info("Circuit breaker '{}' transitioning to CLOSED", extra={"self_name": self.name})
                self.stats.current_state = CircuitState.CLOSED
                self._last_state_change = time.time()
                
    def _record_failure_sync(self):
        """Record failed operation (synchronous)"""
        self.stats.total_requests += 1
        self.stats.failed_requests += 1
        self.stats.last_failure_time = time.time()
        self.stats.consecutive_failures += 1
        self.stats.consecutive_successes = 0
        
        if self.stats.current_state == CircuitState.CLOSED:
            if self.stats.consecutive_failures >= self.config.failure_threshold:
                logger.warning("Circuit breaker '{}' transitioning to OPEN", extra={"self_name": self.name})
                self.stats.current_state = CircuitState.OPEN
                self._last_state_change = time.time()
        elif self.stats.current_state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker '{}' transitioning back to OPEN", extra={"self_name": self.name})
            self.stats.current_state = CircuitState.OPEN
            self._last_state_change = time.time()
            
    def get_stats(self) -> CircuitBreakerStats:
        """Get current circuit breaker statistics"""
        return self.stats
        
    def reset(self):
        """Reset circuit breaker to closed state"""
        self.stats.current_state = CircuitState.CLOSED
        self.stats.consecutive_failures = 0
        self.stats.consecutive_successes = 0
        self._last_state_change = time.time()
        logger.info("Circuit breaker '{}' reset to CLOSED", extra={"self_name": self.name})

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

# =============================
# Circuit Breaker Registry
# =============================

class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers"""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
        
    async def get_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker"""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]
            
    def get_breaker_sync(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker (synchronous)"""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config)
        return self._breakers[name]
        
    def get_all_breakers(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers"""
        return self._breakers.copy()
        
    async def reset_all(self):
        """Reset all circuit breakers"""
        async with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
                
    def get_stats(self) -> Dict[str, CircuitBreakerStats]:
        """Get statistics for all circuit breakers"""
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}

# Global circuit breaker registry
circuit_breaker_registry = CircuitBreakerRegistry()

# =============================
# Decorators
# =============================

def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to add circuit breaker protection to functions"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            breaker = await circuit_breaker_registry.get_breaker(name, config)
            return await breaker.call(func, *args, **kwargs)
            
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            breaker = circuit_breaker_registry.get_breaker_sync(name, config)
            return breaker.call_sync(func, *args, **kwargs)
            
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator

# =============================
# Context Managers
# =============================

@asynccontextmanager
async def circuit_breaker_context(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Async context manager for circuit breaker protection"""
    breaker = await circuit_breaker_registry.get_breaker(name, config)
    
    async with breaker._lock:
        await breaker._check_state()
        
    if breaker.stats.current_state == CircuitState.OPEN:
        raise CircuitBreakerOpenError(f"Circuit breaker '{name}' is OPEN")
        
    try:
        yield breaker
    except breaker.config.expected_exception:
        await breaker._record_failure()
        raise

@contextmanager
def circuit_breaker_context_sync(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Synchronous context manager for circuit breaker protection"""
    breaker = circuit_breaker_registry.get_breaker_sync(name, config)
    
    if breaker.stats.current_state == CircuitState.OPEN:
        raise CircuitBreakerOpenError(f"Circuit breaker '{name}' is OPEN")
        
    try:
        yield breaker
    except breaker.config.expected_exception:
        breaker._record_failure_sync()
        raise

# =============================
# Pre-configured Circuit Breakers
# =============================

# Database circuit breaker configuration
DB_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=30.0,
    expected_exception=Exception,
    success_threshold=2
)

# External API circuit breaker configuration
API_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0,
    expected_exception=Exception,
    success_threshold=3
)

# Redis circuit breaker configuration
REDIS_CIRCUIT_BREAKER_CONFIG = CircuitBreakerConfig(
    failure_threshold=2,
    recovery_timeout=15.0,
    expected_exception=Exception,
    success_threshold=1
)

# =============================
# Utility Functions
# =============================

async def get_db_circuit_breaker() -> CircuitBreaker:
    """Get database circuit breaker"""
    return await circuit_breaker_registry.get_breaker("database", DB_CIRCUIT_BREAKER_CONFIG)

async def get_api_circuit_breaker() -> CircuitBreaker:
    """Get API circuit breaker"""
    return await circuit_breaker_registry.get_breaker("api", API_CIRCUIT_BREAKER_CONFIG)

async def get_redis_circuit_breaker() -> CircuitBreaker:
    """Get Redis circuit breaker"""
    return await circuit_breaker_registry.get_breaker("redis", REDIS_CIRCUIT_BREAKER_CONFIG)

def get_all_circuit_breaker_stats() -> Dict[str, CircuitBreakerStats]:
    """Get statistics for all circuit breakers"""
    return circuit_breaker_registry.get_stats()

async def reset_all_circuit_breakers():
    """Reset all circuit breakers"""
    await circuit_breaker_registry.reset_all() 