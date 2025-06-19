"""
Circuit breaker pattern implementation
"""

import time
from typing import Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass

from src.utils.logging import get_logger
from typing import Optional, Dict, Any, List, Union, Tuple

logger = get_logger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_attempts: int = 3
    expected_exception: type = Exception

class CircuitBreaker:
    """Circuit breaker implementation with exponential backoff"""
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None) -> None:
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_attempts = 0
        
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.config.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
                self.half_open_attempts = 0
                logger.info("Circuit breaker transitioning to half-open state")
                return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            return self.half_open_attempts < self.config.half_open_attempts
        
        return False
    
    def record_success(self) -> Any:
        """Record a successful execution"""
        if self.state == CircuitState.HALF_OPEN:
            # Success in half-open state, close the circuit
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.half_open_attempts = 0
            logger.info("Circuit breaker closed after successful half-open execution")
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def record_failure(self, exception: Optional[Exception] = None) -> Any:
        """Record a failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            # Failure in half-open state, open the circuit
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker opened after failure in half-open state")
        elif (self.state == CircuitState.CLOSED and 
              self.failure_count >= self.config.failure_threshold):
            # Too many failures, open the circuit
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker opened after {} failures", extra={"self_failure_count": self.failure_count})
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if not self.can_execute():
            raise Exception(f"Circuit breaker is {self.state.value}")
        
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_attempts += 1
        
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(e)
            raise
    
    async def acall(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection"""
        if not self.can_execute():
            raise Exception(f"Circuit breaker is {self.state.value}")
        
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_attempts += 1
        
        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure(e)
            raise
    
    def get_status(self) -> dict:
        """Get current circuit breaker status"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "half_open_attempts": self.half_open_attempts,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "half_open_attempts": self.config.half_open_attempts
            }
        }
    
    def reset(self) -> Any:
        """Reset circuit breaker to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.half_open_attempts = 0
        logger.info("Circuit breaker reset to closed state") 