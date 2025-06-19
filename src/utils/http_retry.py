"""
HTTP retry logic with configurable decorators
"""

from typing import Optional, Dict, Any, Union, Callable, TypeVar, Tuple
import asyncio
import logging
from functools import wraps
from datetime import datetime
from typing import Optional, Dict, Any, List, Union, Tuple

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        wait_fixed,
        retry_if_exception_type,
        retry_if_result,
        before_sleep_log,
        after_log,
        RetryCallState
    )
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False
    logger.info("Warning: tenacity not installed. Using basic retry logic.")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from src.utils.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

# Type variable for generic return types
T = TypeVar('T')

# Default retry exceptions
DEFAULT_EXCEPTIONS = []

if HAS_REQUESTS:
    DEFAULT_EXCEPTIONS.extend([
        requests.exceptions.RequestException,
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
    ])

if HAS_AIOHTTP:
    DEFAULT_EXCEPTIONS.extend([
        aiohttp.ClientError,
        aiohttp.ServerTimeoutError,
        aiohttp.ClientConnectionError,
    ])

DEFAULT_EXCEPTIONS.extend([
    TimeoutError,
    ConnectionError,
    OSError,
    IOError,
])

# Convert to tuple for use in retry
DEFAULT_EXCEPTIONS = tuple(DEFAULT_EXCEPTIONS)

def is_retryable_status_code(response: Any) -> bool:
    """Check if HTTP response has retryable status code"""
    status_code = None
    
    # Extract status code from different response types
    if hasattr(response, 'status_code'):
        status_code = response.status_code
    elif hasattr(response, 'status'):
        status_code = response.status
    
    if status_code is None:
        return False
    
    # Retry on 5xx errors and specific 4xx errors
    retryable_codes = {408, 429, 500, 502, 503, 504}
    return status_code in retryable_codes

def log_retry_attempt(retry_state: 'RetryCallState') -> None:
    """Log retry attempt details"""
    logger.warning("Retrying HTTP request", extra={
        "attempt": retry_state.attempt_number,
        "wait_time": retry_state.next_action.sleep if retry_state.next_action else 0,
        "exception": str(retry_state.outcome.exception()) if retry_state.outcome else None
    })

if HAS_TENACITY:
    # Create retry decorator with tenacity
    def create_retry_decorator(
        max_attempts: int = 3,
        initial_wait: float = 1.0,
        max_wait: float = 60.0,
        exponential_base: float = 2.0,
        exceptions: Optional[Tuple[type, ...]] = None,
        retry_on_result: Optional[Callable[[Any], bool]] = None
    ) -> Callable:
        """Create a retry decorator with custom settings"""
        
        if exceptions is None:
            exceptions = DEFAULT_EXCEPTIONS
        
        retry_conditions = [retry_if_exception_type(exceptions)]
        
        if retry_on_result:
            retry_conditions.append(retry_if_result(retry_on_result))
        
        # Combine retry conditions
        retry_condition = retry_conditions[0]
        for condition in retry_conditions[1:]:
            retry_condition = retry_condition | condition
        
        return retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(
                multiplier=initial_wait,
                max=max_wait,
                exp_base=exponential_base
            ),
            retry=retry_condition,
            before_sleep=before_sleep_log(logger, logging.WARNING),
            after=after_log(logger, logging.DEBUG)
        )
else:
    # Fallback retry decorator without tenacity
    def create_retry_decorator(
        max_attempts: int = 3,
        initial_wait: float = 1.0,
        max_wait: float = 60.0,
        exponential_base: float = 2.0,
        exceptions: Optional[Tuple[type, ...]] = None,
        retry_on_result: Optional[Callable[[Any], bool]] = None
    ) -> Callable:
        """Create a basic retry decorator without tenacity"""
        
        if exceptions is None:
            exceptions = DEFAULT_EXCEPTIONS
        
        def decorator(func) -> Any:
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                last_exception = None
                wait_time = initial_wait
                
                for attempt in range(max_attempts):
                    try:
                        result = await func(*args, **kwargs)
                        
                        # Check if we should retry based on result
                        if retry_on_result and retry_on_result(result):
                            if attempt < max_attempts - 1:
                                logger.warning("Retrying due to result check", extra={
                                    "attempt": attempt + 1,
                                    "wait_time": wait_time
                                })
                                await asyncio.sleep(wait_time)
                                wait_time = min(wait_time * exponential_base, max_wait)
                                continue
                        
                        return result
                        
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            logger.warning("Retrying after exception", extra={
                                "attempt": attempt + 1,
                                "wait_time": wait_time,
                                "exception": str(e)
                            })
                            await asyncio.sleep(wait_time)
                            wait_time = min(wait_time * exponential_base, max_wait)
                        else:
                            logger.error("Max retry attempts reached", extra={
                                "attempts": max_attempts,
                                "exception": str(e)
                            })
                
                if last_exception:
                    raise last_exception
                else:
                    raise Exception("Max retry attempts reached")
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                last_exception = None
                wait_time = initial_wait
                
                for attempt in range(max_attempts):
                    try:
                        result = func(*args, **kwargs)
                        
                        if retry_on_result and retry_on_result(result):
                            if attempt < max_attempts - 1:
                                logger.warning("Retrying due to result check", extra={
                                    "attempt": attempt + 1,
                                    "wait_time": wait_time
                                })
                                import time
                                time.sleep(wait_time)
                                wait_time = min(wait_time * exponential_base, max_wait)
                                continue
                        
                        return result
                        
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            logger.warning("Retrying after exception", extra={
                                "attempt": attempt + 1,
                                "wait_time": wait_time,
                                "exception": str(e)
                            })
                            import time
                            time.sleep(wait_time)
                            wait_time = min(wait_time * exponential_base, max_wait)
                        else:
                            logger.error("Max retry attempts reached", extra={
                                "attempts": max_attempts,
                                "exception": str(e)
                            })
                
                if last_exception:
                    raise last_exception
                else:
                    raise Exception("Max retry attempts reached")
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator

# Standard HTTP retry decorator
http_retry = create_retry_decorator(
    max_attempts=3,
    initial_wait=1.0,
    max_wait=60.0,
    retry_on_result=is_retryable_status_code
)

# Aggressive retry for critical operations
critical_http_retry = create_retry_decorator(
    max_attempts=5,
    initial_wait=2.0,
    max_wait=120.0,
    retry_on_result=is_retryable_status_code
)

# Quick retry for fast failures
quick_http_retry = create_retry_decorator(
    max_attempts=2,
    initial_wait=0.5,
    max_wait=10.0,
    retry_on_result=is_retryable_status_code
)

# No retry decorator (for operations that should not retry)
no_retry = lambda func: func

# Retry with custom backoff
def custom_retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0
) -> Callable:
    """Create custom retry decorator with specified parameters"""
    return create_retry_decorator(
        max_attempts=max_attempts,
        initial_wait=1.0,
        max_wait=max_delay,
        exponential_base=backoff_factor,
        retry_on_result=is_retryable_status_code
    )

# Retry only on specific status codes
def retry_on_status(*status_codes: int) -> Callable:
    """Retry only on specific HTTP status codes"""
    def check_status(response: Any) -> bool:
        status = None
        if hasattr(response, 'status_code'):
            status = response.status_code
        elif hasattr(response, 'status'):
            status = response.status
        
        return status in status_codes
    
    return create_retry_decorator(
        max_attempts=3,
        retry_on_result=check_status
    )

# Helper functions for HTTP operations
@http_retry
async def fetch_with_retry(
    session: 'aiohttp.ClientSession',
    url: str,
    method: str = 'GET',
    **kwargs
) -> 'aiohttp.ClientResponse':
    """Fetch URL with automatic retry using aiohttp"""
    if not HAS_AIOHTTP:
        raise ImportError("aiohttp is required for async HTTP operations")
    
    logger.info("HTTP request", extra={
        "method": method,
        "url": url
    })
    
    async with session.request(method, url, **kwargs) as response:
        # Raise for status to trigger retry on error codes
        response.raise_for_status()
        return response

@http_retry
def requests_with_retry(
    url: str,
    method: str = 'GET',
    **kwargs
) -> 'requests.Response':
    """Make HTTP request with automatic retry using requests"""
    if not HAS_REQUESTS:
        raise ImportError("requests is required for sync HTTP operations")
    
    logger.info("HTTP request", extra={
        "method": method,
        "url": url
    })
    
    response = requests.request(method, url, **kwargs)
    response.raise_for_status()
    return response

# Context manager for retry scope
class RetryContext:
    """Context manager for retry operations"""
    
    def __init__(self, ,

    
            max_attempts: int = 3        exceptions: Optional[Tuple[type        ...]]: Optional[Any] = None        operation_name: str = "operation") -> None:
        self.max_attempts = max_attempts
        self.exceptions = exceptions or DEFAULT_EXCEPTIONS
        self.operation_name = operation_name
        self.attempts = 0
    
    async def __aenter__(self) -> Any:
        self.attempts = 0
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        if exc_type and issubclass(exc_type, self.exceptions):
            self.attempts += 1
            if self.attempts < self.max_attempts:
                wait_time = 2 ** self.attempts
                logger.warning("Retrying operation", extra={
                    "operation": self.operation_name,
                    "attempt": self.attempts,
                    "wait_time": wait_time,
                    "exception": str(exc_val)
                })
                await asyncio.sleep(wait_time)
                return True  # Suppress exception to retry
        
        return False  # Don't suppress exception

# Utility function to check if request should be retried
def should_retry_request(
    exception: Optional[Exception] = None,
    response: Optional[Any] = None,
    status_code: Optional[int] = None
) -> bool:
    """Determine if a request should be retried"""
    
    # Check exception
    if exception:
        return isinstance(exception, DEFAULT_EXCEPTIONS)
    
    # Check response
    if response:
        return is_retryable_status_code(response)
    
    # Check status code directly
    if status_code:
        return status_code in {408, 429, 500, 502, 503, 504}
    
    return False

# Export main decorators and utilities
__all__ = [
    # Decorators
    'http_retry',
    'critical_http_retry',
    'quick_http_retry',
    'no_retry',
    'custom_retry',
    'retry_on_status',
    
    # Functions
    'create_retry_decorator',
    'fetch_with_retry',
    'requests_with_retry',
    'should_retry_request',
    'is_retryable_status_code',
    
    # Classes
    'RetryContext',
    
    # Constants
    'DEFAULT_EXCEPTIONS',
] 