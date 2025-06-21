from agent import response
from app import app
from migrations.env import config
from tests.load_test import args
from tests.load_test import headers
from tests.load_test import wait_time
from tests.unit.simple_test import func

from src.api.rate_limiter import basic_config
from src.api.rate_limiter import enterprise_config
from src.api.rate_limiter import free_config
from src.api.rate_limiter import limiter
from src.api.rate_limiter import premium_config
from src.api.rate_limiter import retry_after
from src.api.rate_limiter import space_needed
from src.api.rate_limiter import tier
from src.api.rate_limiter import time_passed
from src.api.rate_limiter import tokens_needed
from src.api.rate_limiter import tokens_to_add
from src.api.rate_limiter import water_leaked
from src.api.rate_limiter import window
from src.api.rate_limiter import window_start
from src.api.rate_limiter_enhanced import auth_header
from src.api.rate_limiter_enhanced import bucket
from src.api_server import message
from src.core.monitoring import key
from src.database.models import user_id
from src.gaia_components.enhanced_memory_system import current_time

"""
from collections import deque
from dataclasses import dataclass
# TODO: Fix undefined variables: Any, Dict, Enum, Optional, Tuple, allowed, app, args, auth_header, basic_config, bucket, config, current_time, dataclass, defaultdict, deque, endpoint, enterprise_config, free_config, func, headers, key, kwargs, limiter, logging, max_wait, message, premium_config, rate_limiter, receive, response, retry_after, scope, send, space_needed, tier, time, time_passed, tokens_needed, tokens_to_add, user_id, wait_time, water_leaked, window, window_start
# TODO: Fix undefined variables: allowed, app, args, auth_header, basic_config, bucket, config, current_time, endpoint, enterprise_config, free_config, func, headers, key, kwargs, limiter, max_wait, message, premium_config, rate_limiter, receive, response, retry_after, scope, self, send, space_needed, tier, time_passed, tokens_needed, tokens_to_add, user_id, wait_time, water_leaked, window, window_start

from sqlalchemy import func
Rate Limiting System for GAIA API
Implements comprehensive rate limiting with multiple strategies
"""

from typing import Tuple
from typing import Optional
from typing import Any
import requests

import time
import asyncio
from typing import Dict, Optional, Tuple

from collections import defaultdict, deque
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_allowance: int = 10
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    window_size: int = 60  # seconds
    tokens_per_second: float = 1.0
    bucket_capacity: int = 60

class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message)
        self.retry_after = retry_after

class FixedWindowRateLimiter:
    """Fixed window rate limiter"""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.windows = defaultdict(lambda: {"count": 0, "reset_time": 0})

    def is_allowed(self, key: str) -> Tuple[bool, Optional[int]]:
        """Check if request is allowed"""
        current_time = time.time()
        window = self.windows[key]

        # Check if window has reset
        if current_time >= window["reset_time"]:
            window["count"] = 0
            window["reset_time"] = current_time + self.config.window_size

        # Check if limit exceeded
        if window["count"] >= self.config.requests_per_minute:
            retry_after = int(window["reset_time"] - current_time)
            return False, retry_after

        # Increment counter
        window["count"] += 1
        return True, None

class SlidingWindowRateLimiter:
    """Sliding window rate limiter"""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests = defaultdict(deque)

    def is_allowed(self, key: str) -> Tuple[bool, Optional[int]]:
        """Check if request is allowed"""
        current_time = time.time()
        window_start = current_time - self.config.window_size

        # Remove old requests
        requests = self.requests[key]
        while requests and requests[0] < window_start:
            requests.popleft()

        # Check if limit exceeded
        if len(requests) >= self.config.requests_per_minute:
            retry_after = int(requests[0] + self.config.window_size - current_time)
            return False, retry_after

        # Add current request
        requests.append(current_time)
        return True, None

class TokenBucketRateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.buckets = defaultdict(lambda: {
            "tokens": config.bucket_capacity,
            "last_refill": time.time()
        })

    def is_allowed(self, key: str) -> Tuple[bool, Optional[int]]:
        """Check if request is allowed"""
        current_time = time.time()
        bucket = self.buckets[key]

        # Refill tokens
        time_passed = current_time - bucket["last_refill"]
        tokens_to_add = time_passed * self.config.tokens_per_second
        bucket["tokens"] = min(
            self.config.bucket_capacity,
            bucket["tokens"] + tokens_to_add
        )
        bucket["last_refill"] = current_time

        # Check if tokens available
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True, None
        else:
            # Calculate retry time
            tokens_needed = 1 - bucket["tokens"]
            retry_after = int(tokens_needed / self.config.tokens_per_second)
            return False, retry_after

class LeakyBucketRateLimiter:
    """Leaky bucket rate limiter"""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.buckets = defaultdict(lambda: {
            "water": 0,
            "last_leak": time.time()
        })

    def is_allowed(self, key: str) -> Tuple[bool, Optional[int]]:
        """Check if request is allowed"""
        current_time = time.time()
        bucket = self.buckets[key]

        # Leak water
        time_passed = current_time - bucket["last_leak"]
        water_leaked = time_passed * self.config.tokens_per_second
        bucket["water"] = max(0, bucket["water"] - water_leaked)
        bucket["last_leak"] = current_time

        # Check if bucket has space
        if bucket["water"] < self.config.bucket_capacity:
            bucket["water"] += 1
            return True, None
        else:
            # Calculate retry time
            space_needed = bucket["water"] - self.config.bucket_capacity + 1
            retry_after = int(space_needed / self.config.tokens_per_second)
            return False, retry_after

class MultiTierRateLimiter:
    """Multi-tier rate limiter with different limits for different user types"""

    def __init__(self):
        self.limiters = {}
        self.user_tiers = {}

        # Configure different tiers
        self._configure_tiers()

    def _configure_tiers(self):
        """Configure rate limiting tiers"""
        # Free tier
        free_config = RateLimitConfig(
            requests_per_minute=10,
            requests_per_hour=100,
            requests_per_day=1000,
            strategy=RateLimitStrategy.SLIDING_WINDOW
        )
        self.limiters["free"] = self._create_limiter(free_config)

        # Basic tier
        basic_config = RateLimitConfig(
            requests_per_minute=60,
            requests_per_hour=1000,
            requests_per_day=10000,
            strategy=RateLimitStrategy.SLIDING_WINDOW
        )
        self.limiters["basic"] = self._create_limiter(basic_config)

        # Premium tier
        premium_config = RateLimitConfig(
            requests_per_minute=300,
            requests_per_hour=5000,
            requests_per_day=50000,
            strategy=RateLimitStrategy.TOKEN_BUCKET
        )
        self.limiters["premium"] = self._create_limiter(premium_config)

        # Enterprise tier
        enterprise_config = RateLimitConfig(
            requests_per_minute=1000,
            requests_per_hour=20000,
            requests_per_day=200000,
            strategy=RateLimitStrategy.TOKEN_BUCKET
        )
        self.limiters["enterprise"] = self._create_limiter(enterprise_config)

    def _create_limiter(self, config: RateLimitConfig):
        """Create rate limiter based on strategy"""
        if config.strategy == RateLimitStrategy.FIXED_WINDOW:
            return FixedWindowRateLimiter(config)
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return SlidingWindowRateLimiter(config)
        elif config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return TokenBucketRateLimiter(config)
        elif config.strategy == RateLimitStrategy.LEAKY_BUCKET:
            return LeakyBucketRateLimiter(config)
        else:
            raise ValueError(f"Unknown rate limiting strategy: {config.strategy}")

    def set_user_tier(self, user_id: str, tier: str):
        """Set user tier"""
        if tier not in self.limiters:
            raise ValueError(f"Unknown tier: {tier}")
        self.user_tiers[user_id] = tier

    def is_allowed(self, user_id: str, endpoint: str = "default") -> Tuple[bool, Optional[int]]:
        """Check if request is allowed for user"""
        tier = self.user_tiers.get(user_id, "free")
        limiter = self.limiters[tier]

        # Create key based on user and endpoint
        key = f"{user_id}:{endpoint}"
        return limiter.is_allowed(key)

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get rate limiting stats for user"""
        tier = self.user_tiers.get(user_id, "free")
        return {
            "tier": tier,
            "limits": {
                "requests_per_minute": self.limiters[tier].config.requests_per_minute,
                "requests_per_hour": self.limiters[tier].config.requests_per_hour,
                "requests_per_day": self.limiters[tier].config.requests_per_day
            }
        }

class AsyncRateLimiter:
    """Async rate limiter for use with FastAPI"""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.limiter = self._create_limiter(config)
        self.lock = asyncio.Lock()

    def _create_limiter(self, config: RateLimitConfig):
        """Create appropriate rate limiter"""
        if config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return SlidingWindowRateLimiter(config)
        elif config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return TokenBucketRateLimiter(config)
        else:
            return SlidingWindowRateLimiter(config)

    async def is_allowed(self, key: str) -> Tuple[bool, Optional[int]]:
        """Check if request is allowed (async)"""
        async with self.lock:
            return self.limiter.is_allowed(key)

    async def wait_if_needed(self, key: str, max_wait: int = 60):
        """Wait if rate limit is exceeded"""
        while True:
            allowed, retry_after = await self.is_allowed(key)
            if allowed:
                break

            if retry_after and retry_after > max_wait:
                raise RateLimitExceeded(
                    f"Rate limit exceeded. Retry after {retry_after} seconds",
                    retry_after
                )

            # Wait before retrying
            wait_time = min(retry_after or 1, max_wait)
            await asyncio.sleep(wait_time)

# Global rate limiter instance
global_rate_limiter = MultiTierRateLimiter()

def get_rate_limiter() -> MultiTierRateLimiter:
    """Get global rate limiter instance"""
    return global_rate_limiter

# Decorator for rate limiting
def rate_limit(requests_per_minute: int = 60, user_tier: str = "free"):
    """Decorator for rate limiting endpoints"""
    def decorator(self, func):
        async def wrapper(*args, **kwargs):
            # Extract user_id from request (implementation depends on auth system)
            user_id = kwargs.get('user_id', 'anonymous')

            # Check rate limit
            allowed, retry_after = global_rate_limiter.is_allowed(user_id)

            if not allowed:
                raise RateLimitExceeded(
                    f"Rate limit exceeded. Retry after {retry_after} seconds",
                    retry_after
                )

            return await func(*args, **kwargs)

        return wrapper
    return decorator

# Rate limiting middleware for FastAPI
class RateLimitMiddleware:
    """FastAPI middleware for rate limiting"""

    def __init__(self, app, rate_limiter: MultiTierRateLimiter = None):
        self.app = app
        self.rate_limiter = rate_limiter or global_rate_limiter

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Extract user_id from request
            user_id = self._extract_user_id(scope)

            # Check rate limit
            allowed, retry_after = self.rate_limiter.is_allowed(user_id)

            if not allowed:
                # Return rate limit exceeded response
                await self._send_rate_limit_response(send, retry_after)
                return

        await self.app(scope, receive, send)

    def _extract_user_id(self, scope) -> str:
        """Extract user ID from request scope"""
        # This would be implemented based on your authentication system
        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode()

        if auth_header.startswith("Bearer "):
            # Extract user ID from JWT token
            token = auth_header[7:]
            # In a real implementation, you would decode the JWT and extract user_id
            return "user_from_token"  # Placeholder

        return "anonymous"

    async def _send_rate_limit_response(self, send, retry_after: int):
        """Send rate limit exceeded response"""
        response = {
            "status": 429,
            "headers": [
                (b"content-type", b"application/json"),
                (b"retry-after", str(retry_after).encode())
            ],
            "body": {
                "error": "Rate limit exceeded",
                "retry_after": retry_after,
                "message": f"Too many requests. Retry after {retry_after} seconds."
            }
        }

        await send({
            "type": "http.response.start",
            "status": response["status"],
            "headers": response["headers"]
        })

        await send({
            "type": "http.response.body",
            "body": str(response["body"]).encode()
        })