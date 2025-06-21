"""
Rate limiting implementation for API endpoints
"""
from typing import Dict, Optional
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict, deque
from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(
        self, 
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        self.minute_buckets: Dict[str, deque] = defaultdict(lambda: deque(maxlen=requests_per_minute))
        self.hour_buckets: Dict[str, deque] = defaultdict(lambda: deque(maxlen=requests_per_hour))
        self._cleanup_task = None
        
    async def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limits"""
        now = datetime.utcnow()
        
        # Check minute limit
        minute_bucket = self.minute_buckets[identifier]
        minute_cutoff = now - timedelta(minutes=1)
        
        # Remove old entries
        while minute_bucket and minute_bucket[0] < minute_cutoff:
            minute_bucket.popleft()
            
        if len(minute_bucket) >= self.requests_per_minute:
            return False
            
        # Check hour limit
        hour_bucket = self.hour_buckets[identifier]
        hour_cutoff = now - timedelta(hours=1)
        
        while hour_bucket and hour_bucket[0] < hour_cutoff:
            hour_bucket.popleft()
            
        if len(hour_bucket) >= self.requests_per_hour:
            return False
            
        # Add current request
        minute_bucket.append(now)
        hour_bucket.append(now)
        
        return True
    
    async def __call__(self, request: Request) -> None:
        """FastAPI dependency for rate limiting"""
        # Get identifier (IP address or user ID)
        identifier = request.client.host
        
        # Check for authenticated user
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # Use user ID if authenticated
            # This would need to decode the JWT token
            identifier = f"user_{auth_header}"
            
        if not await self.check_rate_limit(identifier):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
            
    def start_cleanup(self):
        """Start periodic cleanup of old entries"""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
    async def _cleanup_loop(self):
        """Periodic cleanup of old entries"""
        while True:
            await asyncio.sleep(300)  # Clean up every 5 minutes
            now = datetime.utcnow()
            
            # Clean up minute buckets
            for key in list(self.minute_buckets.keys()):
                bucket = self.minute_buckets[key]
                cutoff = now - timedelta(minutes=2)
                if bucket and bucket[-1] < cutoff:
                    del self.minute_buckets[key]
                    
            # Clean up hour buckets  
            for key in list(self.hour_buckets.keys()):
                bucket = self.hour_buckets[key]
                cutoff = now - timedelta(hours=2)
                if bucket and bucket[-1] < cutoff:
                    del self.hour_buckets[key]

# Global rate limiter instances
general_limiter = RateLimiter(requests_per_minute=60, requests_per_hour=1000)
auth_limiter = RateLimiter(requests_per_minute=10, requests_per_hour=100)  # Stricter for auth
api_limiter = RateLimiter(requests_per_minute=30, requests_per_hour=500)

# Endpoint-specific limiters
ENDPOINT_LIMITS = {
    "/api/v1/auth/login": auth_limiter,
    "/api/v1/auth/register": auth_limiter,
    "/api/v1/agents/execute": api_limiter,
    "/api/v1/tools/execute": api_limiter,
}

async def get_rate_limiter(request: Request) -> RateLimiter:
    """Get appropriate rate limiter for endpoint"""
    path = request.url.path
    return ENDPOINT_LIMITS.get(path, general_limiter)
