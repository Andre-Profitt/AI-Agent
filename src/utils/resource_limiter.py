"""
Resource limiting and quota management
Prevents resource exhaustion and ensures fair usage
"""

from typing import Dict, Optional, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncio
import psutil
import resource
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ResourceType(str, Enum):
    """Types of resources to limit"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    API_CALLS = "api_calls"
    TOKENS = "tokens"
    CONCURRENT_REQUESTS = "concurrent_requests"
    
@dataclass
class ResourceQuota:
    """Resource quota definition"""
    resource_type: ResourceType
    limit: float
    window: timedelta = timedelta(minutes=1)
    burst_limit: Optional[float] = None
    
@dataclass
class ResourceUsage:
    """Track resource usage"""
    used: float = 0.0
    quota: float = 0.0
    window_start: datetime = field(default_factory=datetime.utcnow)
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def available(self) -> float:
        return max(0, self.quota - self.used)
        
    @property
    def usage_percent(self) -> float:
        return (self.used / self.quota * 100) if self.quota > 0 else 0
        
    def reset_if_expired(self, window: timedelta):
        """Reset usage if window expired"""
        if datetime.utcnow() - self.window_start > window:
            self.history.append({
                "used": self.used,
                "quota": self.quota,
                "timestamp": self.window_start
            })
            self.used = 0.0
            self.window_start = datetime.utcnow()

class SystemResourceMonitor:
    """Monitor system resource usage"""
    
    def __init__(self):
        self.process = psutil.Process()
        
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        return self.process.cpu_percent(interval=0.1)
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory = self.process.memory_info()
        return {
            "rss_mb": memory.rss / 1024 / 1024,
            "vms_mb": memory.vms / 1024 / 1024,
            "percent": self.process.memory_percent()
        }
        
    def get_disk_usage(self) -> Dict[str, float]:
        """Get disk usage for temp directory"""
        usage = psutil.disk_usage('/tmp')
        return {
            "used_gb": usage.used / 1024 / 1024 / 1024,
            "free_gb": usage.free / 1024 / 1024 / 1024,
            "percent": usage.percent
        }
        
    def get_network_usage(self) -> Dict[str, float]:
        """Get network I/O statistics"""
        io = psutil.net_io_counters()
        return {
            "bytes_sent_mb": io.bytes_sent / 1024 / 1024,
            "bytes_recv_mb": io.bytes_recv / 1024 / 1024,
            "packets_sent": io.packets_sent,
            "packets_recv": io.packets_recv
        }

class ResourceLimiter:
    """Resource limiting with quotas"""
    
    def __init__(self):
        self.quotas: Dict[str, Dict[ResourceType, ResourceQuota]] = defaultdict(dict)
        self.usage: Dict[str, Dict[ResourceType, ResourceUsage]] = defaultdict(dict)
        self.system_monitor = SystemResourceMonitor()
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._locks = defaultdict(asyncio.Lock)
        
    def set_quota(
        self, 
        user_id: str, 
        resource_type: ResourceType, 
        limit: float,
        window: timedelta = timedelta(minutes=1),
        burst_limit: Optional[float] = None
    ):
        """Set resource quota for user"""
        quota = ResourceQuota(
            resource_type=resource_type,
            limit=limit,
            window=window,
            burst_limit=burst_limit or limit * 1.5
        )
        
        self.quotas[user_id][resource_type] = quota
        
        if user_id not in self.usage:
            self.usage[user_id] = {}
            
        if resource_type not in self.usage[user_id]:
            self.usage[user_id][resource_type] = ResourceUsage(quota=limit)
            
        # Create semaphore for concurrent requests
        if resource_type == ResourceType.CONCURRENT_REQUESTS:
            self._semaphores[user_id] = asyncio.Semaphore(int(limit))
            
    async def check_quota(
        self, 
        user_id: str, 
        resource_type: ResourceType, 
        amount: float = 1.0
    ) -> bool:
        """Check if user has available quota"""
        async with self._locks[user_id]:
            if user_id not in self.quotas or resource_type not in self.quotas[user_id]:
                return True  # No quota set
                
            quota = self.quotas[user_id][resource_type]
            usage = self.usage[user_id][resource_type]
            
            # Reset if window expired
            usage.reset_if_expired(quota.window)
            
            # Check burst limit
            if usage.used + amount > quota.burst_limit:
                return False
                
            return True
            
    async def consume_quota(
        self, 
        user_id: str, 
        resource_type: ResourceType, 
        amount: float = 1.0
    ) -> bool:
        """Consume resource quota"""
        async with self._locks[user_id]:
            if not await self.check_quota(user_id, resource_type, amount):
                return False
                
            if user_id in self.usage and resource_type in self.usage[user_id]:
                usage = self.usage[user_id][resource_type]
                usage.used += amount
                
            return True
            
    async def acquire_concurrent_slot(self, user_id: str):
        """Acquire concurrent request slot"""
        if user_id in self._semaphores:
            return self._semaphores[user_id]
        return None
        
    def get_usage_stats(self, user_id: str) -> Dict[str, Any]:
        """Get usage statistics for user"""
        if user_id not in self.usage:
            return {}
            
        stats = {}
        for resource_type, usage in self.usage[user_id].items():
            if user_id in self.quotas and resource_type in self.quotas[user_id]:
                quota = self.quotas[user_id][resource_type]
                usage.reset_if_expired(quota.window)
                
            stats[resource_type.value] = {
                "used": usage.used,
                "quota": usage.quota,
                "available": usage.available,
                "usage_percent": usage.usage_percent,
                "window_start": usage.window_start.isoformat()
            }
            
        return stats
        
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system resource statistics"""
        return {
            "cpu": self.system_monitor.get_cpu_usage(),
            "memory": self.system_monitor.get_memory_usage(),
            "disk": self.system_monitor.get_disk_usage(),
            "network": self.system_monitor.get_network_usage()
        }

class ResourceMiddleware:
    """Middleware for resource limiting"""
    
    def __init__(self, limiter: ResourceLimiter):
        self.limiter = limiter
        
    async def __call__(self, request, call_next):
        """FastAPI middleware for resource limiting"""
        user_id = getattr(request.state, "user_id", "anonymous")
        
        # Check concurrent request limit
        semaphore = await self.limiter.acquire_concurrent_slot(user_id)
        
        if semaphore:
            async with semaphore:
                # Check API call quota
                if not await self.limiter.consume_quota(
                    user_id, 
                    ResourceType.API_CALLS, 
                    1.0
                ):
                    return JSONResponse(
                        status_code=429,
                        content={"error": "API quota exceeded"}
                    )
                    
                response = await call_next(request)
                return response
        else:
            response = await call_next(request)
            return response

# Process-level resource limits
def set_process_limits():
    """Set process-level resource limits"""
    try:
        # Set memory limit (1GB)
        resource.setrlimit(
            resource.RLIMIT_AS,
            (1024 * 1024 * 1024, 1024 * 1024 * 1024)
        )
        
        # Set CPU time limit (1 hour)
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (3600, 3600)
        )
        
        # Set max file descriptors
        resource.setrlimit(
            resource.RLIMIT_NOFILE,
            (4096, 4096)
        )
        
        logger.info("Set process resource limits")
        
    except Exception as e:
        logger.warning(f"Could not set resource limits: {e}")

# Token usage limiter for LLM calls
class TokenLimiter:
    """Limit token usage for LLM API calls"""
    
    def __init__(self, max_tokens_per_minute: int = 10000):
        self.max_tokens_per_minute = max_tokens_per_minute
        self.token_usage = deque()
        self._lock = asyncio.Lock()
        
    async def check_tokens(self, tokens: int) -> bool:
        """Check if tokens are available"""
        async with self._lock:
            now = datetime.utcnow()
            cutoff = now - timedelta(minutes=1)
            
            # Remove old entries
            while self.token_usage and self.token_usage[0][0] < cutoff:
                self.token_usage.popleft()
                
            # Calculate current usage
            current_usage = sum(t[1] for t in self.token_usage)
            
            return current_usage + tokens <= self.max_tokens_per_minute
            
    async def consume_tokens(self, tokens: int) -> bool:
        """Consume tokens if available"""
        if await self.check_tokens(tokens):
            async with self._lock:
                self.token_usage.append((datetime.utcnow(), tokens))
            return True
        return False
        
    def get_usage(self) -> Dict[str, Any]:
        """Get current token usage"""
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=1)
        
        # Calculate current usage
        current_usage = sum(
            t[1] for t in self.token_usage 
            if t[0] >= cutoff
        )
        
        return {
            "used": current_usage,
            "limit": self.max_tokens_per_minute,
            "available": self.max_tokens_per_minute - current_usage,
            "usage_percent": (current_usage / self.max_tokens_per_minute) * 100
        }

# Global resource limiter
_resource_limiter = None

def get_resource_limiter() -> ResourceLimiter:
    """Get global resource limiter"""
    global _resource_limiter
    if not _resource_limiter:
        _resource_limiter = ResourceLimiter()
        # Set default quotas
        _resource_limiter.set_quota(
            "default",
            ResourceType.API_CALLS,
            1000,
            timedelta(hours=1)
        )
        _resource_limiter.set_quota(
            "default",
            ResourceType.CONCURRENT_REQUESTS,
            10
        )
    return _resource_limiter

# Decorators for resource limiting
def limit_resources(
    resource_type: ResourceType,
    amount: float = 1.0,
    user_id_func: Optional[Callable] = None
):
    """Decorator for limiting resource usage"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get user ID
            if user_id_func:
                user_id = user_id_func(*args, **kwargs)
            else:
                user_id = kwargs.get("user_id", "anonymous")
                
            limiter = get_resource_limiter()
            
            # Check and consume quota
            if not await limiter.consume_quota(user_id, resource_type, amount):
                raise Exception(f"Resource quota exceeded for {resource_type.value}")
                
            return await func(*args, **kwargs)
            
        return wrapper
    return decorator
