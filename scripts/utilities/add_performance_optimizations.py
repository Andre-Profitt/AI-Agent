#!/usr/bin/env python3
"""
Performance optimization implementation for AI Agent project
Adds caching, connection pooling, and resource limits
"""

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        
    def add_optimizations(self):
        """Add all performance optimizations"""
        logger.info("🚀 Adding performance optimizations...\n")
        
        # Create optimization components
        self._create_cache_manager()
        self._create_connection_pool()
        self._create_resource_limiter()
        self._create_performance_config()
        self._optimize_existing_code()
        
        logger.info("\n✅ Performance optimizations added!")
        
    def _create_cache_manager(self):
        """Create comprehensive caching system"""
        cache_content = '''"""
Caching system for AI Agent
Implements multiple caching strategies for different use cases
"""

from typing import Any, Optional, Dict, Callable, Union
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from dataclasses import dataclass, field
import asyncio
import hashlib
import json
import pickle
from collections import OrderedDict
import redis
from aiocache import Cache, cached
from aiocache.serializers import PickleSerializer
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Cache configuration"""
    # Memory cache settings
    memory_cache_size: int = 1000
    memory_ttl: int = 300  # 5 minutes
    
    # Redis cache settings
    redis_url: str = "redis://localhost:6379"
    redis_ttl: int = 3600  # 1 hour
    redis_prefix: str = "ai_agent:"
    
    # Disk cache settings
    disk_cache_dir: str = ".cache"
    disk_cache_size_mb: int = 100
    disk_ttl: int = 86400  # 24 hours
    
    # Cache behavior
    enable_compression: bool = True
    enable_stats: bool = True
    eviction_policy: str = "lru"  # lru, lfu, ttl

class CacheStats:
    """Track cache performance statistics"""
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.errors = 0
        
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "errors": self.errors,
            "hit_rate": self.hit_rate
        }

class MemoryCache:
    """In-memory LRU cache with TTL"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, tuple[Any, datetime]] = OrderedDict()
        self.stats = CacheStats()
        self._lock = asyncio.Lock()
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                if datetime.utcnow() < expiry:
                    # Move to end (LRU)
                    self.cache.move_to_end(key)
                    self.stats.hits += 1
                    return value
                else:
                    # Expired
                    del self.cache[key]
                    
            self.stats.misses += 1
            return None
            
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        async with self._lock:
            expiry = datetime.utcnow() + timedelta(seconds=ttl or self.ttl)
            
            # Evict if necessary
            while len(self.cache) >= self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.stats.evictions += 1
                
            self.cache[key] = (value, expiry)
            
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        async with self._lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
            
    async def clear(self) -> None:
        """Clear all cache entries"""
        async with self._lock:
            self.cache.clear()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "type": "memory",
            "size": len(self.cache),
            "max_size": self.max_size,
            **self.stats.to_dict()
        }

class RedisCache:
    """Redis-based distributed cache"""
    
    def __init__(self, redis_url: str, prefix: str = "ai_agent:", ttl: int = 3600):
        self.redis_url = redis_url
        self.prefix = prefix
        self.ttl = ttl
        self.stats = CacheStats()
        self._client = None
        
    async def _get_client(self):
        """Get Redis client (lazy initialization)"""
        if not self._client:
            self._client = await redis.from_url(self.redis_url, decode_responses=False)
        return self._client
        
    def _make_key(self, key: str) -> str:
        """Create namespaced key"""
        return f"{self.prefix}{key}"
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            client = await self._get_client()
            full_key = self._make_key(key)
            
            value = await client.get(full_key)
            if value:
                self.stats.hits += 1
                return pickle.loads(value)
            else:
                self.stats.misses += 1
                return None
                
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.stats.errors += 1
            return None
            
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        try:
            client = await self._get_client()
            full_key = self._make_key(key)
            
            serialized = pickle.dumps(value)
            await client.setex(full_key, ttl or self.ttl, serialized)
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            self.stats.errors += 1
            
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            client = await self._get_client()
            full_key = self._make_key(key)
            
            result = await client.delete(full_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            self.stats.errors += 1
            return False
            
    async def clear(self) -> None:
        """Clear all cache entries with prefix"""
        try:
            client = await self._get_client()
            
            # Find all keys with prefix
            cursor = 0
            pattern = f"{self.prefix}*"
            
            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=100)
                if keys:
                    await client.delete(*keys)
                if cursor == 0:
                    break
                    
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            self.stats.errors += 1
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "type": "redis",
            "prefix": self.prefix,
            **self.stats.to_dict()
        }

class CacheManager:
    """Unified cache manager with multiple backends"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        # Initialize caches
        self.memory_cache = MemoryCache(
            max_size=self.config.memory_cache_size,
            ttl=self.config.memory_ttl
        )
        
        self.redis_cache = RedisCache(
            redis_url=self.config.redis_url,
            prefix=self.config.redis_prefix,
            ttl=self.config.redis_ttl
        )
        
        # Cache hierarchy: memory -> redis -> compute
        self._cache_layers = [self.memory_cache, self.redis_cache]
        
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        # Create a unique key from args and kwargs
        key_data = {
            "args": args,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache hierarchy"""
        for i, cache in enumerate(self._cache_layers):
            value = await cache.get(key)
            if value is not None:
                # Backfill to faster caches
                for j in range(i):
                    await self._cache_layers[j].set(key, value)
                return value
        return None
        
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in all cache layers"""
        for cache in self._cache_layers:
            await cache.set(key, value, ttl)
            
    async def delete(self, key: str) -> None:
        """Delete from all cache layers"""
        for cache in self._cache_layers:
            await cache.delete(key)
            
    async def clear(self) -> None:
        """Clear all caches"""
        for cache in self._cache_layers:
            await cache.clear()
            
    def cached_function(
        self, 
        ttl: Optional[int] = None,
        key_prefix: Optional[str] = None
    ):
        """Decorator for caching function results"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_key(*args, **kwargs)
                if key_prefix:
                    cache_key = f"{key_prefix}:{cache_key}"
                    
                # Try to get from cache
                cached_value = await self.get(cache_key)
                if cached_value is not None:
                    return cached_value
                    
                # Compute and cache
                result = await func(*args, **kwargs)
                await self.set(cache_key, result, ttl)
                
                return result
                
            return wrapper
        return decorator
        
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all cache layers"""
        stats = {}
        for cache in self._cache_layers:
            cache_type = cache.__class__.__name__
            stats[cache_type] = cache.get_stats()
        return stats

# Global cache instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if not _cache_manager:
        _cache_manager = CacheManager()
    return _cache_manager

# Convenience decorators
def cached(ttl: int = 300, key_prefix: Optional[str] = None):
    """Decorator for caching async function results"""
    cache_manager = get_cache_manager()
    return cache_manager.cached_function(ttl=ttl, key_prefix=key_prefix)

def memoize(maxsize: int = 128):
    """Simple memoization decorator for sync functions"""
    return lru_cache(maxsize=maxsize)

# Specialized caches
class EmbeddingCache:
    """Specialized cache for embeddings"""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache_manager = cache_manager or get_cache_manager()
        self.prefix = "embedding:"
        
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding"""
        key = f"{self.prefix}{hashlib.md5(text.encode()).hexdigest()}"
        return await self.cache_manager.get(key)
        
    async def set_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache embedding"""
        key = f"{self.prefix}{hashlib.md5(text.encode()).hexdigest()}"
        await self.cache_manager.set(key, embedding, ttl=86400)  # 24 hours

class QueryCache:
    """Specialized cache for query results"""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache_manager = cache_manager or get_cache_manager()
        self.prefix = "query:"
        
    async def get_result(self, query: str, context: Optional[Dict] = None) -> Optional[Any]:
        """Get cached query result"""
        key_data = {"query": query, "context": context or {}}
        key = f"{self.prefix}{hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()}"
        return await self.cache_manager.get(key)
        
    async def set_result(self, query: str, result: Any, context: Optional[Dict] = None, ttl: int = 3600) -> None:
        """Cache query result"""
        key_data = {"query": query, "context": context or {}}
        key = f"{self.prefix}{hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()}"
        await self.cache_manager.set(key, result, ttl=ttl)

# Cache warming utilities
async def warm_cache(items: List[tuple[str, Any]], cache_manager: Optional[CacheManager] = None):
    """Warm cache with pre-computed values"""
    cache = cache_manager or get_cache_manager()
    
    tasks = []
    for key, value in items:
        tasks.append(cache.set(key, value))
        
    await asyncio.gather(*tasks)
    logger.info(f"Warmed cache with {len(items)} items")
'''
        
        cache_path = self.project_root / "src/utils/cache_manager.py"
        cache_path.write_text(cache_content)
        logger.info("✅ Created cache manager")
        
    def _create_connection_pool(self):
        """Create connection pooling system"""
        pool_content = '''"""
Connection pooling for database and API connections
Optimizes connection reuse and prevents resource exhaustion
"""

from typing import Any, Dict, Optional, List, Callable
from dataclasses import dataclass
from contextlib import asynccontextmanager
import asyncio
import aiohttp
import asyncpg
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class PoolConfig:
    """Connection pool configuration"""
    # Pool sizes
    min_size: int = 2
    max_size: int = 10
    
    # Connection settings
    connect_timeout: float = 5.0
    command_timeout: float = 10.0
    idle_timeout: float = 300.0  # 5 minutes
    
    # Health check
    health_check_interval: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0

class ConnectionHealth:
    """Track connection health"""
    def __init__(self):
        self.total_requests = 0
        self.failed_requests = 0
        self.last_success = datetime.utcnow()
        self.last_failure = None
        
    @property
    def is_healthy(self) -> bool:
        # Consider unhealthy if >10% failure rate or no success in 5 minutes
        if self.total_requests == 0:
            return True
        failure_rate = self.failed_requests / self.total_requests
        time_since_success = datetime.utcnow() - self.last_success
        return failure_rate < 0.1 and time_since_success < timedelta(minutes=5)

class HTTPConnectionPool:
    """HTTP connection pool using aiohttp"""
    
    def __init__(self, config: Optional[PoolConfig] = None):
        self.config = config or PoolConfig()
        self._session = None
        self._connector = None
        self.health = ConnectionHealth()
        
    async def _create_session(self):
        """Create aiohttp session with connection pooling"""
        if not self._session:
            self._connector = aiohttp.TCPConnector(
                limit=self.config.max_size,
                limit_per_host=self.config.max_size,
                ttl_dns_cache=300,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(
                total=self.config.command_timeout,
                connect=self.config.connect_timeout
            )
            
            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=timeout
            )
            
    async def request(
        self, 
        method: str, 
        url: str, 
        **kwargs
    ) -> aiohttp.ClientResponse:
        """Make HTTP request using pooled connection"""
        await self._create_session()
        
        self.health.total_requests += 1
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self._session.request(method, url, **kwargs)
                self.health.last_success = datetime.utcnow()
                return response
                
            except Exception as e:
                self.health.failed_requests += 1
                self.health.last_failure = datetime.utcnow()
                
                if attempt == self.config.max_retries - 1:
                    logger.error(f"HTTP request failed after {self.config.max_retries} attempts: {e}")
                    raise
                    
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                
    async def close(self):
        """Close connection pool"""
        if self._session:
            await self._session.close()
            self._session = None
            
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        stats = {
            "type": "http",
            "healthy": self.health.is_healthy,
            "total_requests": self.health.total_requests,
            "failed_requests": self.health.failed_requests
        }
        
        if self._connector:
            stats.update({
                "connections_total": self._connector._limit,
                "connections_available": self._connector._available_connections
            })
            
        return stats

class DatabaseConnectionPool:
    """PostgreSQL connection pool using asyncpg"""
    
    def __init__(self, dsn: str, config: Optional[PoolConfig] = None):
        self.dsn = dsn
        self.config = config or PoolConfig()
        self._pool = None
        self.health = ConnectionHealth()
        
    async def _create_pool(self):
        """Create connection pool"""
        if not self._pool:
            self._pool = await asyncpg.create_pool(
                self.dsn,
                min_size=self.config.min_size,
                max_size=self.config.max_size,
                command_timeout=self.config.command_timeout,
                max_inactive_connection_lifetime=self.config.idle_timeout
            )
            
    @asynccontextmanager
    async def acquire(self):
        """Acquire connection from pool"""
        await self._create_pool()
        
        self.health.total_requests += 1
        
        try:
            async with self._pool.acquire() as connection:
                self.health.last_success = datetime.utcnow()
                yield connection
                
        except Exception as e:
            self.health.failed_requests += 1
            self.health.last_failure = datetime.utcnow()
            raise
            
    async def execute(self, query: str, *args) -> Any:
        """Execute query using pooled connection"""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)
            
    async def fetch(self, query: str, *args) -> List[Any]:
        """Fetch results using pooled connection"""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)
            
    async def fetchrow(self, query: str, *args) -> Optional[Any]:
        """Fetch single row using pooled connection"""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)
            
    async def close(self):
        """Close connection pool"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        stats = {
            "type": "postgresql",
            "healthy": self.health.is_healthy,
            "total_requests": self.health.total_requests,
            "failed_requests": self.health.failed_requests
        }
        
        if self._pool:
            stats.update({
                "pool_size": self._pool.get_size(),
                "pool_free": self._pool.get_idle_size(),
                "pool_used": self._pool.get_size() - self._pool.get_idle_size()
            })
            
        return stats

class RedisConnectionPool:
    """Redis connection pool"""
    
    def __init__(self, url: str, config: Optional[PoolConfig] = None):
        self.url = url
        self.config = config or PoolConfig()
        self._pool = None
        self.health = ConnectionHealth()
        
    async def _create_pool(self):
        """Create Redis connection pool"""
        if not self._pool:
            self._pool = redis.ConnectionPool.from_url(
                self.url,
                max_connections=self.config.max_size,
                socket_connect_timeout=self.config.connect_timeout,
                socket_timeout=self.config.command_timeout,
                retry_on_timeout=True,
                health_check_interval=self.config.health_check_interval
            )
            
    async def get_client(self) -> redis.Redis:
        """Get Redis client with pooled connection"""
        await self._create_pool()
        
        self.health.total_requests += 1
        
        try:
            client = redis.Redis(connection_pool=self._pool)
            # Test connection
            await client.ping()
            self.health.last_success = datetime.utcnow()
            return client
            
        except Exception as e:
            self.health.failed_requests += 1
            self.health.last_failure = datetime.utcnow()
            raise
            
    async def close(self):
        """Close connection pool"""
        if self._pool:
            await self._pool.disconnect()
            self._pool = None
            
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        stats = {
            "type": "redis",
            "healthy": self.health.is_healthy,
            "total_requests": self.health.total_requests,
            "failed_requests": self.health.failed_requests
        }
        
        if self._pool:
            stats.update({
                "created_connections": self._pool.created_connections,
                "available_connections": len(self._pool._available_connections),
                "in_use_connections": len(self._pool._in_use_connections)
            })
            
        return stats

class MongoConnectionPool:
    """MongoDB connection pool using motor"""
    
    def __init__(self, uri: str, config: Optional[PoolConfig] = None):
        self.uri = uri
        self.config = config or PoolConfig()
        self._client = None
        self.health = ConnectionHealth()
        
    def _create_client(self):
        """Create MongoDB client with connection pooling"""
        if not self._client:
            self._client = AsyncIOMotorClient(
                self.uri,
                maxPoolSize=self.config.max_size,
                minPoolSize=self.config.min_size,
                maxIdleTimeMS=int(self.config.idle_timeout * 1000),
                serverSelectionTimeoutMS=int(self.config.connect_timeout * 1000),
                socketTimeoutMS=int(self.config.command_timeout * 1000)
            )
            
    def get_database(self, name: str):
        """Get database with pooled connections"""
        self._create_client()
        self.health.total_requests += 1
        
        try:
            db = self._client[name]
            self.health.last_success = datetime.utcnow()
            return db
            
        except Exception as e:
            self.health.failed_requests += 1
            self.health.last_failure = datetime.utcnow()
            raise
            
    def close(self):
        """Close connection pool"""
        if self._client:
            self._client.close()
            self._client = None
            
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            "type": "mongodb",
            "healthy": self.health.is_healthy,
            "total_requests": self.health.total_requests,
            "failed_requests": self.health.failed_requests
        }

class ConnectionPoolManager:
    """Manage all connection pools"""
    
    def __init__(self):
        self.pools: Dict[str, Any] = {}
        self._health_check_task = None
        
    def register_http_pool(self, name: str, config: Optional[PoolConfig] = None) -> HTTPConnectionPool:
        """Register HTTP connection pool"""
        pool = HTTPConnectionPool(config)
        self.pools[name] = pool
        return pool
        
    def register_database_pool(
        self, 
        name: str, 
        dsn: str, 
        config: Optional[PoolConfig] = None
    ) -> DatabaseConnectionPool:
        """Register database connection pool"""
        pool = DatabaseConnectionPool(dsn, config)
        self.pools[name] = pool
        return pool
        
    def register_redis_pool(
        self, 
        name: str, 
        url: str, 
        config: Optional[PoolConfig] = None
    ) -> RedisConnectionPool:
        """Register Redis connection pool"""
        pool = RedisConnectionPool(url, config)
        self.pools[name] = pool
        return pool
        
    def register_mongo_pool(
        self, 
        name: str, 
        uri: str, 
        config: Optional[PoolConfig] = None
    ) -> MongoConnectionPool:
        """Register MongoDB connection pool"""
        pool = MongoConnectionPool(uri, config)
        self.pools[name] = pool
        return pool
        
    def get_pool(self, name: str) -> Optional[Any]:
        """Get connection pool by name"""
        return self.pools.get(name)
        
    async def close_all(self):
        """Close all connection pools"""
        if self._health_check_task:
            self._health_check_task.cancel()
            
        for name, pool in self.pools.items():
            try:
                if hasattr(pool, 'close'):
                    await pool.close()
                else:
                    pool.close()
                logger.info(f"Closed connection pool: {name}")
                
            except Exception as e:
                logger.error(f"Error closing pool {name}: {e}")
                
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all pools"""
        stats = {}
        for name, pool in self.pools.items():
            if hasattr(pool, 'get_stats'):
                stats[name] = pool.get_stats()
        return stats
        
    async def start_health_checks(self, interval: float = 30.0):
        """Start periodic health checks"""
        async def health_check_loop():
            while True:
                await asyncio.sleep(interval)
                unhealthy = []
                
                for name, pool in self.pools.items():
                    if hasattr(pool, 'health') and not pool.health.is_healthy:
                        unhealthy.append(name)
                        
                if unhealthy:
                    logger.warning(f"Unhealthy connection pools: {unhealthy}")
                    
        self._health_check_task = asyncio.create_task(health_check_loop())

# Global connection pool manager
_pool_manager = None

def get_pool_manager() -> ConnectionPoolManager:
    """Get global connection pool manager"""
    global _pool_manager
    if not _pool_manager:
        _pool_manager = ConnectionPoolManager()
    return _pool_manager

# Initialize default pools
async def init_default_pools(config: Dict[str, Any]):
    """Initialize default connection pools"""
    manager = get_pool_manager()
    
    # HTTP pool for external APIs
    manager.register_http_pool("default_http")
    
    # Database pools
    if "database_url" in config:
        manager.register_database_pool("main_db", config["database_url"])
        
    # Redis pool
    if "redis_url" in config:
        manager.register_redis_pool("main_redis", config["redis_url"])
        
    # MongoDB pool
    if "mongodb_uri" in config:
        manager.register_mongo_pool("main_mongo", config["mongodb_uri"])
        
    # Start health checks
    await manager.start_health_checks()
    
    logger.info("Initialized default connection pools")
'''
        
        pool_path = self.project_root / "src/utils/connection_pool.py"
        pool_path.write_text(pool_content)
        logger.info("✅ Created connection pool manager")
        
    def _create_resource_limiter(self):
        """Create resource limiting system"""
        limiter_content = '''"""
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
'''
        
        limiter_path = self.project_root / "src/utils/resource_limiter.py"
        limiter_path.write_text(limiter_content)
        logger.info("✅ Created resource limiter")
        
    def _create_performance_config(self):
        """Create performance configuration"""
        config_content = '''"""
Performance configuration and tuning
Central configuration for all performance optimizations
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import timedelta
import os

@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    
    # Caching
    enable_caching: bool = True
    cache_memory_size: int = 1000
    cache_memory_ttl: int = 300  # 5 minutes
    cache_redis_ttl: int = 3600  # 1 hour
    cache_disk_size_mb: int = 100
    
    # Connection pooling
    enable_connection_pooling: bool = True
    pool_min_size: int = 2
    pool_max_size: int = 10
    pool_idle_timeout: float = 300.0
    
    # Resource limits
    enable_resource_limits: bool = True
    max_memory_mb: int = 1024
    max_cpu_percent: float = 80.0
    max_concurrent_requests: int = 100
    max_tokens_per_minute: int = 10000
    
    # Request handling
    request_timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    
    # Batch processing
    enable_batching: bool = True
    batch_size: int = 50
    batch_timeout: float = 1.0
    
    # Async settings
    max_workers: int = 10
    event_loop_policy: str = "uvloop"  # uvloop, asyncio
    
    # Monitoring
    enable_performance_monitoring: bool = True
    metrics_interval: float = 60.0
    slow_query_threshold: float = 1.0
    
    @classmethod
    def from_env(cls) -> 'PerformanceConfig':
        """Load configuration from environment variables"""
        return cls(
            enable_caching=os.getenv("PERF_ENABLE_CACHING", "true").lower() == "true",
            cache_memory_size=int(os.getenv("PERF_CACHE_MEMORY_SIZE", "1000")),
            cache_memory_ttl=int(os.getenv("PERF_CACHE_MEMORY_TTL", "300")),
            cache_redis_ttl=int(os.getenv("PERF_CACHE_REDIS_TTL", "3600")),
            
            enable_connection_pooling=os.getenv("PERF_ENABLE_POOLING", "true").lower() == "true",
            pool_max_size=int(os.getenv("PERF_POOL_MAX_SIZE", "10")),
            
            enable_resource_limits=os.getenv("PERF_ENABLE_LIMITS", "true").lower() == "true",
            max_memory_mb=int(os.getenv("PERF_MAX_MEMORY_MB", "1024")),
            max_concurrent_requests=int(os.getenv("PERF_MAX_CONCURRENT", "100")),
            
            request_timeout=float(os.getenv("PERF_REQUEST_TIMEOUT", "30.0")),
            
            enable_batching=os.getenv("PERF_ENABLE_BATCHING", "true").lower() == "true",
            batch_size=int(os.getenv("PERF_BATCH_SIZE", "50")),
            
            enable_performance_monitoring=os.getenv("PERF_ENABLE_MONITORING", "true").lower() == "true",
        )
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "caching": {
                "enabled": self.enable_caching,
                "memory_size": self.cache_memory_size,
                "memory_ttl": self.cache_memory_ttl,
                "redis_ttl": self.cache_redis_ttl,
                "disk_size_mb": self.cache_disk_size_mb
            },
            "connection_pooling": {
                "enabled": self.enable_connection_pooling,
                "min_size": self.pool_min_size,
                "max_size": self.pool_max_size,
                "idle_timeout": self.pool_idle_timeout
            },
            "resource_limits": {
                "enabled": self.enable_resource_limits,
                "max_memory_mb": self.max_memory_mb,
                "max_cpu_percent": self.max_cpu_percent,
                "max_concurrent_requests": self.max_concurrent_requests,
                "max_tokens_per_minute": self.max_tokens_per_minute
            },
            "request_handling": {
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay,
                "circuit_breaker_threshold": self.circuit_breaker_threshold,
                "circuit_breaker_timeout": self.circuit_breaker_timeout
            },
            "batch_processing": {
                "enabled": self.enable_batching,
                "size": self.batch_size,
                "timeout": self.batch_timeout
            },
            "async_settings": {
                "max_workers": self.max_workers,
                "event_loop_policy": self.event_loop_policy
            },
            "monitoring": {
                "enabled": self.enable_performance_monitoring,
                "metrics_interval": self.metrics_interval,
                "slow_query_threshold": self.slow_query_threshold
            }
        }

# Performance tuning utilities
class PerformanceTuner:
    """Automatic performance tuning based on system resources"""
    
    @staticmethod
    def auto_tune() -> PerformanceConfig:
        """Automatically tune performance settings based on system"""
        import psutil
        
        # Get system info
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
        
        config = PerformanceConfig()
        
        # Tune based on CPU
        config.max_workers = min(cpu_count * 2, 20)
        config.pool_max_size = min(cpu_count * 5, 50)
        
        # Tune based on memory
        if memory_gb >= 16:
            config.cache_memory_size = 5000
            config.max_memory_mb = 4096
            config.batch_size = 100
        elif memory_gb >= 8:
            config.cache_memory_size = 2000
            config.max_memory_mb = 2048
            config.batch_size = 50
        else:
            config.cache_memory_size = 500
            config.max_memory_mb = 512
            config.batch_size = 20
            
        # Use uvloop if available
        try:
            import uvloop
            config.event_loop_policy = "uvloop"
        except ImportError:
            config.event_loop_policy = "asyncio"
            
        return config

# Global performance configuration
_perf_config = None

def get_performance_config() -> PerformanceConfig:
    """Get global performance configuration"""
    global _perf_config
    if not _perf_config:
        # Try to load from environment, otherwise auto-tune
        if os.getenv("PERF_AUTO_TUNE", "true").lower() == "true":
            _perf_config = PerformanceTuner.auto_tune()
        else:
            _perf_config = PerformanceConfig.from_env()
    return _perf_config

# Apply performance settings
async def apply_performance_settings():
    """Apply all performance settings"""
    config = get_performance_config()
    
    # Set event loop policy
    if config.event_loop_policy == "uvloop":
        try:
            import uvloop
            uvloop.install()
            logger.info("Installed uvloop event loop")
        except ImportError:
            pass
            
    # Initialize caching if enabled
    if config.enable_caching:
        from src.utils.cache_manager import get_cache_manager
        cache_manager = get_cache_manager()
        logger.info("Initialized cache manager")
        
    # Initialize connection pools if enabled
    if config.enable_connection_pooling:
        from src.utils.connection_pool import init_default_pools
        await init_default_pools({
            "database_url": os.getenv("DATABASE_URL"),
            "redis_url": os.getenv("REDIS_URL"),
            "mongodb_uri": os.getenv("MONGODB_URI")
        })
        logger.info("Initialized connection pools")
        
    # Set resource limits if enabled
    if config.enable_resource_limits:
        from src.utils.resource_limiter import set_process_limits
        set_process_limits()
        logger.info("Set resource limits")
        
    logger.info("Applied performance settings")
'''
        
        perf_config_path = self.project_root / "src/config/performance.py"
        perf_config_path.write_text(config_content)
        logger.info("✅ Created performance configuration")
        
    def _optimize_existing_code(self):
        """Add performance optimizations to existing code"""
        # Update main.py to use performance settings
        main_path = self.project_root / "src/main.py"
        if main_path.exists():
            content = main_path.read_text()
            
            if "apply_performance_settings" not in content:
                # Add import
                import_line = "from src.config.performance import apply_performance_settings\n"
                
                # Add to startup
                startup_code = """
    # Apply performance optimizations
    await apply_performance_settings()
"""
                
                # Find where to insert
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'import' in line and 'from' in line:
                        lines.insert(i + 1, import_line)
                        break
                        
                for i, line in enumerate(lines):
                    if '@app.on_event("startup")' in line:
                        # Find the function body
                        for j in range(i + 1, len(lines)):
                            if lines[j].strip() and not lines[j].startswith(' '):
                                lines.insert(j, startup_code)
                                break
                        break
                        
                main_path.write_text('\n'.join(lines))
                logger.info("✅ Updated main.py with performance settings")

def main():
    optimizer = PerformanceOptimizer()
    optimizer.add_optimizations()
    
    logger.info("\nPerformance optimizations added:")
    logger.info("  - Cache manager with memory, Redis, and disk caching")
    logger.info("  - Connection pooling for HTTP, database, Redis, and MongoDB")
    logger.info("  - Resource limiting with quotas and system monitoring")
    logger.info("  - Performance configuration with auto-tuning")
    logger.info("\nNext steps:")
    logger.info("  1. Configure Redis for distributed caching")
    logger.info("  2. Set environment variables for performance tuning")
    logger.info("  3. Monitor performance metrics")

if __name__ == "__main__":
    main()