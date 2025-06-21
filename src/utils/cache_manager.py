"""
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
