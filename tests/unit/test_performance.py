"""
Performance Tests
Tests for caching, connection pooling, and resource limits
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
import aiohttp

from src.utils.cache_manager import CacheManager, CacheConfig
from src.utils.connection_pool import ConnectionPoolManager
from src.utils.resource_limiter import ResourceLimiter, ResourceConfig


class TestPerformanceOptimizations:
    """Test performance optimization components"""
    
    @pytest.fixture
    def cache_config(self):
        """Create test cache configuration"""
        return CacheConfig(
            memory_cache_size=100,
            memory_ttl=60,
            redis_url="redis://localhost:6379",
            redis_ttl=300,
            disk_cache_dir=".test_cache",
            enable_compression=True
        )
        
    @pytest.fixture
    def cache_manager(self, cache_config):
        """Create cache manager"""
        return CacheManager(cache_config)
        
    @pytest.mark.asyncio
    async def test_memory_cache(self, cache_manager):
        """Test in-memory caching"""
        # Set value
        await cache_manager.set("test_key", {"data": "test_value"})
        
        # Get value
        result = await cache_manager.get("test_key")
        assert result == {"data": "test_value"}
        
        # Test TTL
        await cache_manager.set("ttl_key", "value", ttl=1)
        await asyncio.sleep(1.5)
        result = await cache_manager.get("ttl_key")
        assert result is None
        
    @pytest.mark.asyncio
    async def test_cache_decorator(self, cache_manager):
        """Test cache decorator"""
        call_count = 0
        
        @cache_manager.cached(ttl=60)
        async def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return x * 2
            
        # First call
        result1 = await expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call (cached)
        result2 = await expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented
        
    @pytest.mark.asyncio
    async def test_connection_pool(self):
        """Test connection pool manager"""
        pool_manager = ConnectionPoolManager()
        
        # Test HTTP pool
        async with pool_manager.get_http_session() as session:
            assert isinstance(session, aiohttp.ClientSession)
            
        # Test connection reuse
        sessions = []
        for _ in range(5):
            async with pool_manager.get_http_session() as session:
                sessions.append(id(session))
                
        # Should reuse same session
        assert len(set(sessions)) == 1
        
    @pytest.mark.asyncio
    async def test_resource_limiter(self):
        """Test resource limiting"""
        config = ResourceConfig(
            max_concurrent_requests=2,
            max_requests_per_minute=10,
            max_memory_mb=100,
            max_cpu_percent=50
        )
        
        limiter = ResourceLimiter(config)
        
        # Test concurrent limit
        async def task():
            async with limiter.acquire("test_resource"):
                await asyncio.sleep(0.1)
                return True
                
        # Run 5 tasks with limit of 2
        start_time = time.time()
        results = await asyncio.gather(*[task() for _ in range(5)])
        end_time = time.time()
        
        assert all(results)
        # Should take at least 0.3s (3 batches of 0.1s)
        assert end_time - start_time >= 0.25
        
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting"""
        config = ResourceConfig(max_requests_per_minute=5)
        limiter = ResourceLimiter(config)
        
        # Make 5 requests quickly
        for i in range(5):
            assert await limiter.check_rate_limit("test_key")
            
        # 6th request should fail
        assert not await limiter.check_rate_limit("test_key")
        
    def test_cache_stats(self, cache_manager):
        """Test cache statistics"""
        # Generate some cache activity
        cache_manager.memory_cache.get("hit_key")  # Miss
        cache_manager.memory_cache.set("hit_key", "value")
        cache_manager.memory_cache.get("hit_key")  # Hit
        
        stats = cache_manager.get_stats()
        assert stats["memory"]["hits"] == 1
        assert stats["memory"]["misses"] == 1
        assert stats["memory"]["hit_rate"] == 0.5
        
    @pytest.mark.asyncio
    async def test_connection_pool_cleanup(self):
        """Test connection pool cleanup"""
        pool_manager = ConnectionPoolManager()
        
        # Create session
        async with pool_manager.get_http_session() as session:
            pass
            
        # Close pool
        await pool_manager.close()
        
        # Verify closed
        with pytest.raises(RuntimeError):
            async with pool_manager.get_http_session() as session:
                pass


class TestPerformanceIntegration:
    """Integration tests for performance features"""
    
    @pytest.mark.asyncio
    async def test_cached_api_calls(self):
        """Test caching of API calls"""
        cache_manager = CacheManager()
        pool_manager = ConnectionPoolManager()
        
        call_count = 0
        
        @cache_manager.cached(ttl=300)
        async def fetch_data(url: str) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            
            async with pool_manager.get_http_session() as session:
                # Mock response
                return {"data": f"Response for {url}", "timestamp": time.time()}
                
        # First call
        result1 = await fetch_data("https://api.example.com/data")
        assert call_count == 1
        
        # Second call (cached)
        result2 = await fetch_data("https://api.example.com/data")
        assert call_count == 1
        assert result1 == result2
        
        # Different URL
        result3 = await fetch_data("https://api.example.com/other")
        assert call_count == 2
        
        await pool_manager.close()
