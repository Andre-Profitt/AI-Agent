"""
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
