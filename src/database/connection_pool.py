from agent import query
from examples.enhanced_unified_example import start_time
from migrations.env import url

from src.core.monitoring import key
from src.database.connection_pool import conn
from src.database.connection_pool import connection
from src.database.connection_pool import idle_connections
from src.infrastructure.database_enhanced import pool
from src.query_classifier import params
from src.utils.logging import get_logger
from src.utils.tavily_search import max_retries

"""
import logging
from multiprocessing import connection
# TODO: Fix undefined variables: Any, Dict, List, Optional, asynccontextmanager, c, conn, connection, connection_id, connection_timeout, create_client, dataclass, datetime, e, i, idle_connections, key, max_overflow, max_retries, params, pool, pool_size, query, query_timeout, result, retry, retry_if_exception_type, start_time, stop_after_attempt, time, url, wait_exponential
from src.utils.structured_logging import get_logger

# TODO: Fix undefined variables: aiohttp, c, conn, connection_id, connection_timeout, create_client, e, get_logger, i, idle_connections, key, max_overflow, max_retries, params, pool, pool_size, query, query_timeout, result, retry, retry_if_exception_type, self, start_time, stop_after_attempt, url, wait_exponential
logger = logging.getLogger(__name__)

Database connection pooling with retry logic
"""

from typing import Dict
from typing import Any
from typing import List

import asyncio
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import time
from dataclasses import dataclass
from datetime import datetime
import aiohttp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from src.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ConnectionStats:
    """Track connection statistics"""
    created: int = 0
    active: int = 0
    idle: int = 0
    failed: int = 0
    total_wait_time: float = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None

class DatabaseConnection:
    """Wrapper for a database connection"""

    def __init__(self, connection_id: str, pool: "DatabasePool") -> None:
        self.id = connection_id
        self.pool = pool
        self.created_at = datetime.now()
        self.last_used = datetime.now()
        self.in_use = False
        self.connection = None

    async def execute(self, query: str, params: Optional[Dict] = None) -> Any:
        """Execute a query with automatic retry"""
        self.last_used = datetime.now()
        return await self.pool.execute_with_retry(self.connection, query, params)

    def release(self) -> Any:
        """Release connection back to pool"""
        self.in_use = False
        self.pool.release_connection(self)

class DatabasePool:
    """Connection pool with advanced features"""

    def __init__(
        self,
        url: str,
        key: str,
        pool_size: int = 10,
        max_overflow: int = 5,
        connection_timeout: float = 30.0,
        query_timeout: float = 60.0,
        max_retries: int = 3
    ):
        self.url = url
        self.key = key
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.connection_timeout = connection_timeout
        self.query_timeout = query_timeout
        self.max_retries = max_retries

        # Connection tracking
        self.connections: List[DatabaseConnection] = []
        self.available_connections: asyncio.Queue = asyncio.Queue()
        self.stats = ConnectionStats()

        # Health monitoring
        self._health_check_task = None
        self._closed = False

    async def initialize(self) -> Any:
        """Initialize the connection pool"""
        logger.info("Initializing database pool with {} connections...", extra={"self_pool_size": self.pool_size})

        # Create initial connections
        for i in range(self.pool_size):
            try:
                conn = await self._create_connection(f"conn_{i}")
                self.connections.append(conn)
                await self.available_connections.put(conn)
                self.stats.created += 1
                self.stats.idle += 1
            except Exception as e:
                logger.error("Failed to create connection {}: {}", extra={"i": i, "e": e})
                self.stats.failed += 1

        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info("Pool initialized with {} connections", extra={"len_self_connections_": len(self.connections)})

    async def _create_connection(self, connection_id: str) -> DatabaseConnection:
        """Create a new database connection"""
        conn = DatabaseConnection(connection_id, self)

        # Initialize the actual connection (Supabase client)
        from supabase import create_client
        conn.connection = create_client(self.url, self.key)

        return conn

    @asynccontextmanager
    async def acquire(self) -> Any:
        """Acquire a connection from the pool"""
        start_time = time.time()
        connection = None

        try:
            # Try to get an available connection
            try:
                connection = await asyncio.wait_for(
                    self.available_connections.get(),
                    timeout=self.connection_timeout
                )
            except asyncio.TimeoutError:
                # Try to create overflow connection
                if len(self.connections) < self.pool_size + self.max_overflow:
                    logger.warning("Pool exhausted, creating overflow connection")
                    connection = await self._create_connection(f"overflow_{len(self.connections)}")
                    self.connections.append(connection)
                    self.stats.created += 1
                else:
                    raise TimeoutError("Connection pool exhausted")

            # Mark as in use
            connection.in_use = True
            self.stats.active += 1
            self.stats.idle -= 1
            self.stats.total_wait_time += time.time() - start_time

            yield connection

        finally:
            # Release connection back to pool
            if connection:
                connection.in_use = False
                self.stats.active -= 1
                self.stats.idle += 1

                if not self._closed:
                    await self.available_connections.put(connection)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def execute_with_retry(self, connection, query: str, params: Optional[Dict] = None) -> Any:
        """Execute query with retry logic"""
        try:
            # Execute the query
            result = await asyncio.wait_for(
                connection.table(query).select("*").execute(),
                timeout=self.query_timeout
            )
            return result

        except Exception as e:
            logger.error("Query execution failed: {}", extra={"e": e})
            self.stats.last_error = str(e)
            self.stats.last_error_time = datetime.now()
            raise

    async def execute(self, query: str, params: Optional[Dict] = None) -> Any:
        """Execute a query using a connection from the pool"""
        async with self.acquire() as conn:
            return await conn.execute(query, params)

    async def _health_check_loop(self) -> Any:
        """Periodic health check of connections"""
        while not self._closed:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Test each idle connection
                idle_connections = [c for c in self.connections if not c.in_use]

                for conn in idle_connections:
                    try:
                        # Simple health check query
                        await conn.execute("health_check", {})
                    except Exception as e:
                        logger.warning("Health check failed for {}: {}", extra={"conn_id": conn.id, "e": e})
                        # Mark connection for recreation
                        # In production, implement connection recreation logic

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check loop error: {}", extra={"e": e})

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            'created': self.stats.created,
            'active': self.stats.active,
            'idle': self.stats.idle,
            'failed': self.stats.failed,
            'average_wait_time': (
                self.stats.total_wait_time / self.stats.created
                if self.stats.created > 0 else 0
            ),
            'last_error': self.stats.last_error,
            'last_error_time': (
                self.stats.last_error_time.isoformat()
                if self.stats.last_error_time else None
            ),
            'pool_size': self.pool_size,
            'max_overflow': self.max_overflow
        }

    async def close(self) -> None:
        """Close all connections in the pool"""
        logger.info("Closing database pool...")
        self._closed = True

        # Cancel health check
        if self._health_check_task:
            self._health_check_task.cancel()

        # Close all connections
        for conn in self.connections:
            try:
                # Supabase client doesn't need explicit close
                pass
            except Exception as e:
                logger.error("Error closing connection {}: {}", extra={"conn_id": conn.id, "e": e})

        logger.info("Database pool closed")
