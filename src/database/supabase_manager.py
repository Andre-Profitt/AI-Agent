"""
Supabase-specific database manager with enhanced features
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from uuid import uuid4

from src.database.connection_pool import DatabasePool
from src.utils.logging import get_logger
from src.services.circuit_breaker import CircuitBreaker

logger = get_logger(__name__)

class SupabaseManager:
    """Enhanced Supabase manager with connection pooling and monitoring"""
    
    def __init__(self, url: str, key: str, pool_size: int = 10, max_retries: int = 3):
        self.pool = DatabasePool(
            url=url,
            key=key,
            pool_size=pool_size,
            max_retries=max_retries
        )
        
        # Table names
        self.tables = {
            'messages': 'agent_messages',
            'trajectories': 'agent_trajectory_logs',
            'tools': 'tool_reliability_metrics',
            'sessions': 'user_sessions',
            'metrics': 'performance_metrics',
            'errors': 'error_logs'
        }
        
        # Circuit breaker for each table
        self.circuit_breakers = {
            table: CircuitBreaker(failure_threshold=5, recovery_timeout=60)
            for table in self.tables.values()
        }
        
        # Cache for frequently accessed data
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def initialize(self):
        """Initialize the database manager"""
        await self.pool.initialize()
        
        # Verify tables exist
        await self._verify_tables()
        
        logger.info("SupabaseManager initialized successfully")
    
    async def _verify_tables(self):
        """Verify all required tables exist"""
        for name, table in self.tables.items():
            try:
                # Try to query the table
                async with self.pool.acquire() as conn:
                    result = await conn.connection.table(table).select("count").limit(1).execute()
                logger.info(f"Table '{table}' verified")
            except Exception as e:
                logger.error(f"Table '{table}' verification failed: {e}")
                # In production, you might want to create the table here
    
    async def test_connection(self):
        """Test database connectivity"""
        try:
            async with self.pool.acquire() as conn:
                # Simple test query
                result = await conn.connection.table(self.tables['sessions']).select("count").execute()
                logger.info("Database connection test successful")
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    async def log_message(self, session_id: str, message: str, history: List) -> Optional[str]:
        """Log a message with circuit breaker protection"""
        table = self.tables['messages']
        
        if not self.circuit_breakers[table].can_execute():
            logger.warning(f"Circuit breaker open for table {table}")
            return None
        
        try:
            message_id = str(uuid4())
            
            async with self.pool.acquire() as conn:
                data = {
                    'id': message_id,
                    'session_id': session_id,
                    'message': message,
                    'history': json.dumps(history),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                result = await conn.connection.table(table).insert(data).execute()
                
            self.circuit_breakers[table].record_success()
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to log message: {e}")
            self.circuit_breakers[table].record_failure()
            return None
    
    async def log_trajectory(self, session_id: str, trajectory_data: Dict[str, Any]):
        """Log agent trajectory for analysis"""
        table = self.tables['trajectories']
        
        if not self.circuit_breakers[table].can_execute():
            return
        
        try:
            async with self.pool.acquire() as conn:
                data = {
                    'id': str(uuid4()),
                    'trajectory': json.dumps(trajectory_data),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                await conn.connection.table(table).insert(data).execute()
                
            self.circuit_breakers[table].record_success()
            
        except Exception as e:
            logger.error(f"Failed to log trajectory: {e}")
            self.circuit_breakers[table].record_failure()
    
    async def update_tool_metrics(self, tool_name: str, success: bool, duration: float):
        """Update tool reliability metrics"""
        table = self.tables['tools']
        
        try:
            async with self.pool.acquire() as conn:
                # First, try to get existing metrics
                result = await conn.connection.table(table).select("*").eq('tool_name', tool_name).execute()
                
                if result.data:
                    # Update existing record
                    existing = result.data[0]
                    total_calls = existing['total_calls'] + 1
                    success_count = existing['success_count'] + (1 if success else 0)
                    total_duration = existing['total_duration'] + duration
                    
                    update_data = {
                        'total_calls': total_calls,
                        'success_count': success_count,
                        'failure_count': total_calls - success_count,
                        'success_rate': success_count / total_calls,
                        'average_duration': total_duration / total_calls,
                        'total_duration': total_duration,
                        'last_updated': datetime.utcnow().isoformat()
                    }
                    
                    await conn.connection.table(table).update(update_data).eq('tool_name', tool_name).execute()
                else:
                    # Create new record
                    insert_data = {
                        'tool_name': tool_name,
                        'total_calls': 1,
                        'success_count': 1 if success else 0,
                        'failure_count': 0 if success else 1,
                        'success_rate': 1.0 if success else 0.0,
                        'average_duration': duration,
                        'total_duration': duration,
                        'last_updated': datetime.utcnow().isoformat()
                    }
                    
                    await conn.connection.table(table).insert(insert_data).execute()
                    
        except Exception as e:
            logger.error(f"Failed to update tool metrics: {e}")
    
    async def get_tool_reliability(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool reliability metrics with caching"""
        cache_key = f"tool_reliability_{tool_name}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                return cached_data
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.connection.table(self.tables['tools']).select("*").eq('tool_name', tool_name).execute()
                
                if result.data:
                    data = result.data[0]
                    # Update cache
                    self.cache[cache_key] = (data, datetime.now())
                    return data
                    
        except Exception as e:
            logger.error(f"Failed to get tool reliability: {e}")
        
        return None
    
    async def log_error(self, error_type: str, error_message: str, context: Dict[str, Any]):
        """Log errors for analysis"""
        table = self.tables['errors']
        
        try:
            async with self.pool.acquire() as conn:
                data = {
                    'id': str(uuid4()),
                    'error_type': error_type,
                    'error_message': error_message,
                    'context': json.dumps(context),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                await conn.connection.table(table).insert(data).execute()
                
        except Exception as e:
            logger.error(f"Failed to log error: {e}")
    
    async def log_metric(self, session_id: str, metric_type: str, value: float):
        """Log performance metrics"""
        table = self.tables['metrics']
        
        try:
            async with self.pool.acquire() as conn:
                data = {
                    'id': str(uuid4()),
                    'session_id': session_id,
                    'metric_type': metric_type,
                    'value': value,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                await conn.connection.table(table).insert(data).execute()
                
        except Exception as e:
            logger.error(f"Failed to log metric: {e}")
    
    async def get_session_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent session history"""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.connection.table(self.tables['messages'])\
                    .select("*")\
                    .eq('session_id', session_id)\
                    .order('timestamp', desc=True)\
                    .limit(limit)\
                    .execute()
                
                return result.data if result.data else []
                
        except Exception as e:
            logger.error(f"Failed to get session history: {e}")
            return []
    
    async def save_performance_stats(self, stats: Dict[str, Any]):
        """Save performance statistics"""
        try:
            async with self.pool.acquire() as conn:
                for tool_name, tool_stats in stats.items():
                    await self.update_tool_metrics(
                        tool_name=tool_name,
                        success=tool_stats['successes'] > 0,
                        duration=tool_stats['total_duration'] / max(tool_stats['calls'], 1)
                    )
        except Exception as e:
            logger.error(f"Failed to save performance stats: {e}")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return self.pool.get_stats()
    
    async def close(self):
        """Close database connections"""
        await self.pool.close() 