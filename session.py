"""
Session Management Module
=========================

This module handles user sessions, parallel processing pools, and caching.
"""

import uuid
import time
import threading
import logging
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from queue import Queue
from functools import lru_cache
from dataclasses import dataclass, field
from datetime import datetime

from config import config

logger = logging.getLogger(__name__)


@dataclass
class SessionMetrics:
    """Metrics for a user session"""
    session_id: str
    created_at: float = field(default_factory=time.time)
    total_queries: int = 0
    cache_hits: int = 0
    total_response_time: float = 0.0
    tool_usage: Dict[str, int] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    parallel_executions: int = 0
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time"""
        if self.total_queries == 0:
            return 0.0
        return self.total_response_time / self.total_queries
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage"""
        if self.total_queries == 0:
            return 0.0
        return (self.cache_hits / self.total_queries) * 100
    
    @property
    def uptime_hours(self) -> float:
        """Calculate session uptime in hours"""
        return (time.time() - self.created_at) / 3600


class AsyncResponseCache:
    """Advanced response caching with TTL and intelligent invalidation"""
    
    def __init__(self, max_size: int = None, ttl_seconds: int = None):
        self.max_size = max_size or config.performance.CACHE_MAX_SIZE
        self.ttl_seconds = ttl_seconds or config.performance.CACHE_TTL_SECONDS
        self.cache = {}
        self.timestamps = {}
        self.lock = threading.RLock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
        self.cleanup_thread.start()
        logger.info(f"Initialized AsyncResponseCache with max_size={self.max_size}, ttl={self.ttl_seconds}s")
    
    def _cleanup_expired(self):
        """Background thread to clean up expired cache entries"""
        while True:
            try:
                time.sleep(60)  # Run cleanup every minute
                current_time = time.time()
                
                with self.lock:
                    expired_keys = [
                        key for key, timestamp in self.timestamps.items()
                        if current_time - timestamp > self.ttl_seconds
                    ]
                    
                    for key in expired_keys:
                        self.cache.pop(key, None)
                        self.timestamps.pop(key, None)
                        
                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                        
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        with self.lock:
            if key in self.cache:
                # Check if expired
                if time.time() - self.timestamps[key] <= self.ttl_seconds:
                    logger.debug(f"Cache hit for key: {key[:50]}...")
                    return self.cache[key]
                else:
                    # Remove expired entry
                    self.cache.pop(key, None)
                    self.timestamps.pop(key, None)
                    logger.debug(f"Cache miss (expired) for key: {key[:50]}...")
            
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache with TTL"""
        with self.lock:
            # Implement LRU eviction if cache is full
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.timestamps.keys(), key=lambda k: self.timestamps[k])
                self.cache.pop(oldest_key, None)
                self.timestamps.pop(oldest_key, None)
                logger.debug(f"Evicted oldest cache entry: {oldest_key[:50]}...")
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
            logger.debug(f"Cached value for key: {key[:50]}...")
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            logger.info("Cleared all cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "oldest_entry_age": min(
                    (time.time() - ts for ts in self.timestamps.values()),
                    default=0
                )
            }


class ParallelAgentPool:
    """High-performance parallel agent execution with worker pool"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = min(
            max_workers or config.performance.MAX_PARALLEL_WORKERS,
            config.performance.MAX_PARALLEL_WORKERS
        )
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.active_tasks = {}
        self.task_queue = Queue()
        self.response_cache = AsyncResponseCache()
        self.rate_limiter = RateLimiter()
        
        logger.info(f"Initialized ParallelAgentPool with {self.max_workers} workers")
    
    @lru_cache(maxsize=1000)
    def _get_cache_key(self, message: str, session_id: str) -> str:
        """Generate cache key for response caching"""
        return f"{hash(message)}_{session_id}"
    
    def get_cached_response(self, message: str, session_id: str) -> Optional[str]:
        """Get cached response if available"""
        cache_key = self._get_cache_key(message, session_id)
        return self.response_cache.get(cache_key)
    
    def cache_response(self, message: str, session_id: str, response: str):
        """Cache response for future use"""
        if "error" not in response.lower():
            cache_key = self._get_cache_key(message, session_id)
            self.response_cache.set(cache_key, response)
    
    def execute_agent_parallel(self, execute_func, message: str, history: List[List[str]], 
                             log_to_db: bool, session_id: str):
        """Execute agent with parallel processing and rate limiting"""
        
        # Check cache first
        cached_response = self.get_cached_response(message, session_id)
        if cached_response:
            logger.info(f"Cache hit for message: {message[:50]}...")
            yield "📋 **Retrieved from cache** (Ultra-fast response!)\n\n", cached_response, session_id
            return
        
        # Enforce rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Execute in thread pool
        logger.info(f"Executing in parallel pool (worker count: {self.max_workers})")
        future = self.executor.submit(execute_func, message, history, log_to_db, session_id)
        
        try:
            # Get the generator from the future
            generator = future.result(timeout=config.performance.LONG_RUNNING_TIMEOUT)
            
            final_response = ""
            for steps, response, updated_session_id in generator:
                final_response = response
                yield steps, response, updated_session_id
            
            # Cache the final response
            if final_response:
                self.cache_response(message, session_id, final_response)
                
        except concurrent.futures.TimeoutError:
            logger.error(f"Parallel execution timeout for message: {message[:50]}...")
            yield "❌ **Timeout Error:** Request took too long to process", "Request timeout", session_id
        except Exception as e:
            logger.error(f"Parallel execution error: {e}")
            yield f"❌ **Parallel Execution Error:** {e}", f"Error: {e}", session_id
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status for monitoring"""
        return {
            "max_workers": self.max_workers,
            "active_threads": threading.active_count(),
            "cache_stats": self.response_cache.get_stats(),
            "pending_tasks": self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else 0,
            "rate_limiter_status": self.rate_limiter.get_status()
        }
    
    def shutdown(self):
        """Shutdown the thread pool gracefully"""
        logger.info("Shutting down ParallelAgentPool...")
        self.executor.shutdown(wait=True)
        self.response_cache.clear()


class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, max_requests_per_minute: int = None, burst_allowance: int = 10):
        self.max_requests_per_minute = max_requests_per_minute or config.performance.MAX_REQUESTS_PER_MINUTE
        self.burst_allowance = burst_allowance
        self.requests = []
        self.lock = threading.Lock()
        self.total_requests = 0
        self.last_request_time = 0
    
    def wait_if_needed(self):
        """Wait if needed to comply with rate limits"""
        with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            if len(self.requests) >= self.max_requests_per_minute:
                sleep_time = 60 - (now - self.requests[0]) + config.performance.API_RATE_LIMIT_BUFFER
                if sleep_time > 0:
                    logger.info(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                    time.sleep(sleep_time)
            
            # Enforce minimum spacing between requests
            time_since_last = now - self.last_request_time
            if time_since_last < config.performance.REQUEST_SPACING:
                sleep_time = config.performance.REQUEST_SPACING - time_since_last
                time.sleep(sleep_time)
            
            self.requests.append(time.time())
            self.last_request_time = time.time()
            self.total_requests += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get rate limiter status"""
        with self.lock:
            return {
                "current_rpm": len(self.requests),
                "max_rpm": self.max_requests_per_minute,
                "total_requests": self.total_requests,
                "last_request_time": self.last_request_time
            }


class SessionManager:
    """Manages user sessions and analytics"""
    
    def __init__(self):
        self.sessions: Dict[str, SessionMetrics] = {}
        self.lock = threading.RLock()
        logger.info("Initialized SessionManager")
    
    def create_session(self, session_id: str = None) -> str:
        """Create a new session or return existing one"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = SessionMetrics(session_id=session_id)
                logger.info(f"Created new session: {session_id}")
            else:
                logger.debug(f"Using existing session: {session_id}")
        
        return session_id
    
    def update_query_metrics(self, session_id: str, response_time: float, 
                           cache_hit: bool = False, parallel: bool = False):
        """Update session metrics for a query"""
        with self.lock:
            if session_id in self.sessions:
                metrics = self.sessions[session_id]
                metrics.total_queries += 1
                metrics.total_response_time += response_time
                
                if cache_hit:
                    metrics.cache_hits += 1
                if parallel:
                    metrics.parallel_executions += 1
                    
                logger.debug(f"Updated metrics for session {session_id}: "
                           f"queries={metrics.total_queries}, "
                           f"avg_time={metrics.average_response_time:.2f}s")
    
    def update_tool_usage(self, session_id: str, tool_name: str):
        """Update tool usage statistics"""
        with self.lock:
            if session_id in self.sessions:
                metrics = self.sessions[session_id]
                metrics.tool_usage[tool_name] = metrics.tool_usage.get(tool_name, 0) + 1
    
    def add_error(self, session_id: str, error: Exception, context: Dict[str, Any] = None):
        """Log an error for a session"""
        with self.lock:
            if session_id in self.sessions:
                error_info = {
                    "timestamp": datetime.now().isoformat(),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "context": context or {}
                }
                self.sessions[session_id].errors.append(error_info)
                logger.error(f"Session {session_id} error: {error_info}")
    
    def get_session_metrics(self, session_id: str) -> Optional[SessionMetrics]:
        """Get metrics for a specific session"""
        with self.lock:
            return self.sessions.get(session_id)
    
    def get_global_analytics(self) -> Dict[str, Any]:
        """Get aggregated analytics across all sessions in a nested format."""
        with self.lock:
            total_sessions = len(self.sessions)
            if not self.sessions:
                analytics = self._empty_analytics()
                analytics["total_sessions"] = total_sessions
                return analytics

            total_queries = sum(s.total_queries for s in self.sessions.values())
            total_cache_hits = sum(s.cache_hits for s in self.sessions.values())
            total_response_time = sum(s.total_response_time for s in self.sessions.values())
            total_parallel = sum(s.parallel_executions for s in self.sessions.values())

            # Aggregate tool usage
            tool_usage = {}
            for session in self.sessions.values():
                for tool, count in session.tool_usage.items():
                    tool_usage[tool] = tool_usage.get(tool, 0) + count

            # Aggregate errors
            all_errors = [error for session in self.sessions.values() for error in session.errors]
            error_summary = {}
            for error in all_errors:
                error_type = error["error_type"]
                error_summary[error_type] = error_summary.get(error_type, 0) + 1

            analytics = {
                "performance": {
                    "total_queries": total_queries,
                    "avg_response_time": total_response_time / max(1, total_queries),
                    "parallel_executions": total_parallel,
                    "total_tool_calls": sum(tool_usage.values()),
                    "cache_hits": total_cache_hits,
                },
                "cache_efficiency": {
                    "cache_hits": total_cache_hits,
                    "hit_rate": (total_cache_hits / max(1, total_queries)) * 100,
                    "size": len(self.cache.cache) if hasattr(self, 'cache') else 0,
                },
                "active_sessions": total_sessions,
                "uptime_hours": max((s.uptime_hours for s in self.sessions.values()), default=0),
                "parallel_pool": {
                    "max_workers": config.performance.MAX_PARALLEL_WORKERS,
                    "active_threads": threading.active_count(),
                    "total_requests": sum(s.total_queries for s in self.sessions.values()),
                    "rate_limiting_active": True,
                },
                "tool_analytics": {
                    tool_name: {
                        "calls": count,
                        "successes": count,  # Placeholder - would need actual tracking
                        "avg_time": 0.5,  # Placeholder - would need actual tracking
                    }
                    for tool_name, count in tool_usage.items()
                },
                "error_analytics": {
                    "total_errors": len(all_errors),
                    "error_types": error_summary
                },
                "total_sessions": total_sessions,
            }
            # Add top-level keys for test compatibility
            analytics["total_queries"] = total_queries
            analytics["cache_hits"] = total_cache_hits
            analytics["parallel_executions"] = total_parallel
            analytics["avg_response_time"] = total_response_time / max(1, total_queries)
            analytics["cache_hit_rate"] = (total_cache_hits / max(1, total_queries)) * 100
            analytics["tool_usage"] = tool_usage
            return analytics

    def _empty_analytics(self) -> Dict[str, Any]:
        """Return empty nested analytics structure."""
        return {
            "performance": {
                "total_queries": 0,
                "avg_response_time": 0.0,
                "parallel_executions": 0,
                "total_tool_calls": 0,
                "cache_hits": 0,
            },
            "cache_efficiency": {
                "cache_hits": 0,
                "hit_rate": 0.0,
                "size": 0,
            },
            "active_sessions": 0,
            "uptime_hours": 0.0,
            "parallel_pool": {
                "max_workers": config.performance.MAX_PARALLEL_WORKERS,
                "active_threads": threading.active_count(),
                "total_requests": 0,
                "rate_limiting_active": True,
            },
            "tool_analytics": {},
            "error_analytics": {
                "total_errors": 0,
                "error_types": {}
            },
        }
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up sessions older than specified hours"""
        with self.lock:
            current_time = time.time()
            sessions_to_remove = []
            
            for session_id, metrics in self.sessions.items():
                age_hours = (current_time - metrics.created_at) / 3600
                if age_hours > max_age_hours:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.sessions[session_id]
                logger.info(f"Cleaned up old session: {session_id}")
            
            if sessions_to_remove:
                logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions") 