from agent import query
from agent import tools
from benchmarks.cot_performance import duration
from benchmarks.cot_performance import semaphore
from benchmarks.cot_performance import times
from examples.enhanced_unified_example import start_time
from examples.enhanced_unified_example import task
from examples.enhanced_unified_example import tasks
from examples.parallel_execution_example import items
from examples.parallel_execution_example import operations
from examples.parallel_execution_example import results
from tests.load_test import args
from tests.load_test import success
from tests.unit.simple_test import func

from src.api_server import resource
from src.application.tools.tool_executor import operation
from src.core.langgraph_compatibility import batch
from src.core.langgraph_compatibility import loop
from src.core.llamaindex_enhanced import embedding_model
from src.core.monitoring import key
from src.database.connection_pool import connection
from src.database.models import parameters
from src.database.models import text
from src.database.models import tool
from src.database.supabase_manager import cache_key
from src.gaia_components.advanced_reasoning_engine import embedding
from src.gaia_components.advanced_reasoning_engine import embeddings
from src.gaia_components.advanced_reasoning_engine import hash_val
from src.gaia_components.enhanced_memory_system import item
from src.gaia_components.monitoring import memory_info
from src.gaia_components.monitoring import process
from src.gaia_components.performance_optimization import after_mb
from src.gaia_components.performance_optimization import all_embeddings
from src.gaia_components.performance_optimization import batch_results
from src.gaia_components.performance_optimization import batch_size
from src.gaia_components.performance_optimization import before_mb
from src.gaia_components.performance_optimization import cache_hit_rate
from src.gaia_components.performance_optimization import cache_size_before
from src.gaia_components.performance_optimization import collected
from src.gaia_components.performance_optimization import durations
from src.gaia_components.performance_optimization import entry
from src.gaia_components.performance_optimization import fallback_embeddings
from src.gaia_components.performance_optimization import key_data
from src.gaia_components.performance_optimization import max_workers
from src.gaia_components.performance_optimization import new_soft
from src.gaia_components.performance_optimization import op_type
from src.gaia_components.performance_optimization import operation_groups
from src.gaia_components.performance_optimization import operation_stats
from src.gaia_components.performance_optimization import optimizations
from src.gaia_components.performance_optimization import optimized
from src.gaia_components.performance_optimization import optimizer_stats
from src.gaia_components.performance_optimization import pool_stats
from src.gaia_components.performance_optimization import remove_count
from src.gaia_components.performance_optimization import sorted_keys
from src.gaia_components.performance_optimization import successes
from src.gaia_components.performance_optimization import valid_results
from src.gaia_components.production_vector_store import texts
from src.tools_introspection import error
from src.utils.structured_logging import op_name

"""
import datetime
from typing import Any
from datetime import datetime
from multiprocessing import connection
# TODO: Fix undefined variables: Any, Callable, Dict, List, ProcessPoolExecutor, after_mb, all_embeddings, args, arr, arrays, batch, batch_results, batch_size, before_mb, cache_hit_rate, cache_key, cache_size_before, collected, connection, connection_timeout, datetime, duration, durations, e, embedding, embedding_model, embeddings, entry, error, fallback_embeddings, func, func_name, gc, hard, hash_val, i, item, items, k, key, key_data, kwargs, logging, loop, max_connections, max_process_workers, max_size, max_thread_workers, max_workers, memory_info, new_soft, op, op_name, op_type, operation, operation_groups, operation_name, operation_stats, operations, ops, optimizations, optimized, optimizer_stats, os, parameters, pool_stats, process, processor, query, r, remove_count, resource, result, results, semaphore, sorted_keys, start_time, success, successes, t, task, tasks, text, texts, time, times, tools, ttl, valid_results
from concurrent.futures import ThreadPoolExecutor

from src.tools.base_tool import tool

# TODO: Fix undefined variables: ProcessPoolExecutor, ThreadPoolExecutor, after_mb, aiohttp, all_embeddings, args, arr, arrays, batch, batch_results, batch_size, before_mb, cache_hit_rate, cache_key, cache_size_before, collected, connection_timeout, duration, durations, e, embedding, embedding_model, embeddings, entry, error, fallback_embeddings, func, func_name, functools, gc, hard, hash_val, hashlib, i, item, items, k, key, key_data, kwargs, loop, max_connections, max_process_workers, max_size, max_thread_workers, max_workers, memory_info, new_soft, op, op_name, op_type, operation, operation_groups, operation_name, operation_stats, operations, ops, optimizations, optimized, optimizer_stats, parameters, pool_stats, process, processor, psutil, query, r, remove_count, resource, result, results, self, semaphore, sorted_keys, start_time, success, successes, t, task, tasks, text, texts, times, tool, tools, ttl, valid_results

from sqlalchemy import func
Performance Optimization Utilities for GAIA System
Implements caching, batching, connection pooling, and other optimizations
"""

from typing import Dict
from typing import List
from typing import Callable

import asyncio
import functools
import gc
import time
import hashlib

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np

import logging

# Optional imports for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Performance optimization utilities for GAIA system"""

    def __init__(self, max_process_workers: int = 4, max_thread_workers: int = 10):
        self.process_pool = ProcessPoolExecutor(max_workers=max_process_workers)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_thread_workers)
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_size_limit = 1000
        self.cache_ttl = 3600  # 1 hour default

        # Performance tracking
        self.operation_times = {}
        self.memory_snapshots = []

        logger.info(f"Performance optimizer initialized with {max_process_workers} process workers and {max_thread_workers} thread workers")

    def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage across components"""
        optimizations = {
            "before_optimization": self._get_memory_usage(),
            "timestamp": datetime.now().isoformat()
        }

        # Force garbage collection
        collected = gc.collect()
        optimizations["garbage_collected"] = collected

        # Clear caches
        cache_size_before = len(self.cache)
        self.cache.clear()
        optimizations["cache_cleared"] = cache_size_before

        # Reduce memory limits if possible
        if RESOURCE_AVAILABLE:
            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                # Set soft limit to 1GB
                new_soft = min(1024 * 1024 * 1024, hard)
                resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
                optimizations["memory_limit_set"] = f"{new_soft / (1024*1024*1024):.1f}GB"
            except Exception as e:
                optimizations["memory_limit_error"] = str(e)

        # Force another GC
        gc.collect()

        # Get memory after optimization
        optimizations["after_optimization"] = self._get_memory_usage()

        # Calculate memory saved
        before_mb = optimizations["before_optimization"]["rss_mb"]
        after_mb = optimizations["after_optimization"]["rss_mb"]
        optimizations["memory_saved_mb"] = before_mb - after_mb

        logger.info(f"Memory optimization completed. Saved {optimizations['memory_saved_mb']:.1f}MB")
        return optimizations

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        if not PSUTIL_AVAILABLE:
            return {"error": "psutil not available"}

        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            }
        except Exception as e:
            return {"error": str(e)}

    def memoize(self, ttl: int = 3600, max_size: int = 1000):
        """Memoization decorator with TTL and size limits"""
        def decorator(self, func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Create cache key
                cache_key = self._create_cache_key(func.__name__, args, kwargs)

                # Check cache
                if cache_key in self.cache:
                    entry = self.cache[cache_key]
                    if time.time() - entry['timestamp'] < ttl:
                        self.cache_hits += 1
                        return entry['result']

                # Cache miss
                self.cache_misses += 1
                result = await func(*args, **kwargs)

                # Store in cache
                self.cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time()
                }

                # Limit cache size
                self._limit_cache_size(max_size)

                return result

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                cache_key = self._create_cache_key(func.__name__, args, kwargs)

                if cache_key in self.cache:
                    entry = self.cache[cache_key]
                    if time.time() - entry['timestamp'] < ttl:
                        self.cache_hits += 1
                        return entry['result']

                self.cache_misses += 1
                result = func(*args, **kwargs)

                self.cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time()
                }

                self._limit_cache_size(max_size)

                return result

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def _create_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Create a cache key from function name and arguments"""
        # Create a hash of the function name and arguments
        key_data = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _limit_cache_size(self, max_size: int):
        """Limit cache size by removing oldest entries"""
        if len(self.cache) > max_size:
            # Sort by timestamp and remove oldest entries
            sorted_keys = sorted(
                self.cache.keys(),
                key=lambda k: self.cache[k]['timestamp']
            )

            # Remove oldest 20% of entries
            remove_count = max(1, len(sorted_keys) // 5)
            for key in sorted_keys[:remove_count]:
                del self.cache[key]

    async def batch_process(self, items: List[Any], processor: Callable,
                          batch_size: int = 32, max_workers: int = None) -> List[Any]:
        """Process items in batches for efficiency"""
        results = []

        # Use provided max_workers or default
        if max_workers is None:
            max_workers = self.thread_pool._max_workers

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]

            # Process batch in parallel
            if asyncio.iscoroutinefunction(processor):
                batch_results = await asyncio.gather(*[processor(item) for item in batch])
            else:
                # Use thread pool for sync functions
                loop = asyncio.get_event_loop()
                batch_results = await asyncio.gather(*[
                    loop.run_in_executor(self.thread_pool, processor, item)
                    for item in batch
                ])

            results.extend(batch_results)

        return results

    def vectorize_embeddings(self, texts: List[str], embedding_model) -> np.ndarray:
        """Vectorized embedding generation for efficiency"""
        # Batch encode for efficiency
        batch_size = 64
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                embeddings = embedding_model.encode(
                    batch,
                    batch_size=len(batch),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=len(batch) > 100
                )
                all_embeddings.append(embeddings)
            except Exception as e:
                logger.error(f"Embedding generation failed for batch {i}: {e}")
                # Generate fallback embeddings
                fallback_embeddings = self._generate_fallback_embeddings(batch)
                all_embeddings.append(fallback_embeddings)

        return np.vstack(all_embeddings)

    def _generate_fallback_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate fallback embeddings when model fails"""
        embeddings = []
        for text in texts:
            # Simple hash-based embedding
            hash_val = hash(text.lower()) % 1000
            embedding = [float(hash_val + i) / 1000.0 for i in range(384)]
            embeddings.append(embedding)
        return np.array(embeddings)

    def track_operation_time(self, operation_name: str):
        """Decorator to track operation execution time"""
        def decorator(self, func: Callable) -> Callable:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    self._record_operation_time(operation_name, duration, success=True)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self._record_operation_time(operation_name, duration, success=False, error=str(e))
                    raise

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self._record_operation_time(operation_name, duration, success=True)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self._record_operation_time(operation_name, duration, success=False, error=str(e))
                    raise

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def _record_operation_time(self, operation_name: str, duration: float,
                             success: bool, error: str = None):
        """Record operation execution time"""
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []

        self.operation_times[operation_name].append({
            'duration': duration,
            'timestamp': time.time(),
            'success': success,
            'error': error
        })

        # Keep only last 1000 operations
        if len(self.operation_times[operation_name]) > 1000:
            self.operation_times[operation_name] = self.operation_times[operation_name][-1000:]

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        cache_hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0
            else 0
        )

        # Calculate operation statistics
        operation_stats = {}
        for op_name, times in self.operation_times.items():
            if times:
                durations = [t['duration'] for t in times]
                successes = [t['success'] for t in times]

                operation_stats[op_name] = {
                    'count': len(times),
                    'avg_duration': sum(durations) / len(durations),
                    'min_duration': min(durations),
                    'max_duration': max(durations),
                    'success_rate': sum(successes) / len(successes)
                }

        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "memory_usage_mb": self._get_memory_usage().get("rss_mb", 0),
            "cpu_percent": self._get_cpu_usage(),
            "operation_stats": operation_stats
        }

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        if not PSUTIL_AVAILABLE:
            return 0.0

        try:
            return psutil.Process().cpu_percent(interval=0.1)
        except:
            return 0.0

# Async connection pooling
class ConnectionPool:
    """Connection pool for external services"""

    def __init__(self, max_connections: int = 10, connection_timeout: float = 30.0):
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.connections = asyncio.Queue(maxsize=max_connections)
        self.created_connections = 0
        self.active_connections = 0

        logger.info(f"Connection pool initialized with {max_connections} max connections")

    async def acquire(self):
        """Acquire a connection from the pool"""
        try:
            # Try to get existing connection
            connection = self.connections.get_nowait()
            self.active_connections += 1
            return connection
        except asyncio.QueueEmpty:
            # Create new connection if under limit
            if self.created_connections < self.max_connections:
                connection = await self._create_connection()
                self.created_connections += 1
                self.active_connections += 1
                return connection
            else:
                # Wait for available connection
                connection = await asyncio.wait_for(
                    self.connections.get(),
                    timeout=self.connection_timeout
                )
                self.active_connections += 1
                return connection

    async def release(self, connection):
        """Release connection back to pool"""
        try:
            await self.connections.put(connection)
            self.active_connections -= 1
        except asyncio.QueueFull:
            # Pool is full, close the connection
            await self._close_connection(connection)
            self.active_connections -= 1

    async def _create_connection(self):
        """Create a new connection"""
        # This is a generic implementation
        # Specific implementations would override this
        import aiohttp
        return aiohttp.ClientSession()

    async def _close_connection(self, connection):
        """Close a connection"""
        if hasattr(connection, 'close'):
            await connection.close()

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            "max_connections": self.max_connections,
            "created_connections": self.created_connections,
            "active_connections": self.active_connections,
            "available_connections": self.connections.qsize(),
            "utilization": self.active_connections / self.max_connections
        }

# Optimized GAIA components
class OptimizedGAIASystem:
    """Optimized GAIA system with performance enhancements"""

    def __init__(self):
        self.optimizer = PerformanceOptimizer()
        self.connection_pool = ConnectionPool()
        self.batch_size = 32
        self.max_concurrent_tasks = 5

    @PerformanceOptimizer().memoize(ttl=600)
    async def cached_reasoning(self, query: str) -> Any:
        """Cached reasoning for repeated queries"""
        # This would contain the actual reasoning logic
        # For now, return a mock result
        return {
            "reasoning_path": f"Cached reasoning for: {query}",
            "confidence": 0.8,
            "timestamp": datetime.now().isoformat()
        }

    async def parallel_tool_execution(self, tools: List[Any],
                                    parameters: Dict[str, Any]) -> List[Any]:
        """Execute tools in parallel with optimization"""
        tasks = []

        for tool in tools:
            if hasattr(tool, '__call__'):
                if asyncio.iscoroutinefunction(tool):
                    tasks.append(tool(**parameters))
                else:
                    # Wrap sync function
                    loop = asyncio.get_event_loop()
                    tasks.append(
                        loop.run_in_executor(
                            self.optimizer.thread_pool,
                            tool,
                            **parameters
                        )
                    )

        # Execute with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        async def limited_task(task):
            async with semaphore:
                return await task

        results = await asyncio.gather(*[limited_task(task) for task in tasks],
                                     return_exceptions=True)

        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]

        return valid_results

    @PerformanceOptimizer().track_operation_time("batch_embedding")
    async def batch_embed_texts(self, texts: List[str], embedding_model) -> np.ndarray:
        """Batch embed texts with optimization"""
        return self.optimizer.vectorize_embeddings(texts, embedding_model)

    async def optimized_memory_operations(self, operations: List[Dict[str, Any]]) -> List[Any]:
        """Optimized memory operations with batching"""
        results = []

        # Group operations by type
        operation_groups = {}
        for op in operations:
            op_type = op.get('type', 'unknown')
            if op_type not in operation_groups:
                operation_groups[op_type] = []
            operation_groups[op_type].append(op)

        # Process each group in batches
        for op_type, ops in operation_groups.items():
            batch_results = await self.optimizer.batch_process(
                ops,
                lambda op: self._process_memory_operation(op),
                batch_size=self.batch_size
            )
            results.extend(batch_results)

        return results

    def _process_memory_operation(self, operation: Dict[str, Any]) -> Any:
        """Process a single memory operation"""
        # This would contain the actual memory operation logic
        return {
            "operation_id": operation.get('id'),
            "status": "processed",
            "timestamp": datetime.now().isoformat()
        }

    def get_system_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive system performance statistics"""
        optimizer_stats = self.optimizer.get_optimization_stats()
        pool_stats = self.connection_pool.get_pool_stats()

        return {
            "optimizer": optimizer_stats,
            "connection_pool": pool_stats,
            "system": {
                "batch_size": self.batch_size,
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "uptime": time.time() - self.optimizer.start_time if hasattr(self.optimizer, 'start_time') else 0
            }
        }

# Performance monitoring decorators
def monitor_performance(operation_name: str = None):
    """Decorator to monitor function performance"""
    def decorator(self, func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"{operation_name or func.__name__} completed in {duration:.4f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{operation_name or func.__name__} failed after {duration:.4f}s: {e}")
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"{operation_name or func.__name__} completed in {duration:.4f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{operation_name or func.__name__} failed after {duration:.4f}s: {e}")
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

# Memory optimization utilities
class MemoryOptimizer:
    """Memory optimization utilities"""

    @staticmethod
    def optimize_numpy_arrays(self, arrays: List[np.ndarray]) -> List[np.ndarray]:
        """Optimize numpy arrays for memory usage"""
        optimized = []
        for arr in arrays:
            # Use smaller dtype if possible
            if arr.dtype == np.float64 and arr.max() < 1e6 and arr.min() > -1e6:
                optimized.append(arr.astype(np.float32))
            else:
                optimized.append(arr)
        return optimized

    @staticmethod
    def clear_memory():
        """Clear memory and force garbage collection"""
        gc.collect()
        if PSUTIL_AVAILABLE:
            # Clear page cache on Linux
            try:
                import os
                os.system('sync')  # Flush filesystem buffers
            except:
                pass