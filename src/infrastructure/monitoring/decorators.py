from src.tools.base_tool import Tool

from src.agents.advanced_agent_fsm import Agent

"""

from sqlalchemy import func
from typing import Callable
Monitoring decorators for metrics collection

This module provides decorators for automatically collecting metrics
from various parts of the AI Agent system.
"""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Dict, Optional
from contextlib import asynccontextmanager

from src.infrastructure.monitoring.metrics import (
    record_agent_registration, record_task_execution, record_task_duration,
    record_error, AGENT_TASK_EXECUTIONS, AGENT_TASK_DURATION, ERRORS_TOTAL
)

logger = logging.getLogger(__name__)


def async_metrics(func: Callable) -> Callable:
    """
    Decorator for async metrics collection.
    
    This decorator automatically records execution time and success/failure
    metrics for async functions.
    
    Args:
        func: The async function to decorate
        
    Returns:
        Decorated function with metrics collection
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        function_name = func.__name__
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Record success metrics
            logger.debug("Function {} completed successfully in {}s", extra={"function_name": function_name, "execution_time": execution_time})
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Record error metrics
            record_error(type(e).__name__, function_name, "error")
            logger.error("Function {} failed after {}s: {}", extra={"function_name": function_name, "execution_time": execution_time, "e": e})
            
            raise
    
    return wrapper


def agent_metrics(agent_name: str) -> Callable:
    """
    Decorator for agent-specific metrics collection.
    
    This decorator records agent-specific metrics including task execution,
    duration, and error rates.
    
    Args:
        agent_name: Name of the agent for metrics labeling
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Extract task information if available
            task_type = "unknown"
            if args and hasattr(args[0], 'task_type'):
                task_type = args[0].task_type
            elif 'task' in kwargs and hasattr(kwargs['task'], 'task_type'):
                task_type = kwargs['task'].task_type
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Record agent task execution metrics
                record_task_execution(agent_name, task_type, "success")
                record_task_duration(agent_name, task_type, execution_time)
                
                logger.debug("Agent {} completed {} task in {}s", extra={"agent_name": agent_name, "task_type": task_type, "execution_time": execution_time})
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Record error metrics
                record_task_execution(agent_name, task_type, "error")
                record_error(type(e).__name__, agent_name, "agent_error")
                
                logger.error("Agent {} failed {} task after {}s: {}", extra={"agent_name": agent_name, "task_type": task_type, "execution_time": execution_time, "e": e})
                
                raise
        
        return wrapper
    return decorator


def tool_metrics(tool_name: str) -> Callable:
    """
    Decorator for tool-specific metrics collection.
    
    This decorator records tool execution metrics including usage frequency,
    execution time, and error rates.
    
    Args:
        tool_name: Name of the tool for metrics labeling
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Record tool execution metrics
                record_task_execution(tool_name, "tool_execution", "success")
                record_task_duration(tool_name, "tool_execution", execution_time)
                
                logger.debug("Tool {} executed successfully in {}s", extra={"tool_name": tool_name, "execution_time": execution_time})
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Record error metrics
                record_task_execution(tool_name, "tool_execution", "error")
                record_error(type(e).__name__, tool_name, "tool_error")
                
                logger.error("Tool {} failed after {}s: {}", extra={"tool_name": tool_name, "execution_time": execution_time, "e": e})
                
                raise
        
        return wrapper
    return decorator


def performance_metrics(operation_name: str) -> Callable:
    """
    Decorator for performance metrics collection.
    
    This decorator records detailed performance metrics including
    execution time, memory usage, and throughput.
    
    Args:
        operation_name: Name of the operation for metrics labeling
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Record performance metrics
                record_task_duration(operation_name, "performance", execution_time)
                
                # Log performance information
                if execution_time > 1.0:  # Log slow operations
                    logger.warning("Slow operation {}: {}s", extra={"operation_name": operation_name, "execution_time": execution_time})
                else:
                    logger.debug("Operation {} completed in {}s", extra={"operation_name": operation_name, "execution_time": execution_time})
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Record error metrics
                record_error(type(e).__name__, operation_name, "performance_error")
                
                logger.error("Performance operation {} failed after {}s: {}", extra={"operation_name": operation_name, "execution_time": execution_time, "e": e})
                
                raise
        
        return wrapper
    return decorator


@asynccontextmanager
async def metrics_context(operation_name: str, **labels):
    """
    Context manager for metrics collection.
    
    This context manager automatically records metrics for operations
    that span multiple function calls or have complex execution patterns.
    
    Args:
        operation_name: Name of the operation
        **labels: Additional labels for metrics
        
    Yields:
        Context for metrics collection
    """
    start_time = time.time()
    
    try:
        yield
        execution_time = time.time() - start_time
        
        # Record success metrics
        record_task_execution(operation_name, "context", "success")
        record_task_duration(operation_name, "context", execution_time)
        
        logger.debug("Context {} completed successfully in {}s", extra={"operation_name": operation_name, "execution_time": execution_time})
        
    except Exception as e:
        execution_time = time.time() - start_time
        
        # Record error metrics
        record_task_execution(operation_name, "context", "error")
        record_error(type(e).__name__, operation_name, "context_error")
        
        logger.error("Context {} failed after {}s: {}", extra={"operation_name": operation_name, "execution_time": execution_time, "e": e})
        
        raise


def error_tracking(func: Callable) -> Callable:
    """
    Decorator for error tracking and reporting.
    
    This decorator specifically focuses on error collection and reporting,
    providing detailed error context for debugging.
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function with error tracking
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        function_name = func.__name__
        
        try:
            return await func(*args, **kwargs)
            
        except Exception as e:
            # Record detailed error information
            error_context = {
                "function": function_name,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "args_count": len(args),
                "kwargs_keys": list(kwargs.keys())
            }
            
            record_error(type(e).__name__, function_name, "function_error")
            
            logger.error("Error in {}: {}", extra={})
            
            raise
    
    return wrapper


def throughput_metrics(operation_name: str) -> Callable:
    """
    Decorator for throughput metrics collection.
    
    This decorator tracks the rate of operations and provides
    throughput statistics.
    
    Args:
        operation_name: Name of the operation
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        # Track operation counts
        operation_count = 0
        last_reset_time = time.time()
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal operation_count, last_reset_time
            
            current_time = time.time()
            
            # Reset counter every minute
            if current_time - last_reset_time > 60:
                operation_count = 0
                last_reset_time = current_time
            
            operation_count += 1
            
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Record throughput metrics
                throughput = operation_count / (current_time - last_reset_time + 1)
                
                logger.debug("Operation {} throughput: {} ops/sec", extra={})
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Record error in throughput context
                record_error(type(e).__name__, operation_name, "throughput_error", extra={"_function_name_": "function_name"})
                
                logger.error("Throughput operation {} failed: {}", extra={"operation_name": operation_name, "e": e})
                
                raise
        
        return wrapper
    return decorator


def resource_metrics(func: Callable) -> Callable:
    """
    Decorator for resource usage metrics collection.
    
    This decorator tracks resource usage including memory and CPU
    consumption during function execution.
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function with resource tracking
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Record initial resource usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent()
        
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Record final resource usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            final_cpu = process.cpu_percent()
            
            memory_delta = final_memory - initial_memory
            cpu_delta = final_cpu - initial_cpu
            
            logger.debug(f"Resource usage for {func.__name__}: "
                        f"Memory: {memory_delta}MB, CPU: {cpu_delta}%, "
                        f"Time: {execution_time}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Record error with resource context
            record_error(type(e).__name__, func.__name__, "resource_error")
            
            logger.error("Resource operation {} failed after {}s: {}", extra={"func___name__": func.__name__, "execution_time": execution_time, "e": e})
            
            raise
    
    return wrapper


# Convenience function for getting metrics decorator
def get_metrics_decorator(metric_type: str, **kwargs) -> Callable:
    """
    Get a metrics decorator by type.
    
    Args:
        metric_type: Type of metrics decorator
        **kwargs: Additional arguments for the decorator
        
    Returns:
        Metrics decorator function
    """
    decorators = {
        "async": async_metrics,
        "agent": agent_metrics,
        "tool": tool_metrics,
        "performance": performance_metrics,
        "error": error_tracking,
        "throughput": throughput_metrics,
        "resource": resource_metrics
    }
    
    if metric_type not in decorators:
        raise ValueError(f"Unknown metric type: {metric_type}")
    
    decorator = decorators[metric_type]
    
    if kwargs:
        return decorator(**kwargs)
    else:
        return decorator 