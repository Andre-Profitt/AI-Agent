from benchmarks.cot_performance import duration
from examples.basic.demo_hybrid_architecture import percentage
from examples.enhanced_unified_example import start_time
from performance_dashboard import cpu_percent
from performance_dashboard import metric
from tests.load_test import args
from tests.load_test import success
from tests.unit.simple_test import func

from src.application.tools.tool_executor import operation
from src.core.health_check import disk
from src.core.monitoring import memory
from src.database.models import agent_id
from src.database.models import agent_type
from src.database.models import component
from src.database.models import labels
from src.database.models import metadata
from src.database.models import priority
from src.database.models import resource_type
from src.database.models import status
from src.database.supabase_manager import table
from src.gaia_components.production_vector_store import count
from src.infrastructure.monitoring.metrics import operation_id
from src.tools_introspection import error
from src.tools_introspection import error_type
from src.tools_introspection import name
from src.unified_architecture.conflict_resolution import severity
from src.unified_architecture.enhanced_platform import task_type

from src.agents.advanced_agent_fsm import Agent
# TODO: Fix undefined variables: Any, CONTENT_TYPE_LATEST, Callable, CollectorRegistry, Dict, Optional, agent_id, agent_type, args, asynccontextmanager, component, contextmanager, count, cpu_percent, database, dataclass, disk, duration, e, endpoint, error, error_type, func, generate_latest, interval, kwargs, labels, logging, memory, metadata, metric, name, operation, operation_id, operation_name, percentage, priority, resource_type, result, service, severity, size, start_time, status, success, table, task_type, threading, time
from prometheus_client import Counter
from prometheus_client import Gauge
from prometheus_client import Histogram


"""

from collections import Counter
from contextlib import contextmanager
from fastapi import status
from sqlalchemy import func
from typing import Any
from typing import Callable
from typing import Optional
Comprehensive Metrics Collection System
Provides Prometheus metrics, timing decorators, and distributed tracing
"""

import time
import functools
import asyncio
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from contextlib import asynccontextmanager, contextmanager
import logging
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, 
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, multiprocess
)
import threading
from datetime import datetime

logger = logging.getLogger(__name__)

# =============================
# Metrics Registry
# =============================

class MetricsRegistry:
    """Central registry for all application metrics"""
    
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
    def register_metric(self, name: str, metric: Any) -> None:
        """Register a metric with the registry"""
        with self._lock:
            self._metrics[name] = metric
            
    def get_metric(self, name: str) -> Optional[Any]:
        """Get a metric by name"""
        return self._metrics.get(name)
        
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all registered metrics"""
        return self._metrics.copy()
        
    def generate_metrics(self) -> str:
        """Generate Prometheus metrics output"""
        return generate_latest(self.registry)

# Global metrics registry
metrics_registry = MetricsRegistry()

# =============================
# Core Metrics
# =============================

# Agent Metrics
AGENT_REGISTRATIONS = Counter(
    'agent_registrations_total',
    'Total number of agent registrations',
    ['agent_type', 'status']
)

AGENT_TASK_EXECUTIONS = Counter(
    'agent_task_executions_total',
    'Total number of task executions',
    ['agent_id', 'task_type', 'status']
)

AGENT_TASK_DURATION = Histogram(
    'agent_task_duration_seconds',
    'Task execution duration in seconds',
    ['agent_id', 'task_type'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

AGENT_AVAILABILITY = Gauge(
    'agent_availability',
    'Agent availability status',
    ['agent_id', 'status']
)

# Task Metrics
TASKS_SUBMITTED = Counter(
    'tasks_submitted_total',
    'Total number of tasks submitted',
    ['task_type', 'priority']
)

TASKS_COMPLETED = Counter(
    'tasks_completed_total',
    'Total number of tasks completed',
    ['task_type', 'status']
)

TASK_QUEUE_SIZE = Gauge(
    'task_queue_size',
    'Current number of tasks in queue',
    ['priority']
)

# Database Metrics
DB_OPERATIONS = Counter(
    'database_operations_total',
    'Total database operations',
    ['operation', 'table', 'status']
)

DB_OPERATION_DURATION = Histogram(
    'database_operation_duration_seconds',
    'Database operation duration',
    ['operation', 'table'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

DB_CONNECTIONS = Gauge(
    'database_connections_active',
    'Active database connections',
    ['database']
)

# External Service Metrics
EXTERNAL_API_CALLS = Counter(
    'external_api_calls_total',
    'Total external API calls',
    ['service', 'endpoint', 'status']
)

EXTERNAL_API_DURATION = Histogram(
    'external_api_duration_seconds',
    'External API call duration',
    ['service', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Resource Metrics
RESOURCE_UTILIZATION = Gauge(
    'resource_utilization_percent',
    'Resource utilization percentage',
    ['resource_type', 'agent_id']
)

MEMORY_USAGE = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    ['component']
)

CPU_USAGE = Gauge(
    'cpu_usage_percent',
    'CPU usage percentage',
    ['component']
)

# Error Metrics
ERRORS_TOTAL = Counter(
    'errors_total',
    'Total number of errors',
    ['error_type', 'component', 'severity']
)

# =============================
# Timing Decorators
# =============================

def time_function(metric_name: str, labels: Optional[Dict[str, str]] = None) -> Any:
    """Decorator to time function execution and record metrics"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record success metric
                if labels:
                    EXTERNAL_API_DURATION.labels(**labels).observe(duration)
                else:
                    EXTERNAL_API_DURATION.observe(duration)
                    
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                # Record error metric
                ERRORS_TOTAL.labels(
                    error_type=type(e).__name__,
                    component=func.__module__,
                    severity='error'
                ).inc()
                
                # Record duration even for errors
                if labels:
                    EXTERNAL_API_DURATION.labels(**labels).observe(duration)
                else:
                    EXTERNAL_API_DURATION.observe(duration)
                    
                raise
                
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record success metric
                if labels:
                    EXTERNAL_API_DURATION.labels(**labels).observe(duration)
                else:
                    EXTERNAL_API_DURATION.observe(duration)
                    
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                # Record error metric
                ERRORS_TOTAL.labels(
                    error_type=type(e).__name__,
                    component=func.__module__,
                    severity='error'
                ).inc()
                
                # Record duration even for errors
                if labels:
                    EXTERNAL_API_DURATION.labels(**labels).observe(duration)
                else:
                    EXTERNAL_API_DURATION.observe(duration)
                    
                raise
                
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator

def track_database_operation(operation: str, table: str) -> Any:
    """Decorator to track database operations"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record success
                DB_OPERATIONS.labels(
                    operation=operation,
                    table=table,
                    status='success'
                ).inc()
                
                DB_OPERATION_DURATION.labels(
                    operation=operation,
                    table=table
                ).observe(duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                # Record error
                DB_OPERATIONS.labels(
                    operation=operation,
                    table=table,
                    status='error'
                ).inc()
                
                DB_OPERATION_DURATION.labels(
                    operation=operation,
                    table=table
                ).observe(duration)
                
                ERRORS_TOTAL.labels(
                    error_type=type(e).__name__,
                    component='database',
                    severity='error'
                ).inc()
                
                raise
                
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record success
                DB_OPERATIONS.labels(
                    operation=operation,
                    table=table,
                    status='success'
                ).inc()
                
                DB_OPERATION_DURATION.labels(
                    operation=operation,
                    table=table
                ).observe(duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                # Record error
                DB_OPERATIONS.labels(
                    operation=operation,
                    table=table,
                    status='error'
                ).inc()
                
                DB_OPERATION_DURATION.labels(
                    operation=operation,
                    table=table
                ).observe(duration)
                
                ERRORS_TOTAL.labels(
                    error_type=type(e).__name__,
                    component='database',
                    severity='error'
                ).inc()
                
                raise
                
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator

# =============================
# Context Managers
# =============================

@contextmanager
def track_operation(operation_name: str, labels: Optional[Dict[str, str]] = None) -> Any:
    """Context manager for tracking operations"""
    start_time = time.time()
    try:
        yield
        duration = time.time() - start_time
        
        # Record success metric
        if labels:
            EXTERNAL_API_DURATION.labels(**labels).observe(duration)
        else:
            EXTERNAL_API_DURATION.observe(duration)
            
    except Exception as e:
        duration = time.time() - start_time
        
        # Record error metric
        ERRORS_TOTAL.labels(
            error_type=type(e).__name__,
            component=operation_name,
            severity='error'
        ).inc()
        
        # Record duration even for errors
        if labels:
            EXTERNAL_API_DURATION.labels(**labels).observe(duration)
        else:
            EXTERNAL_API_DURATION.observe(duration)
            
        raise

@asynccontextmanager
async def track_async_operation(operation_name: str, labels: Optional[Dict[str, str]] = None) -> Any:
    """Async context manager for tracking operations"""
    start_time = time.time()
    try:
        yield
        duration = time.time() - start_time
        
        # Record success metric
        if labels:
            EXTERNAL_API_DURATION.labels(**labels).observe(duration)
        else:
            EXTERNAL_API_DURATION.observe(duration)
            
    except Exception as e:
        duration = time.time() - start_time
        
        # Record error metric
        ERRORS_TOTAL.labels(
            error_type=type(e).__name__,
            component=operation_name,
            severity='error'
        ).inc()
        
        # Record duration even for errors
        if labels:
            EXTERNAL_API_DURATION.labels(**labels).observe(duration)
        else:
            EXTERNAL_API_DURATION.observe(duration)
            
        raise

# =============================
# Performance Monitoring
# =============================

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: bool = True
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self) -> Any:
        if self.metadata is None:
            self.metadata = {}
            
    def complete(self, success: bool = True, error: Optional[Exception] = None) -> Any:
        """Complete the performance measurement"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.error = error
        
        # Record metrics
        if success:
            EXTERNAL_API_DURATION.labels(
                service=self.operation_name
            ).observe(self.duration)
        else:
            ERRORS_TOTAL.labels(
                error_type=type(error).__name__ if error else 'unknown',
                component=self.operation_name,
                severity='error'
            ).inc()
            
            EXTERNAL_API_DURATION.labels(
                service=self.operation_name
            ).observe(self.duration)

class PerformanceTracker:
    """Tracks performance metrics across operations"""
    
    def __init__(self) -> None:
        self.metrics: Dict[str, PerformanceMetrics] = {}
        
    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking an operation"""
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        self.metrics[operation_id] = PerformanceMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        return operation_id
        
    def complete_operation(self, operation_id: str, success: bool = True, 
                          error: Optional[Exception] = None) -> None:
        """Complete tracking an operation"""
        if operation_id in self.metrics:
            self.metrics[operation_id].complete(success, error)
            
    def get_operation_metrics(self, operation_id: str) -> Optional[PerformanceMetrics]:
        """Get metrics for a specific operation"""
        return self.metrics.get(operation_id)
        
    def get_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Get all tracked metrics"""
        return self.metrics.copy()

# Global performance tracker
performance_tracker = PerformanceTracker()

# =============================
# Resource Monitoring
# =============================

import psutil
import threading
import time

class ResourceMonitor:
    """Monitors system resources"""
    
    def __init__(self, interval: float = 30.0) -> None:
        self.interval = interval
        self.running = False
        self.monitor_thread = None
        
    def start(self) -> None:
        """Start resource monitoring"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
    def stop(self) -> None:
        """Stop resource monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_loop(self) -> Any:
        """Main monitoring loop"""
        while self.running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                CPU_USAGE.labels(component='system').set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                MEMORY_USAGE.labels(component='system').set(memory.used)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                RESOURCE_UTILIZATION.labels(
                    resource_type='disk',
                    agent_id='system'
                ).set((disk.used / disk.total) * 100)
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error("Error in resource monitoring: {}", extra={"e": e})
                time.sleep(self.interval)

# Global resource monitor
resource_monitor = ResourceMonitor()

# =============================
# Metrics Endpoint
# =============================

def get_metrics_response() -> Any:
    """Get metrics response for FastAPI endpoint"""
    return generate_latest(), CONTENT_TYPE_LATEST

# =============================
# Utility Functions
# =============================

def record_agent_registration(agent_type: str, status: str = 'success') -> Any:
    """Record agent registration metric"""
    AGENT_REGISTRATIONS.labels(agent_type=agent_type, status=status).inc()

def record_task_execution(agent_id: str, task_type: str, status: str = 'success') -> Any:
    """Record task execution metric"""
    AGENT_TASK_EXECUTIONS.labels(agent_id=agent_id, task_type=task_type, status=status).inc()

def record_task_duration(agent_id: str, task_type: str, duration: float) -> Any:
    """Record task duration metric"""
    AGENT_TASK_DURATION.labels(agent_id=agent_id, task_type=task_type).observe(duration)

def record_agent_availability(agent_id: str, status: str) -> Any:
    """Record agent availability metric"""
    AGENT_AVAILABILITY.labels(agent_id=agent_id, status=status).set(1 if status == 'available' else 0)

def record_task_submission(task_type: str, priority: int) -> Any:
    """Record task submission metric"""
    TASKS_SUBMITTED.labels(task_type=task_type, priority=str(priority)).inc()

def record_task_completion(task_type: str, status: str) -> Any:
    """Record task completion metric"""
    TASKS_COMPLETED.labels(task_type=task_type, status=status).inc()

def record_external_api_call(service: str, endpoint: str, status: str = 'success') -> Any:
    """Record external API call metric"""
    EXTERNAL_API_CALLS.labels(service=service, endpoint=endpoint, status=status).inc()

def record_error(error_type: str, component: str, severity: str = 'error') -> Any:
    """Record error metric"""
    ERRORS_TOTAL.labels(error_type=error_type, component=component, severity=severity).inc()

def update_resource_utilization(resource_type: str, agent_id: str, percentage: float) -> bool:
    """Update resource utilization metric"""
    RESOURCE_UTILIZATION.labels(resource_type=resource_type, agent_id=agent_id).set(percentage)

def update_task_queue_size(priority: str, size: int) -> bool:
    """Update task queue size metric"""
    TASK_QUEUE_SIZE.labels(priority=priority).set(size)

def update_db_connections(database: str, count: int) -> bool:
    """Update database connections metric"""
    DB_CONNECTIONS.labels(database=database).set(count) 