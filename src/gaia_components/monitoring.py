"""
Production Monitoring System for GAIA
Implements comprehensive monitoring with Prometheus and OpenTelemetry
"""

import os
import time
import functools
import asyncio
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
import logging

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create mock metrics if prometheus_client is not available
    class MockMetric:
        def __init__(self, *args, **kwargs):
            pass
        def inc(self, *args, **kwargs):
            pass
        def observe(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass
    
    Counter = Histogram = Gauge = Info = MockMetric

# OpenTelemetry setup
try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global metrics
if PROMETHEUS_AVAILABLE:
    # Query metrics
    query_counter = Counter('gaia_queries_total', 'Total number of queries processed')
    query_success_counter = Counter('gaia_queries_success', 'Successful queries')
    query_failure_counter = Counter('gaia_queries_failure', 'Failed queries')
    query_duration = Histogram('gaia_query_duration_seconds', 'Query processing duration')
    
    # Tool metrics
    tool_execution_counter = Counter('gaia_tool_executions_total', 'Total tool executions', ['tool_name', 'tool_type'])
    tool_execution_duration = Histogram('gaia_tool_duration_seconds', 'Tool execution duration', ['tool_name'])
    tool_failure_counter = Counter('gaia_tool_failures_total', 'Tool execution failures', ['tool_name', 'error_type'])
    
    # Memory metrics
    memory_size_gauge = Gauge('gaia_memory_size', 'Current memory system size', ['memory_type'])
    memory_operations_counter = Counter('gaia_memory_operations_total', 'Memory operations', ['operation_type'])
    
    # Agent metrics
    active_agents_gauge = Gauge('gaia_active_agents', 'Number of active agents', ['agent_type'])
    agent_task_counter = Counter('gaia_agent_tasks_total', 'Agent tasks processed', ['agent_type', 'task_status'])
    
    # System metrics
    system_memory_gauge = Gauge('gaia_system_memory_bytes', 'System memory usage')
    system_cpu_gauge = Gauge('gaia_system_cpu_percent', 'System CPU usage')
    
    # GAIA info
    gaia_info = Info('gaia_system', 'GAIA system information')
else:
    # Mock metrics
    query_counter = query_success_counter = query_failure_counter = query_duration = None
    tool_execution_counter = tool_execution_duration = tool_failure_counter = None
    memory_size_gauge = memory_operations_counter = None
    active_agents_gauge = agent_task_counter = None
    system_memory_gauge = system_cpu_gauge = None
    gaia_info = None

def setup_tracing():
    """Setup distributed tracing with Jaeger"""
    if not OPENTELEMETRY_AVAILABLE:
        logger.warning("OpenTelemetry not available, tracing disabled")
        return None
    
    try:
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)
        
        # Create Jaeger exporter
        jaeger_endpoint = os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces")
        jaeger_exporter = JaegerExporter(
            agent_host_name=os.getenv("JAEGER_HOST", "localhost"),
            agent_port=int(os.getenv("JAEGER_PORT", 6831)),
            collector_endpoint=jaeger_endpoint
        )
        
        # Create span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        logger.info("Distributed tracing initialized with Jaeger")
        return tracer
        
    except Exception as e:
        logger.error(f"Failed to setup tracing: {e}")
        return None

# Monitoring decorators
def monitor_performance(metric_name: str = None):
    """Decorator to monitor function performance"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            tracer = trace.get_tracer(__name__) if OPENTELEMETRY_AVAILABLE else None
            
            if tracer:
                with tracer.start_as_current_span(metric_name or func.__name__) as span:
                    try:
                        result = await func(*args, **kwargs)
                        duration = time.time() - start_time
                        
                        # Record metrics
                        if query_duration:
                            query_duration.observe(duration)
                        if query_success_counter:
                            query_success_counter.inc()
                        
                        # Add span attributes
                        span.set_attribute("duration", duration)
                        span.set_attribute("success", True)
                        
                        return result
                        
                    except Exception as e:
                        duration = time.time() - start_time
                        
                        # Record failure metrics
                        if query_failure_counter:
                            query_failure_counter.inc()
                        if query_duration:
                            query_duration.observe(duration)
                        
                        # Add error to span
                        span.set_attribute("success", False)
                        span.set_attribute("error", str(e))
                        span.set_attribute("duration", duration)
                        
                        raise
            else:
                # No tracing available
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Record metrics
                    if query_duration:
                        query_duration.observe(duration)
                    if query_success_counter:
                        query_success_counter.inc()
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    
                    if query_failure_counter:
                        query_failure_counter.inc()
                    if query_duration:
                        query_duration.observe(duration)
                    
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            tracer = trace.get_tracer(__name__) if OPENTELEMETRY_AVAILABLE else None
            
            if tracer:
                with tracer.start_as_current_span(metric_name or func.__name__) as span:
                    try:
                        result = func(*args, **kwargs)
                        duration = time.time() - start_time
                        
                        if query_duration:
                            query_duration.observe(duration)
                        if query_success_counter:
                            query_success_counter.inc()
                        
                        span.set_attribute("duration", duration)
                        span.set_attribute("success", True)
                        
                        return result
                        
                    except Exception as e:
                        duration = time.time() - start_time
                        
                        if query_failure_counter:
                            query_failure_counter.inc()
                        if query_duration:
                            query_duration.observe(duration)
                        
                        span.set_attribute("success", False)
                        span.set_attribute("error", str(e))
                        span.set_attribute("duration", duration)
                        
                        raise
            else:
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    if query_duration:
                        query_duration.observe(duration)
                    if query_success_counter:
                        query_success_counter.inc()
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    
                    if query_failure_counter:
                        query_failure_counter.inc()
                    if query_duration:
                        query_duration.observe(duration)
                    
                    raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

def monitor_tool_execution(tool_name: str, tool_type: str):
    """Decorator specifically for tool execution monitoring"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            if tool_execution_counter:
                tool_execution_counter.labels(tool_name=tool_name, tool_type=tool_type).inc()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                if tool_execution_duration:
                    tool_execution_duration.labels(tool_name=tool_name).observe(duration)
                
                return result
                
            except Exception as e:
                if tool_failure_counter:
                    tool_failure_counter.labels(
                        tool_name=tool_name,
                        error_type=type(e).__name__
                    ).inc()
                raise
        
        return wrapper
    
    return decorator

# Health check endpoint
class HealthCheckHandler:
    """Health check handler for monitoring"""
    
    def __init__(self, agent_system):
        self.agent_system = agent_system
        self.start_time = datetime.now()
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "components": {}
        }
        
        # Check reasoning engine
        try:
            if hasattr(self.agent_system, 'reasoning_engine') and self.agent_system.reasoning_engine:
                status["components"]["reasoning_engine"] = {
                    "status": "healthy",
                    "vector_store_available": bool(getattr(self.agent_system.reasoning_engine, 'vector_store', None))
                }
            else:
                status["components"]["reasoning_engine"] = {
                    "status": "not_initialized"
                }
        except Exception as e:
            status["components"]["reasoning_engine"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            status["status"] = "degraded"
        
        # Check memory system
        try:
            if hasattr(self.agent_system, 'memory_system') and self.agent_system.memory_system:
                mem_stats = self.agent_system.memory_system.get_memory_statistics()
                status["components"]["memory_system"] = {
                    "status": "healthy",
                    "episodic_count": mem_stats.get("episodic_count", 0),
                    "semantic_count": mem_stats.get("semantic_count", 0),
                    "working_count": mem_stats.get("working_count", 0)
                }
                
                # Update Prometheus gauges
                if memory_size_gauge:
                    memory_size_gauge.labels(memory_type="episodic").set(mem_stats.get("episodic_count", 0))
                    memory_size_gauge.labels(memory_type="semantic").set(mem_stats.get("semantic_count", 0))
                    memory_size_gauge.labels(memory_type="working").set(mem_stats.get("working_count", 0))
            else:
                status["components"]["memory_system"] = {
                    "status": "not_initialized"
                }
        except Exception as e:
            status["components"]["memory_system"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            status["status"] = "degraded"
        
        # Check adaptive tools
        try:
            if hasattr(self.agent_system, 'adaptive_tools') and self.agent_system.adaptive_tools:
                tool_stats = self.agent_system.adaptive_tools.get_system_statistics()
                status["components"]["adaptive_tools"] = {
                    "status": "healthy",
                    "total_tools": tool_stats.get("total_tools", 0),
                    "available_tools": tool_stats.get("available_tools", 0)
                }
            else:
                status["components"]["adaptive_tools"] = {
                    "status": "not_initialized"
                }
        except Exception as e:
            status["components"]["adaptive_tools"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            status["status"] = "degraded"
        
        # Check multi-agent system
        try:
            if hasattr(self.agent_system, 'multi_agent') and self.agent_system.multi_agent:
                agent_stats = self.agent_system.multi_agent.orchestrator.get_system_statistics()
                status["components"]["multi_agent"] = {
                    "status": "healthy",
                    "total_agents": len(agent_stats.get("agents", {})),
                    "active_tasks": agent_stats.get("tasks", {}).get("in_progress_tasks", 0)
                }
                
                # Update agent gauges
                if active_agents_gauge:
                    for agent_id, agent_info in agent_stats.get("agents", {}).items():
                        if agent_info["status"] == "busy":
                            active_agents_gauge.labels(agent_type=agent_info["type"]).inc()
            else:
                status["components"]["multi_agent"] = {
                    "status": "not_initialized"
                }
        except Exception as e:
            status["components"]["multi_agent"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            status["status"] = "degraded"
        
        # System metrics
        try:
            import psutil
            process = psutil.Process()
            
            status["system"] = {
                "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(interval=0.1),
                "threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "connections": len(process.connections())
            }
            
            # Update system gauges
            if system_memory_gauge:
                system_memory_gauge.set(process.memory_info().rss)
            if system_cpu_gauge:
                system_cpu_gauge.set(process.cpu_percent(interval=0.1))
                
        except ImportError:
            status["system"] = {
                "error": "psutil not available"
            }
        except Exception as e:
            status["system"] = {
                "error": str(e)
            }
        
        return status
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus client not available"
        
        try:
            return generate_latest()
        except Exception as e:
            logger.error(f"Failed to generate metrics: {e}")
            return f"# Error generating metrics: {e}"

# Performance monitoring utilities
class PerformanceMonitor:
    """Enhanced performance monitoring"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a custom metric"""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            'value': value,
            'timestamp': time.time(),
            'labels': labels or {}
        })
    
    def get_metric_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a metric"""
        if name not in self.metrics:
            return {}
        
        values = [m['value'] for m in self.metrics[name]]
        if not values:
            return {}
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'latest': values[-1] if values else None
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all metrics"""
        stats = {}
        for metric_name in self.metrics:
            stats[metric_name] = self.get_metric_stats(metric_name)
        return stats
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.start_time = time.time()

# Memory monitoring
class MemoryMonitor:
    """Memory usage monitoring"""
    
    def __init__(self):
        self.memory_snapshots = []
    
    def take_snapshot(self):
        """Take a memory snapshot"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            snapshot = {
                'timestamp': time.time(),
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
            
            self.memory_snapshots.append(snapshot)
            
            # Keep only last 100 snapshots
            if len(self.memory_snapshots) > 100:
                self.memory_snapshots = self.memory_snapshots[-100:]
            
            return snapshot
            
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
            return None
        except Exception as e:
            logger.error(f"Failed to take memory snapshot: {e}")
            return None
    
    def get_memory_trend(self) -> Dict[str, Any]:
        """Get memory usage trend"""
        if len(self.memory_snapshots) < 2:
            return {}
        
        recent = self.memory_snapshots[-10:]  # Last 10 snapshots
        rss_values = [s['rss_mb'] for s in recent]
        
        return {
            'current_mb': rss_values[-1] if rss_values else 0,
            'average_mb': sum(rss_values) / len(rss_values) if rss_values else 0,
            'trend': 'increasing' if len(rss_values) > 1 and rss_values[-1] > rss_values[0] else 'stable',
            'snapshots_count': len(self.memory_snapshots)
        }

# Initialize monitoring
def initialize_monitoring():
    """Initialize all monitoring components"""
    # Setup tracing
    tracer = setup_tracing()
    
    # Setup GAIA info
    if gaia_info:
        gaia_info.info({
            'version': '1.0.0',
            'components': 'reasoning,memory,tools,multi_agent',
            'monitoring': 'prometheus,opentelemetry'
        })
    
    logger.info("Monitoring system initialized")
    return tracer 