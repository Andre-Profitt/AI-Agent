"""
Comprehensive monitoring with Prometheus metrics
"""

import time
import asyncio
from typing import Dict, Any, Optional
from collections import defaultdict
from datetime import datetime
import psutil
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Info

from src.utils.logging import get_logger
from typing import Optional, Dict, Any, List, Union, Tuple

logger = get_logger(__name__)

class MetricsCollector:
    """Collect and expose metrics for monitoring"""
    
    def __init__(self) -> None:
        # Request metrics
        self.request_counter = Counter(
            'ai_agent_requests_total',
            'Total number of requests',
            ['endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'ai_agent_request_duration_seconds',
            'Request duration in seconds',
            ['endpoint']
        )
        
        # Tool metrics
        self.tool_usage = Counter(
            'ai_agent_tool_usage_total',
            'Tool usage count',
            ['tool_name', 'status']
        )
        
        self.tool_duration = Histogram(
            'ai_agent_tool_duration_seconds',
            'Tool execution duration',
            ['tool_name']
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'ai_agent_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.memory_usage = Gauge(
            'ai_agent_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.active_connections = Gauge(
            'ai_agent_active_connections',
            'Number of active connections',
            ['connection_type']
        )
        
        # Error metrics
        self.error_counter = Counter(
            'ai_agent_errors_total',
            'Total number of errors',
            ['error_type', 'component']
        )
        
        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            'ai_agent_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['component']
        )
        
        # Custom metrics storage
        self.custom_metrics = defaultdict(lambda: defaultdict(float))
        
        # Background tasks
        self._tasks = []
        self._running = False
    
    async def start(self) -> None:
        """Start metrics collection"""
        self._running = True
        
        # Start Prometheus HTTP server
        prometheus_client.start_http_server(8000)
        logger.info("Prometheus metrics server started on port 8000")
        
        # Start background tasks
        self._tasks.append(
            asyncio.create_task(self._collect_system_metrics())
        )
    
    async def stop(self) -> None:
        """Stop metrics collection"""
        self._running = False
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
    
    async def _collect_system_metrics(self) -> Any:
        """Collect system metrics periodically"""
        while self._running:
            try:
                # CPU usage
                self.cpu_usage.set(psutil.cpu_percent(interval=1))
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage.set(memory.used)
                
                # Wait before next collection
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error collecting system metrics: {}", extra={"e": e})
    
    def track_request(self, endpoint: str, status: str = "processing") -> Any:
        """Track an incoming request"""
        self.request_counter.labels(endpoint=endpoint, status=status).inc()
    
    def track_request_duration(self, endpoint: str, duration: float) -> Any:
        """Track request duration"""
        self.request_duration.labels(endpoint=endpoint).observe(duration)
    
    def track_success(self, endpoint: str) -> Any:
        """Track successful request"""
        self.request_counter.labels(endpoint=endpoint, status="success").inc()
    
    def track_error(self, component: str, error_type: str) -> Any:
        """Track an error"""
        self.error_counter.labels(
            error_type=error_type,
            component=component
        ).inc()
    
    def track_tool_usage(self, tool_name: str, status: str, duration: float) -> Any:
        """Track tool usage"""
        self.tool_usage.labels(tool_name=tool_name, status=status).inc()
        
        if status == "success":
            self.tool_duration.labels(tool_name=tool_name).observe(duration)
    
    def update_circuit_breaker(self, component: str, state: str) -> bool:
        """Update circuit breaker state"""
        state_map = {"closed": 0, "open": 1, "half-open": 2}
        self.circuit_breaker_state.labels(component=component).set(
            state_map.get(state, -1)
        )
    
    def track_metric(self, name: str, value: float, labels: Optional[Dict] = None) -> Any:
        """Track a custom metric"""
        key = f"{name}:{str(labels)}" if labels else name
        self.custom_metrics[name][key] = value
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics for display"""
        return {
            "system": {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            },
            "custom": dict(self.custom_metrics),
            "prometheus_endpoint": "http://localhost:8000/metrics"
        } 