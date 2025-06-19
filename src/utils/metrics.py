"""
Metrics utility functions
"""

from typing import Dict, Any, Optional
from datetime import datetime
import time
from functools import wraps

# Global metrics storage
_metrics = {}

def track_metric(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Track a custom metric"""
    if name not in _metrics:
        _metrics[name] = []
    
    metric_data = {
        'value': value,
        'timestamp': datetime.utcnow().isoformat(),
        'labels': labels or {}
    }
    
    _metrics[name].append(metric_data)

def get_metrics() -> Dict[str, Any]:
    """Get all tracked metrics"""
    return _metrics.copy()

def clear_metrics():
    """Clear all metrics"""
    _metrics.clear()

def timing_decorator(metric_name: str):
    """Decorator to track function execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                track_metric(f"{metric_name}_duration", duration)
                track_metric(f"{metric_name}_success", 1)
                return result
            except Exception as e:
                duration = time.time() - start_time
                track_metric(f"{metric_name}_duration", duration)
                track_metric(f"{metric_name}_error", 1)
                raise
        return wrapper
    return decorator

def async_timing_decorator(metric_name: str):
    """Decorator to track async function execution time"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                track_metric(f"{metric_name}_duration", duration)
                track_metric(f"{metric_name}_success", 1)
                return result
            except Exception as e:
                duration = time.time() - start_time
                track_metric(f"{metric_name}_duration", duration)
                track_metric(f"{metric_name}_error", 1)
                raise
        return wrapper
    return decorator 