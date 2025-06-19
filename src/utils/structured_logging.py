"""
Enhanced structured logging with zero f-strings and no print statements
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional, Union
from pathlib import Path
import structlog
from functools import wraps
import asyncio
from typing import Optional, Dict, Any, List, Union, Tuple

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.dict_tracebacks,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

class StructuredLogger:
    """Production-grade structured logger with no f-strings"""
    
    def __init__(self, name: str) -> None:
        self.logger = structlog.get_logger(name)
        self._context = {}
        self.name = name
    
    def bind(self, **kwargs) -> 'StructuredLogger':
        """Bind context variables"""
        self._context.update(kwargs)
        self.logger = self.logger.bind(**kwargs)
        return self
    
    def unbind(self, *keys) -> 'StructuredLogger':
        """Remove context variables"""
        for key in keys:
            self._context.pop(key, None)
        self.logger = self.logger.unbind(*keys)
        return self
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message without f-strings"""
        # This is for the message only, not for interpolation
        return message
    
    def debug(self, message: str, **kwargs) -> Any:
        """Debug level logging"""
        self.logger.debug(message, **self._merge_context(kwargs))
    
    def info(self, message: str, **kwargs) -> Any:
        """Info level logging"""
        self.logger.info(message, **self._merge_context(kwargs))
    
    def warning(self, message: str, **kwargs) -> Any:
        """Warning level logging"""
        self.logger.warning(message, **self._merge_context(kwargs))
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs) -> Any:
        """Error level logging with exception support"""
        if error:
            kwargs['error_type'] = type(error).__name__
            kwargs['error_message'] = str(error)
            kwargs['exc_info'] = error
        self.logger.error(message, **self._merge_context(kwargs))
    
    def critical(self, message: str, **kwargs) -> Any:
        """Critical level logging"""
        self.logger.critical(message, **self._merge_context(kwargs))
    
    def _merge_context(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge permanent context with call-specific context"""
        merged = self._context.copy()
        merged.update(kwargs)
        merged['logger_name'] = self.name
        merged['timestamp'] = datetime.utcnow().isoformat()
        return merged
    
    def log_performance(self, operation: str, duration: float, **kwargs) -> Any:
        """Log performance metrics"""
        self.info("Performance metric", **{
            "operation": operation,
            "duration_seconds": duration,
            "performance_category": "timing",
            **kwargs
        })
    
    def log_api_call(self, method: str, endpoint: str, status_code: int, 
                     duration: float, **kwargs) -> Any:
        """Log API call details"""
        self.info("API call completed", **{
            "http_method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "duration_seconds": duration,
            "api_category": "external",
            **kwargs
        })

# Global logger factory
_loggers: Dict[str, StructuredLogger] = {}

def get_structured_logger(name: str) -> StructuredLogger:
    """Get or create a structured logger"""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)
    return _loggers[name]

# Decorator for automatic performance logging
def log_performance(operation: str = None) -> Any:
    """Decorator to automatically log function performance"""
    def decorator(func) -> Any:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            op_name = operation or func.__name__
            logger = get_structured_logger(func.__module__)
            
            start_time = datetime.utcnow()
            try:
                result = await func(*args, **kwargs)
                duration = (datetime.utcnow() - start_time).total_seconds()
                logger.log_performance(op_name, duration, status="success")
                return result
            except Exception as e:
                duration = (datetime.utcnow() - start_time).total_seconds()
                logger.log_performance(op_name, duration, status="error", error=str(e))
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            op_name = operation or func.__name__
            logger = get_structured_logger(func.__module__)
            
            start_time = datetime.utcnow()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.utcnow() - start_time).total_seconds()
                logger.log_performance(op_name, duration, status="success")
                return result
            except Exception as e:
                duration = (datetime.utcnow() - start_time).total_seconds()
                logger.log_performance(op_name, duration, status="error", error=str(e))
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

# For backward compatibility
def get_logger(name: str) -> StructuredLogger:
    """Backward compatible logger getter"""
    return get_structured_logger(name) 