"""
Structured logging utilities for consistent logging across the application
"""

import logging
from typing import Any, Dict, Optional, Union
from datetime import datetime

class StructuredLogger:
    """Wrapper for structured logging with consistent formatting"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def info(self, message: str, **kwargs):
        """Log info with structured data"""
        self.logger.info(message, extra=kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error with structured data"""
        if error:
            kwargs['error_type'] = type(error).__name__
            kwargs['error_message'] = str(error)
        self.logger.error(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning with structured data"""
        self.logger.warning(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug with structured data"""
        self.logger.debug(message, extra=kwargs)
    
    def critical(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log critical with structured data"""
        if error:
            kwargs['error_type'] = type(error).__name__
            kwargs['error_message'] = str(error)
        self.logger.critical(message, extra=kwargs)

def get_structured_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance"""
    return StructuredLogger(name)

def log_function_call(func_name: str, **kwargs):
    """Decorator to log function calls with structured data"""
    def decorator(func):
        def wrapper(*args, **func_kwargs):
            logger = get_structured_logger(func.__module__)
            logger.info("Function call started", 
                       extra={"function": func_name, "args_count": len(args), **kwargs})
            try:
                result = func(*args, **func_kwargs)
                logger.info("Function call completed", 
                           extra={"function": func_name, "status": "success"})
                return result
            except Exception as e:
                logger.error("Function call failed", 
                           error=e, extra={"function": func_name, "status": "error"})
                raise
        return wrapper
    return decorator

def log_async_function_call(func_name: str, **kwargs):
    """Decorator to log async function calls with structured data"""
    def decorator(func):
        async def wrapper(*args, **func_kwargs):
            logger = get_structured_logger(func.__module__)
            logger.info("Async function call started", 
                       extra={"function": func_name, "args_count": len(args), **kwargs})
            try:
                result = await func(*args, **func_kwargs)
                logger.info("Async function call completed", 
                           extra={"function": func_name, "status": "success"})
                return result
            except Exception as e:
                logger.error("Async function call failed", 
                           error=e, extra={"function": func_name, "status": "error"})
                raise
        return wrapper
    return decorator 