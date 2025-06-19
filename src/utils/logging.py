"""
Enhanced logging with structured output
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional
import structlog
from pathlib import Path
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
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

class StructuredLogger:
    """Wrapper for structured logging"""
    
    def __init__(self, name: str) -> None:
        self.logger = structlog.get_logger(name)
        self._context = {}
    
    def bind(self, **kwargs) -> Any:
        """Bind context variables"""
        self._context.update(kwargs)
        return self
    
    def _log(self, level: str, message: str, **kwargs) -> Any:
        """Internal log method"""
        log_data = {
            **self._context,
            **kwargs,
            'timestamp': datetime.utcnow().isoformat(),
            'level': level,
            'message': message
        }
        
        getattr(self.logger, level)(message, **log_data)
    
    def debug(self, message: str, **kwargs) -> Any:
        self._log('debug', message, **kwargs)
    
    def info(self, message: str, **kwargs) -> Any:
        self._log('info', message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> Any:
        self._log('warning', message, **kwargs)
    
    def error(self, message: str, exc_info=False, **kwargs) -> Any:
        if exc_info:
            kwargs['exc_info'] = exc_info
        self._log('error', message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> Any:
        self._log('critical', message, **kwargs)

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_logs: bool = True
) -> None:
    """Setup logging configuration"""
    
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    if json_logs:
        # JSON formatter for structured logs
        formatter = logging.Formatter('%(message)s')
    else:
        # Human-readable formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress noisy libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance"""
    return StructuredLogger(name)

class LoggerAdapter:
    """Adapter for legacy logging compatibility"""
    
    def __init__(self, logger: StructuredLogger) -> None:
        self.logger = logger
    
    def __getattr__(self, name) -> Any:
        return getattr(self.logger, name) 