"""
Logging service interface for structured logging operations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum


class LogLevel(str, Enum):
    """Log levels for structured logging"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggingService(ABC):
    """
    Abstract interface for structured logging operations.
    
    This interface defines the contract that all logging service
    implementations must follow, ensuring consistent logging
    across the application.
    """
    
    @abstractmethod
    async def log_debug(self, event: str, message: str, 
                       context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a debug message.
        
        Args:
            event: The event identifier
            message: The log message
            context: Optional context data
        """
        pass
    
    @abstractmethod
    async def log_info(self, event: str, message: str, 
                      context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an info message.
        
        Args:
            event: The event identifier
            message: The log message
            context: Optional context data
        """
        pass
    
    @abstractmethod
    async def log_warning(self, event: str, message: str, 
                         context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a warning message.
        
        Args:
            event: The event identifier
            message: The log message
            context: Optional context data
        """
        pass
    
    @abstractmethod
    async def log_error(self, event: str, message: str, 
                       context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error message.
        
        Args:
            event: The event identifier
            message: The log message
            context: Optional context data
        """
        pass
    
    @abstractmethod
    async def log_critical(self, event: str, message: str, 
                          context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a critical message.
        
        Args:
            event: The event identifier
            message: The log message
            context: Optional context data
        """
        pass
    
    @abstractmethod
    async def log_with_level(self, level: LogLevel, event: str, message: str, 
                           context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a message with a specific level.
        
        Args:
            level: The log level
            event: The event identifier
            message: The log message
            context: Optional context data
        """
        pass
    
    @abstractmethod
    async def log_agent_activity(self, agent_id: str, action: str, 
                               details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log agent activity.
        
        Args:
            agent_id: The agent identifier
            action: The action performed
            details: Optional details about the action
        """
        pass
    
    @abstractmethod
    async def log_task_execution(self, task_id: str, agent_id: str, 
                               status: str, duration: Optional[float] = None,
                               details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log task execution.
        
        Args:
            task_id: The task identifier
            agent_id: The agent identifier
            status: The execution status
            duration: Optional execution duration
            details: Optional execution details
        """
        pass
    
    @abstractmethod
    async def log_system_event(self, event: str, component: str, 
                             message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log system events.
        
        Args:
            event: The event identifier
            component: The system component
            message: The log message
            context: Optional context data
        """
        pass
    
    @abstractmethod
    async def log_performance_metric(self, metric_name: str, value: float, 
                                   unit: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log performance metrics.
        
        Args:
            metric_name: The metric name
            value: The metric value
            unit: The metric unit
            context: Optional context data
        """
        pass
    
    @abstractmethod
    async def log_security_event(self, event: str, severity: str, 
                               message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log security events.
        
        Args:
            event: The security event identifier
            severity: The event severity
            message: The log message
            context: Optional context data
        """
        pass
    
    @abstractmethod
    async def get_logs(self, level: Optional[LogLevel] = None, 
                      event: Optional[str] = None, 
                      start_time: Optional[str] = None,
                      end_time: Optional[str] = None,
                      limit: Optional[int] = None) -> list:
        """
        Retrieve logs with optional filtering.
        
        Args:
            level: Optional log level filter
            event: Optional event filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Optional limit on number of logs
            
        Returns:
            List of log entries
        """
        pass
    
    @abstractmethod
    async def get_log_statistics(self, start_time: Optional[str] = None,
                               end_time: Optional[str] = None) -> Dict[str, Any]:
        """
        Get log statistics for a time period.
        
        Args:
            start_time: Optional start time
            end_time: Optional end time
            
        Returns:
            Dictionary containing log statistics
        """
        pass 