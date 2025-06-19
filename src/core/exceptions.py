"""
Custom exceptions for the AI Agent system
"""

class AIAgentException(Exception):
    """Base exception for AI Agent system"""
    pass

class ToolExecutionError(AIAgentException):
    """Raised when tool execution fails"""
    pass

class CircuitBreakerOpenError(AIAgentException):
    """Raised when circuit breaker is open"""
    pass

class MaxRetriesExceededError(AIAgentException):
    """Raised when maximum retries are exceeded"""
    pass

class DatabaseConnectionError(AIAgentException):
    """Raised when database connection fails"""
    pass

class ConfigurationError(AIAgentException):
    """Raised when configuration is invalid"""
    pass

class ValidationError(AIAgentException):
    """Raised when validation fails"""
    pass

class TimeoutError(AIAgentException):
    """Raised when operation times out"""
    pass

class ResourceNotFoundError(AIAgentException):
    """Raised when a resource is not found"""
    pass

class PermissionError(AIAgentException):
    """Raised when permission is denied"""
    pass 