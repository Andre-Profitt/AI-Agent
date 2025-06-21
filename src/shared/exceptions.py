from examples.parallel_execution_example import tool_name
from setup_environment import value

from src.api.rate_limiter import rate_limit
from src.api.rate_limiter import retry_after
from src.api_server import message
from src.application.tools.tool_executor import operation
from src.database.models import agent_id
from src.database.models import agent_type
from src.database.models import component
from src.database.models import details
from src.database.models import resource_id
from src.database.models import resource_type
from src.database.models import tool_type
from src.gaia_components.multi_agent_orchestrator import task_id
from src.unified_architecture.enhanced_platform import task_type
from src.utils.http_retry import status_code
from src.utils.tools_introspection import field

from src.tools.base_tool import Tool

from src.agents.advanced_agent_fsm import Agent
# TODO: Fix undefined variables: Any, Dict, Optional, agent_id, agent_type, auth_method, component, config_key, current_state, data_source, data_type, details, endpoint, error_code, expected_state, field, message, operation, required_permissions, resource_id, resource_type, retry_after, status_code, task_id, task_type, timeout_duration, tool_name, tool_type, use_case, value
from src.api.rate_limiter import rate_limit

# TODO: Fix undefined variables: agent_id, agent_type, auth_method, component, config_key, current_state, data_source, data_type, details, endpoint, error_code, expected_state, message, operation, rate_limit, required_permissions, resource_id, resource_type, retry_after, self, status_code, task_id, task_type, timeout_duration, tool_name, tool_type, use_case, value

"""
Custom exceptions for the AI Agent system.

This module defines custom exception classes that are used throughout
the application to provide meaningful error handling and debugging.
"""

from dataclasses import field
from typing import Dict
from typing import Any

from typing import Optional, Dict, Any

class AIAgentException(Exception):
    """Base exception for all AI Agent system exceptions."""

    def __init__(self,
                 message: str,
                 error_code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

class DomainException(AIAgentException):
    """Exception raised for domain/business logic errors."""

    def __init__(self,
                 message: str,
                 error_code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, error_code or "DOMAIN_ERROR", details)

class ValidationException(AIAgentException):
    """Exception raised for validation errors."""

    def __init__(self,
                 message: str,
                 field: Optional[str] = None,
                 value: Optional[Any] = None,
                 details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field
        self.value = value

class InfrastructureException(AIAgentException):
    """Exception raised for infrastructure/technical errors."""

    def __init__(self,
                 message: str,
                 component: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "INFRASTRUCTURE_ERROR", details)
        self.component = component

class ApplicationException(AIAgentException):
    """Exception raised for application layer errors."""

    def __init__(self,
                 message: str,
                 use_case: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "APPLICATION_ERROR", details)
        self.use_case = use_case

class ConfigurationException(AIAgentException):
    """Exception raised for configuration errors."""

    def __init__(self,
                 message: str,
                 config_key: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "CONFIGURATION_ERROR", details)
        self.config_key = config_key

class AgentException(AIAgentException):
    """Exception raised for agent-related errors."""

    def __init__(self,
                 message: str,
                 agent_id: Optional[str] = None,
                 agent_type: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "AGENT_ERROR", details)
        self.agent_id = agent_id
        self.agent_type = agent_type

class TaskException(AIAgentException):
    """Exception raised for task-related errors."""

    def __init__(self,
                 message: str,
                 task_id: Optional[str] = None,
                 task_type: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "TASK_ERROR", details)
        self.task_id = task_id
        self.task_type = task_type

class ToolException(AIAgentException):
    """Exception raised for tool-related errors."""

    def __init__(self,
                 message: str,
                 tool_name: Optional[str] = None,
                 tool_type: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "TOOL_ERROR", details)
        self.tool_name = tool_name
        self.tool_type = tool_type

class CommunicationException(AIAgentException):
    """Exception raised for communication/API errors."""

    def __init__(self,
                 message: str,
                 endpoint: Optional[str] = None,
                 status_code: Optional[int] = None,
                 details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "COMMUNICATION_ERROR", details)
        self.endpoint = endpoint
        self.status_code = status_code

class ResourceException(AIAgentException):
    """Exception raised for resource-related errors."""

    def __init__(self,
                 message: str,
                 resource_type: Optional[str] = None,
                 resource_id: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "RESOURCE_ERROR", details)
        self.resource_type = resource_type
        self.resource_id = resource_id

class TimeoutException(AIAgentException):
    """Exception raised for timeout errors."""

    def __init__(self,
                 message: str,
                 timeout_duration: Optional[float] = None,
                 operation: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "TIMEOUT_ERROR", details)
        self.timeout_duration = timeout_duration
        self.operation = operation

class RateLimitException(AIAgentException):
    """Exception raised for rate limiting errors."""

    def __init__(self,
                 message: str,
                 rate_limit: Optional[int] = None,
                 retry_after: Optional[float] = None,
                 details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "RATE_LIMIT_ERROR", details)
        self.rate_limit = rate_limit
        self.retry_after = retry_after

class AuthenticationException(AIAgentException):
    """Exception raised for authentication errors."""

    def __init__(self,
                 message: str,
                 auth_method: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "AUTHENTICATION_ERROR", details)
        self.auth_method = auth_method

class AuthorizationException(AIAgentException):
    """Exception raised for authorization errors."""

    def __init__(self,
                 message: str,
                 required_permissions: Optional[list] = None,
                 details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "AUTHORIZATION_ERROR", details)
        self.required_permissions = required_permissions

class DataException(AIAgentException):
    """Exception raised for data-related errors."""

    def __init__(self,
                 message: str,
                 data_type: Optional[str] = None,
                 data_source: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "DATA_ERROR", details)
        self.data_type = data_type
        self.data_source = data_source

class StateException(AIAgentException):
    """Exception raised for state management errors."""

    def __init__(self,
                 message: str,
                 current_state: Optional[str] = None,
                 expected_state: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, "STATE_ERROR", details)
        self.current_state = current_state
        self.expected_state = expected_state

# Convenience functions for common error patterns
def raise_validation_error(message: str, field: Optional[str] = None,
                          value: Optional[Any] = None) -> None:
    """Raise a validation exception with the given details."""
    raise ValidationException(message, field, value)

def raise_infrastructure_error(self, message: str, component: Optional[str] = None) -> None:
    """Raise an infrastructure exception with the given details."""
    raise InfrastructureException(message, component)

def raise_agent_error(self, message: str, agent_id: Optional[str] = None,
                     agent_type: Optional[str] = None) -> None:
    """Raise an agent exception with the given details."""
    raise AgentException(message, agent_id, agent_type)

def raise_task_error(self, message: str, task_id: Optional[str] = None,
                    task_type: Optional[str] = None) -> None:
    """Raise a task exception with the given details."""
    raise TaskException(message, task_id, task_type)

def raise_tool_error(self, message: str, tool_name: Optional[str] = None,
                    tool_type: Optional[str] = None) -> None:
    """Raise a tool exception with the given details."""
    raise ToolException(message, tool_name, tool_type)