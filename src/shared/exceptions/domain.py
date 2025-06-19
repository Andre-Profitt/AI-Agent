"""
Domain-specific exceptions for business logic.
"""

from typing import Optional, Dict, Any


class DomainException(Exception):
    """Base exception for domain layer errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ValidationException(DomainException):
    """Exception raised when domain validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        super().__init__(message, "VALIDATION_ERROR")
        self.field = field
        self.value = value
        self.details = {
            "field": field,
            "value": str(value) if value is not None else None
        }


class BusinessRuleException(DomainException):
    """Exception raised when business rules are violated."""
    
    def __init__(self, message: str, rule_name: Optional[str] = None):
        super().__init__(message, "BUSINESS_RULE_VIOLATION")
        self.rule_name = rule_name
        self.details = {"rule_name": rule_name}


class AgentStateException(DomainException):
    """Exception raised for invalid agent state transitions."""
    
    def __init__(self, message: str, current_state: str, attempted_action: str):
        super().__init__(message, "INVALID_AGENT_STATE")
        self.current_state = current_state
        self.attempted_action = attempted_action
        self.details = {
            "current_state": current_state,
            "attempted_action": attempted_action
        }


class ToolExecutionException(DomainException):
    """Exception raised when tool execution fails."""
    
    def __init__(self, message: str, tool_name: str, tool_input: Optional[Dict[str, Any]] = None):
        super().__init__(message, "TOOL_EXECUTION_FAILED")
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.details = {
            "tool_name": tool_name,
            "tool_input": tool_input
        } 