"""
Enhanced error handling and recovery mechanisms for the AI agent.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorCategory(Enum):
    """Enhanced error categories for better error handling."""
    # API and Rate Limiting
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    AUTH = "auth"
    
    # Data and Validation
    TOOL_VALIDATION = "tool_validation"
    NOT_FOUND = "not_found"
    DATA_FORMAT = "data_format"
    
    # Model and Processing
    MODEL_ERROR = "model_error"
    RESOURCE_LIMIT = "resource_limit"
    PROCESSING_TIMEOUT = "processing_timeout"
    
    # Logic and Reasoning
    LOGIC_ERROR = "logic_error"
    INFERENCE_ERROR = "inference_error"
    
    # Default
    GENERAL = "general"

@dataclass
class RetryStrategy:
    """Configuration for retry behavior."""
    max_retries: int
    backoff_factor: float
    timeout_increase: float
    use_exponential_backoff: bool
    should_switch_tool: bool
    should_validate_input: bool
    should_reduce_scope: bool
    should_verify_reasoning: bool

@dataclass
class ToolExecutionResult:
    """Result of a tool execution attempt."""
    success: bool
    output: Any
    error: Optional[Exception]
    error_category: Optional[ErrorCategory]
    retry_suggestions: List[str]

class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")
                
            raise e

class ErrorHandler:
    """Enhanced error handling and recovery system."""
    
    def __init__(self):
        self.error_counts = {}
        self.recovery_history = {}
        self.circuit_breakers = {}  # Add circuit breakers per tool
    
    def categorize_error(self, error_str: str) -> ErrorCategory:
        """Categorize error with enhanced granularity."""
        error_lower = str(error_str).lower()
        
        # API and Rate Limiting
        if "429" in error_lower or "rate limit" in error_lower:
            return ErrorCategory.RATE_LIMIT
        elif "timeout" in error_lower or "connection" in error_lower:
            return ErrorCategory.NETWORK
        elif "authentication" in error_lower or "401" in error_lower:
            return ErrorCategory.AUTH
        
        # Data and Validation
        elif "validation" in error_lower or "invalid" in error_lower:
            return ErrorCategory.TOOL_VALIDATION
        elif "not found" in error_lower or "404" in error_lower:
            return ErrorCategory.NOT_FOUND
        elif "format" in error_lower or "parse" in error_lower:
            return ErrorCategory.DATA_FORMAT
        
        # Model and Processing
        elif "model" in error_lower and "decommissioned" in error_lower:
            return ErrorCategory.MODEL_ERROR
        elif "memory" in error_lower or "resource" in error_lower:
            return ErrorCategory.RESOURCE_LIMIT
        elif "timeout" in error_lower or "deadline" in error_lower:
            return ErrorCategory.PROCESSING_TIMEOUT
        
        # Logic and Reasoning
        elif "logic" in error_lower or "reasoning" in error_lower:
            return ErrorCategory.LOGIC_ERROR
        elif "inference" in error_lower or "prediction" in error_lower:
            return ErrorCategory.INFERENCE_ERROR
        
        return ErrorCategory.GENERAL
    
    def get_retry_strategy(self, error_category: ErrorCategory, state: Dict[str, Any]) -> RetryStrategy:
        """Get sophisticated retry strategy based on error category and state."""
        base_strategy = RetryStrategy(
            max_retries=3,
            backoff_factor=1.5,
            timeout_increase=1.5,
            use_exponential_backoff=True,
            should_switch_tool=False,
            should_validate_input=False,
            should_reduce_scope=False,
            should_verify_reasoning=False
        )
        
        # Customize strategy based on error type
        if error_category == ErrorCategory.RATE_LIMIT:
            return RetryStrategy(
                max_retries=5,
                backoff_factor=2.0,
                timeout_increase=1.0,
                use_exponential_backoff=True,
                should_switch_tool=True,
                should_validate_input=False,
                should_reduce_scope=False,
                should_verify_reasoning=False
            )
        elif error_category == ErrorCategory.NETWORK:
            return RetryStrategy(
                max_retries=3,
                backoff_factor=1.2,
                timeout_increase=1.0,
                use_exponential_backoff=True,
                should_switch_tool=False,
                should_validate_input=False,
                should_reduce_scope=False,
                should_verify_reasoning=False
            )
        elif error_category == ErrorCategory.TOOL_VALIDATION:
            return RetryStrategy(
                max_retries=2,
                backoff_factor=1.0,
                timeout_increase=1.0,
                use_exponential_backoff=False,
                should_switch_tool=True,
                should_validate_input=True,
                should_reduce_scope=False,
                should_verify_reasoning=False
            )
        elif error_category == ErrorCategory.RESOURCE_LIMIT:
            return RetryStrategy(
                max_retries=2,
                backoff_factor=1.0,
                timeout_increase=1.0,
                use_exponential_backoff=False,
                should_switch_tool=True,
                should_validate_input=False,
                should_reduce_scope=True,
                should_verify_reasoning=False
            )
        elif error_category == ErrorCategory.LOGIC_ERROR:
            return RetryStrategy(
                max_retries=2,
                backoff_factor=1.0,
                timeout_increase=1.0,
                use_exponential_backoff=False,
                should_switch_tool=True,
                should_validate_input=False,
                should_reduce_scope=False,
                should_verify_reasoning=True
            )
        
        return base_strategy
    
    def get_retry_suggestions(self, error_category: ErrorCategory) -> List[str]:
        """Get helpful retry suggestions based on error category."""
        suggestions = {
            ErrorCategory.RATE_LIMIT: [
                "Wait before retrying",
                "Use exponential backoff",
                "Consider alternative tool"
            ],
            ErrorCategory.NETWORK: [
                "Check network connection",
                "Retry with longer timeout",
                "Try smaller request"
            ],
            ErrorCategory.TOOL_VALIDATION: [
                "Fix parameter names",
                "Check data types",
                "Review tool documentation"
            ],
            ErrorCategory.RESOURCE_LIMIT: [
                "Reduce request scope",
                "Use simpler analysis",
                "Try alternative approach"
            ],
            ErrorCategory.LOGIC_ERROR: [
                "Verify reasoning steps",
                "Check assumptions",
                "Try different approach"
            ],
            ErrorCategory.GENERAL: [
                "Check tool availability",
                "Review input format",
                "Consider alternative approach"
            ]
        }
        return suggestions.get(error_category, ["Retry with modified input"])
    
    def track_error(self, error_category: ErrorCategory):
        """Track error frequency for adaptive handling."""
        self.error_counts[error_category] = self.error_counts.get(error_category, 0) + 1
    
    def get_error_stats(self) -> Dict[ErrorCategory, int]:
        """Get error statistics for monitoring."""
        return self.error_counts.copy()
    
    def record_recovery(self, error_category: ErrorCategory, success: bool):
        """Record recovery attempt success/failure."""
        if error_category not in self.recovery_history:
            self.recovery_history[error_category] = {"success": 0, "failure": 0}
        
        if success:
            self.recovery_history[error_category]["success"] += 1
        else:
            self.recovery_history[error_category]["failure"] += 1
    
    def get_recovery_stats(self) -> Dict[ErrorCategory, Dict[str, int]]:
        """Get recovery statistics for monitoring."""
        return self.recovery_history.copy()
    
    def get_circuit_breaker(self, tool_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for tool."""
        if tool_name not in self.circuit_breakers:
            self.circuit_breakers[tool_name] = CircuitBreaker()
        return self.circuit_breakers[tool_name] 