# Stub for langgraph_resilience_patterns
# This file is a placeholder to resolve import errors in advanced_agent_fsm.py

from enum import Enum, auto
from dataclasses import dataclass

class LoopPreventionState(Enum):
    OK = auto()
    LOOP_DETECTED = auto()
    TERMINATED = auto()

class ToolErrorStrategy(Enum):
    RETRY = auto()
    SKIP = auto()
    FAIL = auto()

class ErrorRecoveryState(Enum):
    NORMAL = auto()
    RECOVERING = auto()
    FAILED = auto()

@dataclass
class ToolExecutionResult:
    result: str = "stub"
    success: bool = True

class StateValidator:
    def __init__(self, *args, **kwargs):
        pass
    def validate(self, *args, **kwargs):
        return True

class ResilientAPIClient:
    def __init__(self, *args, **kwargs):
        pass
    def request(self, *args, **kwargs):
        return None

class PlanResponse:
    pass

class PlanStep:
    pass

def calculate_state_hash(*args, **kwargs):
    return "dummy_hash"

def check_for_stagnation(*args, **kwargs):
    return False

def decrement_loop_counter(*args, **kwargs):
    return None

def categorize_tool_error(*args, **kwargs):
    return None

def create_self_correction_prompt(*args, **kwargs):
    return "Self-correction prompt stub"

def create_adaptive_error_handler(*args, **kwargs):
    return None

# Add any other stubs as needed for compatibility

def circuit_breaker(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

def retry_with_backoff(*args, **kwargs):
    def decorator(func):
        return func
    return decorator 