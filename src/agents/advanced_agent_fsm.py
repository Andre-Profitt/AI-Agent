from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field

# Remove problematic imports
from app import error_msg
from benchmarks.cot_performance import duration
from benchmarks.cot_performance import recommendations
from examples.advanced.multiagent_api_deployment import capability_filter
from examples.enhanced_unified_example import adapter
from examples.enhanced_unified_example import final_answer
from examples.enhanced_unified_example import health
from examples.enhanced_unified_example import task
from examples.integration.unified_architecture_example import requirements
from examples.parallel_execution_example import results
from examples.parallel_execution_example import tool_calls
from examples.parallel_execution_example import tool_name
from examples.parallel_execution_example import total_time
from fix_security_issues import content
from fix_security_issues import patterns
from migrations.env import config
from performance_dashboard import metric
from performance_dashboard import stats
from tests.e2e.gaia_testing_framework import extracted
from tests.e2e.gaia_testing_framework import system_prompt
from tests.load_test import args
from tests.load_test import headers
from tests.load_test import success
from tests.unit.simple_test import func

from src.agents.crew_enhanced import orchestrator
from src.agents.enhanced_fsm import state
from src.agents.multi_agent_system import reliable_tools
from src.analytics.usage_analyzer import confidence_score
from src.api.auth import payload
from src.api.rate_limiter_enhanced import now
from src.api_server import execution
from src.application.tools.tool_executor import operation
from src.application.tools.tool_executor import validation_result
from src.collaboration.realtime_collaboration import session
from src.config.integrations import api_key
from src.config.integrations import is_valid
from src.config.settings import issues
from src.core.langchain_enhanced import groups
from src.core.langgraph_compatibility import loop
from src.core.monitoring import memory
from src.core.optimized_chain_of_thought import base_confidence
from src.core.optimized_chain_of_thought import n
from src.core.optimized_chain_of_thought import reasoning_type
from src.core.optimized_chain_of_thought import step
from src.core.optimized_chain_of_thought import strategies
from src.core.optimized_chain_of_thought import words
from src.core.services.working_memory import combined_text
from src.core.services.working_memory import dates
from src.database.models import input_data
from src.database.models import model_name
from src.database.models import reasoning_path
from src.database.models import text
from src.database.models import tool
from src.database_extended import error_counts
from src.database_extended import success_rate
from src.gaia_components.adaptive_tool_system import available_tools
from src.gaia_components.advanced_reasoning_engine import overlap
from src.gaia_components.advanced_reasoning_engine import prompt
from src.gaia_components.enhanced_memory_system import current_time
from src.gaia_components.multi_agent_orchestrator import avg_response_time
from src.gaia_components.multi_agent_orchestrator import workflow_id
from src.gaia_components.production_vector_store import model
from src.gaia_components.tool_executor import create_production_tool_executor
from src.infrastructure.gaia_logic import date_patterns
from src.infrastructure.workflow.workflow_engine import current_step
from src.infrastructure.workflow.workflow_engine import initial_state
from src.infrastructure.workflow.workflow_engine import next_step
from src.infrastructure.workflow.workflow_engine import retry_count
from src.meta_cognition import complexity
from src.meta_cognition import confidence
from src.meta_cognition import keywords
from src.meta_cognition import query_lower
from src.meta_cognition import score
from src.query_classifier import entities
from src.query_classifier import sentences
from src.services.integration_hub import get_error_handler
from src.services.integration_hub import get_tool_orchestrator
from src.services.integration_hub import get_unified_registry
from src.services.integration_hub import tool_doc
from src.services.next_gen_integration import correlation_id
from src.services.next_gen_integration import question
from src.templates.template_factory import pattern
from src.tools_introspection import error
from src.unified_architecture.dashboard import total_successes
from src.unified_architecture.enhanced_platform import task_type
from src.utils.error_category import call
from src.utils.error_category import suggestions
from src.utils.knowledge_utils import word
from src.utils.tools_enhanced import numbers
from src.utils.tools_introspection import tool_introspector
from src.workflow.workflow_automation import errors

from dataclasses import field
from typing import Dict
from typing import List
from typing import Optional
from typing import TypedDict
from typing import Any

import operator
import logging
import time

import json
import re
import uuid
import requests
import os

import asyncio
from typing import Annotated, List, TypedDict, Dict, Any, Optional

from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path







from src.tools.base_tool import BaseTool

from src.errors.error_category import ErrorCategory, ErrorHandler


# Import GAIA components
from src.gaia_components.advanced_reasoning_engine import AdvancedReasoningEngine
from src.core.exceptions import ValidationError
from src.gaia_components.adaptive_tool_system import AdaptiveToolSystem
from src.gaia_components.enhanced_memory_system import EnhancedMemorySystem, MemoryPriority
from src.gaia_components.multi_agent_orchestrator import MultiAgentOrchestrator
from src.infrastructure.workflow.workflow_engine import WorkflowStatus
from src.reasoning.reasoning_path import ReasoningPath, ReasoningType
from src.shared.types.di_types import BaseTool
from src.utils.data_quality import DataQualityLevel
from src.utils.data_quality import ValidatedQuery
from src.utils.error_category import ErrorCategory
from src.utils.error_category import ErrorHandler
from src.utils.error_category import ToolExecutionResult
# TODO: Fix undefined variables: Annotated, AnyMessage, BaseModel, FUNCTION_CALLING_MODELS, Field, HTTPAdapter, REASONING_MODELS, Retry, StateGraph, TEXT_GENERATION_MODELS, VERIFICATION_MODELS, adapter, announcement, answer, answer_terms, api_client, api_key, args, available_tools, avg_response_time, base_confidence, base_url, best_tool, burst_allowance, c, call, cap, capability_filter, char, cls, combined_text, common_entities, complexity, conclusion, conclusions, confidence, confidence_score, config, consistent_count, content, control_chars, correlation_id, create_production_tool_executor, current_step, current_text, current_time, date_matches, date_patterns, dates, duration, e, enforce_json, entities, entities1, entities2, error, error_counts, error_msg, errors, evidence_count, execution, extracted, fact, fact1, fact2, facts, factual_indicators, final_answer, found_control_chars, func, get_error_handler, get_tool_orchestrator, get_unified_registry, grouped_facts, groups, headers, health, i, indicator, initial_state, input_data, is_valid, issues, key_terms, keyword, keywords, kwargs, largest_group, log_handler, loop, max_requests_per_minute, memory, memory_context, memory_path, messages, metric, model, model_name, model_preference, n, name_matches, names, neg, negations, next_step, next_text, now, number_matches, numbers, operation, operator, orchestrator, original_request, other_fact, overall_success_rate, overlap, path, pattern, patterns, payload, plan, plan_json, plan_response, plan_steps, pos, primary_term, problematic_patterns, prompt, qtype, quality_level, query, query_lower, question, question_analysis, question_lower, question_terms, reasoning_path, reasoning_type, recommendations, record, relevant_memories, reliable_tools, req_time, requirements, response, result, results, retry_count, retry_strategy, risk_level, sanitized, score, scored_tools, selected_strategy, self, sentence, sentence_lower, sentences, session, sleep_time, state, statements, stats, step, stmt1, stmt2, strategies, stripped, success, success_rate, successful_calls, suggestions, synthesis_input, system_prompt, task, task_type, text, text1, text1_lower, text2, text2_lower, tool, tool_calls, tool_doc, tool_introspector, tool_learning_path, tool_name, tools, top_k, total_checks, total_operations, total_successes, total_time, url_patterns, use_crew, validation_result, verified_facts, word, words, workflow, workflow_id, workflow_steps, x
from src.gaia_components.advanced_reasoning_engine import (
    AdvancedReasoningEngine, ReasoningPath, ReasoningStep
)

from langchain.tools import BaseTool
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError
from sqlalchemy import func
from unittest.mock import call
from src.gaia_components.enhanced_memory_system import MemoryType, MemoryPriority
from src.gaia_components.adaptive_tool_system import ToolType
from src.gaia_components.multi_agent_orchestrator import MultiAgentGAIASystem, MultiAgentOrchestrator
from src.gaia_components.tool_executor import (
    ProductionToolExecutor, create_production_tool_executor
)

# Import resilience patterns
from src.core.langgraph_resilience_patterns import (
    LoopPreventionState,
    calculate_state_hash,
    check_for_stagnation,
    decrement_loop_counter,
    ToolErrorStrategy,
    ToolExecutionResult,
    categorize_tool_error,
    create_self_correction_prompt,
    StateValidator,
    ErrorRecoveryState,
    create_adaptive_error_handler
)

# Import workflow orchestration
from src.infrastructure.workflow.workflow_engine import AgentOrchestrator, WorkflowStatus

# Import resilience libraries with fallbacks
try:
    from circuitbreaker import circuit
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    # Create a no-op decorator if circuitbreaker is not available
    def circuit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    RETRY_AVAILABLE = True
except ImportError:
    RETRY_AVAILABLE = False

# --- PRODUCTION-GRADE STRUCTURED LOGGING CONFIGURATION ---
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s %(process)d %(thread)d',
        },
        'structured': {
            'format': '[%(asctime)s] %(levelname)-8s [%(name)s:%(lineno)d] [%(correlation_id)s] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'structured',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'structured',
            'filename': 'agent_fsm.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'level': 'DEBUG',
        },
    },
    'loggers': {
        'agent_fsm': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

# Apply logging configuration
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('agent_fsm')

# --- CONTEXTUAL LOGGING WITH CORRELATION IDs ---
class CorrelationFilter(logging.Filter):
    """Add correlation ID to log records"""
    def filter(self, record):
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = getattr(self, 'correlation_id', 'no-correlation')
        return True

correlation_filter = CorrelationFilter()
for handler in logger.handlers:
    handler.addFilter(correlation_filter)

@contextmanager
def correlation_context(correlation_id: str):
    """Context manager for correlation ID"""
    correlation_filter.correlation_id = correlation_id
    try:
        yield correlation_id
    finally:
        correlation_filter.correlation_id = 'no-correlation'

# --- ENHANCED FSM STATES FOR FAULT TOLERANCE ---
class FSMState(str, Enum):
    """Enumeration of all possible FSM states including granular failure states"""
    # Core execution states
    PLANNING = "PLANNING"
    AWAITING_PLAN_RESPONSE = "AWAITING_PLAN_RESPONSE"
    VALIDATING_PLAN = "VALIDATING_PLAN"
    TOOL_EXECUTION = "TOOL_EXECUTION"
    SYNTHESIZING = "SYNTHESIZING"
    VERIFYING = "VERIFYING"
    FINISHED = "FINISHED"
    
    # Granular failure states for fault tolerance
    TRANSIENT_API_FAILURE = "TRANSIENT_API_FAILURE"
    PERMANENT_API_FAILURE = "PERMANENT_API_FAILURE"
    INVALID_PLAN_FAILURE = "INVALID_PLAN_FAILURE"
    TOOL_EXECUTION_FAILURE = "TOOL_EXECUTION_FAILURE"
    FINAL_FAILURE = "FINAL_FAILURE"
    ERROR = "ERROR"  # Add ERROR state for backward compatibility

# --- PYDANTIC DATA CONTRACTS FOR API RESPONSES ---
class PlanStep(BaseModel):
    """Ironclad data contract for planning steps"""
    step_name: str = Field(description="The name of the tool or function to execute")
    parameters: Dict[str, Any] = Field(description="Parameters for the tool")
    reasoning: str = Field(description="Brief explanation of why this step is necessary")
    expected_output: str = Field(description="Expected format/type of output")

class PlanResponse(BaseModel):
    """Data contract for complete plan response"""
    steps: List[PlanStep] = Field(description="List of planned steps")
    total_steps: int = Field(description="Total number of steps")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the plan")

class ExecutionResult(BaseModel):
    """Data contract for tool execution results"""
    success: bool = Field(description="Whether execution was successful")
    output: Any = Field(description="The tool's output")
    error_message: Optional[str] = Field(description="Error message if execution failed")
    execution_time: float = Field(description="Time taken for execution")

# --- ENHANCED INPUT VALIDATION WITH DETAILED RESULTS ---
@dataclass
class ValidationResult:
    """Comprehensive validation result with detailed feedback"""
    is_valid: bool
    validation_errors: List[str] = None
    sanitized_input: str = None
    confidence_score: float = 0.0
    risk_level: str = "low"  # low, medium, high
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []
        if self.suggestions is None:
            self.suggestions = []

def validate_user_prompt(prompt: str) -> ValidationResult:
    """
    Enhanced input validation with comprehensive checks and detailed feedback.
    Returns a ValidationResult object with validation status and detailed information.
    """
    errors = []
    suggestions = []
    risk_level = "low"
    confidence_score = 1.0
    
    # Basic validation
    if not prompt or not isinstance(prompt, str):
        return ValidationResult(
            is_valid=False,
            validation_errors=["Input must be a non-empty string"],
            suggestions=["Please provide a valid text input"]
        )
    
    stripped = prompt.strip()
    
    # Length validation
    if len(stripped) < 3:
        errors.append("Input must be at least 3 characters long")
        suggestions.append("Please provide a more detailed question or request")
        confidence_score -= 0.3
    
    # Content validation
    if not any(c.isalnum() for c in stripped):
        errors.append("Input must contain at least one alphanumeric character")
        suggestions.append("Please include meaningful text in your input")
        confidence_score -= 0.4
    
    # Block problematic patterns
    problematic_patterns = [
        ("{{", "Invalid placeholder syntax"),
        ("}}", "Invalid placeholder syntax"),
        ("{{}}", "Empty placeholder"),
        ("{{user_question}}", "Invalid placeholder"),
        ("{{user_query}}", "Invalid placeholder"),
        ("ignore previous instructions", "Prompt injection detected"),
        ("disregard all prior", "Prompt injection detected"),
        ("system: you are", "System prompt injection detected"),
        ("<|im_start|>", "Invalid token sequence"),
        ("<|im_end|>", "Invalid token sequence"),
        ("[INST]", "Invalid instruction format"),
        ("[/INST]", "Invalid instruction format")
    ]
    
    for pattern, error_msg in problematic_patterns:
        if pattern.lower() in stripped.lower():
            errors.append(error_msg)
            risk_level = "high"
            confidence_score -= 0.5
            suggestions.append("Please rephrase your question without special formatting")
    
    # Control character validation
    control_chars = ''.join(chr(i) for i in range(32) if i not in [9, 10, 13])  # Allow tab, newline, carriage return
    found_control_chars = [char for char in stripped if char in control_chars]
    if found_control_chars:
        errors.append(f"Input contains invalid control characters: {found_control_chars}")
        suggestions.append("Please remove any special control characters from your input")
        confidence_score -= 0.2
    
    # Length limits
    if len(stripped) > 10000:
        errors.append("Input is too long (maximum 10,000 characters)")
        suggestions.append("Please shorten your question or break it into smaller parts")
        confidence_score -= 0.3
        risk_level = "medium"
    
    # Repetitive content detection
    if len(set(stripped.split())) < 2:
        errors.append("Input appears to be repetitive or lacks variety")
        suggestions.append("Please provide a more diverse and meaningful input")
        confidence_score -= 0.2
    
    # URL/script injection detection
    url_patterns = [
        r'https?://',
        r'javascript:',
        r'data:text/html',
        r'<script',
        r'</script>'
    ]
    
    for pattern in url_patterns:
        if re.search(pattern, stripped, re.IGNORECASE):
            errors.append("Input contains potentially unsafe content")
            risk_level = "high"
            confidence_score -= 0.6
            suggestions.append("Please avoid including URLs or script content")
    
    # Sanitize input (remove problematic characters but keep content)
    sanitized = stripped
    for pattern, _ in problematic_patterns:
        sanitized = sanitized.replace(pattern, "")
    
    # Remove control characters from sanitized version
    for char in control_chars:
        sanitized = sanitized.replace(char, "")
    
    # Final validation
    is_valid = len(errors) == 0 and len(sanitized) >= 3 and any(c.isalnum() for c in sanitized)
    
    # Adjust confidence based on risk level
    if risk_level == "high":
        confidence_score *= 0.3
    elif risk_level == "medium":
        confidence_score *= 0.7
    
    return ValidationResult(
        is_valid=is_valid,
        validation_errors=errors,
        sanitized_input=sanitized if sanitized != stripped else None,
        confidence_score=max(0.0, confidence_score),
        risk_level=risk_level,
        suggestions=suggestions
    )

# Note: FSMState enum already defined above with granular failure states

# --- Enhanced State Tracking ---

@dataclass
class ToolCall:
    """Represents a single tool call and its output"""
    tool_name: str
    tool_input: Dict[str, Any]
    output: Any
    step: int
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class EnhancedAgentState(TypedDict):
    """
    Enhanced agent state with FSM control fields, correlation tracking, and failure context
    """
    # Correlation and context
    correlation_id: str
    
    # User input (now validated)
    query: str
    input_query: ValidatedQuery  # Added validated query field
    
    # Planning and execution
    plan: str
    master_plan: List[Dict[str, Any]]
    validated_plan: Optional[PlanResponse]
    
    # Tool execution history with state-passing
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]
    step_outputs: Dict[int, Any]
    
    # Final output
    final_answer: str
    
    # --- FSM Control Fields ---
    current_fsm_state: str
    stagnation_counter: int
    max_stagnation: int
    retry_count: int
    max_retries: int
    
    # Failure context
    failure_history: List[Dict[str, Any]]
    circuit_breaker_status: str  # "closed", "open", "half-open"
    last_api_error: Optional[str]
    
    # Verification and confidence
    verification_level: str
    confidence: float
    cross_validation_sources: List[str]
    
    # Messages for LangGraph compatibility
    messages: Annotated[List[AnyMessage], operator.add]
    
    # Error tracking
    errors: List[str]
    
    # Performance tracking
    step_count: int
    start_time: float
    end_time: float
    
    # --- Adaptive Tool Selection ---
    tool_reliability: Dict[str, Dict[str, Any]]
    tool_preferences: Dict[str, List[str]]
    
    # --- LOOP DETECTION FIELDS (Layer C) ---
    turn_count: int
    action_history: List[str]  # Hashes of recent actions
    stagnation_score: int  # Semantic stagnation detection
    error_log: List[Dict[str, Any]]  # Structured error tracking
    error_counts: Dict[str, int]  # Error frequency by type
    
    # --- ENHANCED LOOP PREVENTION FROM RESILIENCE PATTERNS ---
    remaining_loops: int  # State-based counter for guaranteed termination
    last_state_hash: str  # Hash of previous state to detect stagnation
    force_termination: bool  # Force stop flag
    
    # --- TOOL ERROR RECOVERY ---
    tool_errors: List[ToolExecutionResult]  # Detailed tool error tracking
    recovery_attempts: int  # Number of recovery attempts
    fallback_level: int  # Current fallback strategy level
    
    # --- REFLECTION AND QUALITY ASSURANCE ---
    draft_answer: str  # Draft before reflection
    reflection_passed: bool  # Whether reflection check passed
    reflection_issues: List[str]  # Issues found during reflection
    
    # --- HUMAN IN THE LOOP ---
    requires_human_approval: bool
    approval_request: Optional[Dict[str, Any]]
    execution_paused: bool

# --- Model Configuration ---

class ModelConfig:
    """Configuration for different Groq models optimized for specific tasks."""
    
    # Reasoning models - for complex logical thinking and planning
    REASONING_MODELS = {
        "primary": "llama-3.3-70b-versatile",  # High-reasoning model for planning
        "fast": "llama-3.1-8b-instant",       # Quick reasoning for simple tasks
        "deep": "deepseek-r1-distill-llama-70b"  # Deep reasoning for complex analysis
    }
    
    # Function calling models - for tool use and execution
    FUNCTION_CALLING_MODELS = {
        "primary": "llama-3.3-70b-versatile",  # Reliable function calling
        "fast": "llama-3.1-8b-instant",       # Quick tool execution
        "versatile": "llama3-groq-70b-8192-tool-use-preview"  # Specialized for tool use
    }
    
    # Text generation models - for final answers and synthesis
    TEXT_GENERATION_MODELS = {
        "primary": "llama-3.3-70b-versatile",  # High-quality text generation
        "fast": "llama-3.1-8b-instant",       # Quick responses
        "creative": "llama3-groq-70b-8192-creative"  # Creative writing
    }
    
    # Verification models - for fact checking and validation
    VERIFICATION_MODELS = {
        "primary": "llama-3.3-70b-versatile",  # Thorough verification
        "fast": "llama-3.1-8b-instant",       # Quick checks
        "strict": "llama3-groq-70b-8192-verification"  # Strict validation
    }
    
    # Model configurations for different tasks
    MODEL_CONFIGS = {
        "planning": {
            "model": REASONING_MODELS["primary"],
            "temperature": 0.2,
            "max_tokens": 1024,
            "top_p": 0.95
        },
        "tool_execution": {
            "model": FUNCTION_CALLING_MODELS["primary"],
            "temperature": 0.1,
            "max_tokens": 512,
            "top_p": 0.99
        },
        "synthesis": {
            "model": TEXT_GENERATION_MODELS["primary"],
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.9
        },
        "verification": {
            "model": VERIFICATION_MODELS["primary"],
            "temperature": 0.1,
            "max_tokens": 512,
            "top_p": 0.99
        }
    }
    
    @classmethod
    def get_model_config(cls, task_type: str) -> Dict[str, Any]:
        """Get the optimal model configuration for a specific task type"""
        return cls.MODEL_CONFIGS.get(task_type, cls.MODEL_CONFIGS["planning"])
    
    @classmethod
    def get_model_for_task(cls, task_type: str) -> str:
        """Get the optimal model for a specific task type"""
        config = cls.get_model_config(task_type)
        return config["model"]

# --- Structured Output Schemas (Directive 3) ---

class FinalIntegerAnswer(BaseModel):
    """Schema for returning a single, precise integer answer."""
    answer: int = Field(description="The final integer result of the query.")

class FinalStringAnswer(BaseModel):
    """Schema for returning a single, precise string answer."""
    answer: str = Field(description="The final string result of the query.")

class FinalNameAnswer(BaseModel):
    """Schema for returning a person's name as the answer."""
    nominator_name: str = Field(description="The person's name as the answer.")

class VerificationResult(BaseModel):
    """Schema for verification results."""
    is_valid: bool = Field(description="Whether the answer passes verification")
    confidence: float = Field(description="Confidence score between 0 and 1")
    issues: List[str] = Field(description="List of any issues found during verification")

# --- Rate Limiting ---

class RateLimiter:
    """Rate limiter for API calls to prevent hitting rate limits"""
    
    def __init__(self, max_requests_per_minute=60, burst_allowance=10):
        self.max_requests_per_minute = max_requests_per_minute
        self.burst_allowance = burst_allowance
        self.requests = []
    
    def wait_if_needed(self):
        """Wait if needed to comply with rate limits"""
        now = time.time()
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        if len(self.requests) >= self.max_requests_per_minute:
            sleep_time = 60 - (now - self.requests[0])
            if sleep_time > 0:
                logger.info("Rate limiting: sleeping for {} seconds", extra={"sleep_time": sleep_time})
                time.sleep(sleep_time)
        
        self.requests.append(now)

# Global rate limiter instance
rate_limiter = RateLimiter()

# --- RESILIENT COMMUNICATION LAYER ---
class ResilientAPIClient:
    """Production-grade API client with retries, circuit breaker, and structured error handling"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.groq.com/openai/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = self._create_resilient_session()
        self.rate_limiter = RateLimiter()
        
    def _create_resilient_session(self) -> requests.Session:
        """Create a requests session with intelligent retry strategy"""
        session = requests.Session()
        
        if RETRY_AVAILABLE:
            # Configure exponential backoff retry strategy
            retry_strategy = Retry(
                total=3,  # Total number of retries
                status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry
                backoff_factor=1,  # Exponential backoff factor
                allowed_methods=["POST"]  # Only retry POST requests (our API calls)
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            
        # Set default timeout for all requests
        session.request = self._add_timeout(session.request)
        
        return session
    
    def _add_timeout(self, original_request):
        """Add timeout to all requests"""
        def request_with_timeout(*args, **kwargs):
            kwargs.setdefault('timeout', 30)  # 30 second timeout
            return original_request(*args, **kwargs)
        return request_with_timeout
    
    @circuit(failure_threshold=5, recovery_timeout=60, expected_exception=requests.exceptions.RequestException)
    def make_chat_completion(self, messages: List[Dict], model: str = "llama-3.3-70b-versatile", 
                           enforce_json: bool = True, correlation_id: str = None) -> Dict[str, Any]:
        """Make a chat completion request with full resilience patterns"""
        
        correlation_id = correlation_id or str(uuid.uuid4())
        
        with correlation_context(correlation_id):
            logger.info("Initiating API call to {}/chat/completions", self.base_url,
                       extra={})
            
            # Apply rate limiting
            self.rate_limiter.wait_if_needed()
            
            # Prepare the request payload
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.1
            }
            
            # CRITICAL: Enforce JSON response format
            if enforce_json:
                payload["response_format"] = {}
                logger.debug("JSON response format enforced", extra={"self_base_url": self.base_url, "_model_": 'model', "_model_": "model", "_type_": "type"})
                
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers
                )
                
                # Raise exception for HTTP errors
                response.raise_for_status()
                
                result = response.json()
                
                logger.info("API call successful", 
                           extra={'status_code': response.status_code, 'response_time': response.elapsed.total_seconds()})
                
                return result
                
            except requests.exceptions.HTTPError as http_err:
                logger.error("HTTP error occurred: {}", extra={})
                raise
            except requests.exceptions.ConnectionError as conn_err:
                logger.error("Connection error occurred: {}", extra={})
                raise
            except requests.exceptions.Timeout as timeout_err:
                logger.error("Request timed out: {}", extra={})
                raise
            except requests.exceptions.RequestException as req_err:
                logger.error("Unexpected request error: {}", extra={})
                raise

# --- ENHANCED PLANNING WITH STRUCTURED OUTPUT ---
class EnhancedPlanner:
    """Planner that enforces structured output and validates plans"""
    
    def __init__(self, api_client: ResilientAPIClient):
        self.api_client = api_client
        
    def create_structured_plan(self, query: str, context: Dict = None, correlation_id: str = None) -> PlanResponse:
        """Create a plan with enforced structure and validation"""
        
        correlation_id = correlation_id or str(uuid.uuid4())
        
        with correlation_context(correlation_id):
            logger.info("Creating structured plan", extra={})
        
            # Create the prompt with explicit JSON requirement
            system_prompt = """You are a strategic planning engine. You MUST respond with a valid JSON object."

Create a step-by-step plan to answer the user's query. Your response must be a JSON object with this exact structure:

{
  "steps": [
    {
      "step_name": "tool_name",
      "parameters": {"param1": "value1", "param2": "value2"},
      "reasoning": "Why this step is needed",
      "expected_output": "What format/type of output is expected"
    }
  ],
  "total_steps": 2,
  "confidence": 0.8
}

Available tools and their exact parameters:
- web_researcher: {}
- semantic_search_tool: {}
- python_interpreter: {}
- tavily_search: {}
- file_reader: {}
- video_analyzer: {}
- gaia_video_analyzer: {}

Use EXACT parameter names. Respond ONLY with the JSON object, no markdown or explanations."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            try:
                # Make API call with enforced JSON
                response = self.api_client.make_chat_completion(
                    messages=messages,
                    enforce_json=True,
                    correlation_id=correlation_id
                )
                
                # Extract the content
                plan_json = response["choices"][0]["message"]["content"]
                
                logger.debug("Raw plan response received", extra={"correlation_id": correlation_id})
                
                # Parse and validate using Pydantic
                try:
                    plan_response = PlanResponse.model_validate_json(plan_json)
                    logger.info("Plan validation successful", extra={"correlation_id": correlation_id})
                    return plan_response
                    
                except ValidationError as e:
                    logger.error("Plan validation failed", extra={"correlation_id": correlation_id, "e": e})
                    raise ValueError(f"Plan validation failed: {e}")
                    
            except Exception as e:
                logger.error("Plan creation failed", extra={'error': str(e)})
                raise

# --- Main FSM-based Agent ---

@dataclass
class AgentState:
    """State for the FSM-based agent."""
    input: str
    tools: List[BaseTool]
    tool_names: List[str]
    tool_results: List[Dict[str, Any]]
    reasoning_path: Optional[ReasoningPath] = None
    error_state: Optional[ErrorCategory] = None
    validation_result: Optional[ValidationResult] = None
    current_step: int = 0
    max_steps: int = 15
    
    def __post_init__(self):
        """Validate state after initialization"""
        if not self.tools:
            raise ValueError("Tools list cannot be empty")
        if not self.tool_names:
            raise ValueError("Tool names list cannot be empty")
        if not isinstance(self.input, str):
            raise ValueError("Input must be a string")
        if not isinstance(self.tool_results, list):
            raise ValueError("Tool results must be a list")

class FSMReActAgent:
    """Enhanced FSM-based ReAct agent with comprehensive error handling and resilience."""
    
    def __init__(
        self,
        tools: List[BaseTool],
        model_name: str = "llama-3.3-70b-versatile",
        quality_level: DataQualityLevel = DataQualityLevel.THOROUGH,
        reasoning_type: ReasoningType = ReasoningType.LAYERED,
        log_handler: Optional[logging.Handler] = None,
        model_preference: str = "balanced",
        use_crew: bool = False
    ):
        """Initialize the FSM-based ReAct agent with enhanced tool integration."""
        self.tools = tools
        self.model_name = model_name
        self.quality_level = quality_level
        self.reasoning_type = reasoning_type
        self.model_preference = model_preference
        self.use_crew = use_crew
        
        # Initialize unified tool registry integration
        try:
            from src.integration_hub import get_unified_registry, get_tool_orchestrator
            self.unified_registry = get_unified_registry()
            self.tool_orchestrator = get_tool_orchestrator()
            
            # Register tools with unified registry
            for tool in tools:
                self.unified_registry.register(tool)
            
            logger.info("Registered {} tools with unified registry", extra={"len_tools_": len(tools)})
            
        except ImportError:
            logger.warning("Unified tool registry not available, using local registry")
            self.unified_registry = None
            self.tool_orchestrator = None
        
        # Initialize tool introspection
        try:
            from src.tools_introspection import tool_introspector
            self.tool_introspector = tool_introspector
            
            # Register tools with introspector
            for tool in tools:
                if hasattr(tool, 'name'):
                    self.tool_introspector.tool_registry[tool.name] = tool
            
            logger.info("Tool introspection initialized")
            
        except ImportError:
            logger.warning("Tool introspection not available")
            self.tool_introspector = None
        
        # Initialize API client with resilience
        self.api_client = ResilientAPIClient(
            api_key=os.getenv("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1"
        )
        
        # Initialize enhanced planner
        self.planner = EnhancedPlanner(self.api_client)
        
        # Initialize answer synthesizer
        self.synthesizer = AnswerSynthesizer(self.api_client)
        
        # Initialize performance monitor
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize reasoning validator
        self.reasoning_validator = ReasoningValidator()
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter()
        
        # Initialize GAIA components
        self._initialize_gaia_components()
        
        # Initialize error handler with metric awareness
        try:
            from src.integration_hub import get_error_handler
            self.error_handler = get_error_handler()
        except ImportError:
            from src.errors.error_category import ErrorHandler
            self.error_handler = ErrorHandler()
        
        # Initialize workflow orchestrator
        try:
            self.workflow_orchestrator = AgentOrchestrator()
            logger.info("Workflow orchestrator initialized")
        except Exception as e:
            logger.warning("Workflow orchestrator initialization failed: {}", extra={"e": e})
            self.workflow_orchestrator = None
        
        # Setup logging
        if log_handler:
            logger.addHandler(log_handler)
        
        # Initialize FSM graph
        self.graph = self._create_fsm_graph()
        
        # Register this agent with the orchestrator
        if self.workflow_orchestrator:
            asyncio.create_task(self._register_with_orchestrator())
        
        logger.info("FSMReActAgent initialized with {} tools and workflow orchestration", extra={"len_tools_": len(tools)})
    
    async def _register_with_orchestrator(self):
        """Register this agent with the workflow orchestrator"""
        try:
            await self.workflow_orchestrator.register_fsm_agent('default', self)
            logger.info("FSM agent registered with workflow orchestrator")
        except Exception as e:
            logger.error("Failed to register with orchestrator: {}", extra={"e": e})
    
    def select_best_tool(self, task: str, available_tools: List[BaseTool]) -> Optional[BaseTool]:
        """Select the best tool for a task using introspection and reliability metrics."""
        if not available_tools:
            return None
        
        # Use introspector to analyze task requirements if available
        if self.tool_introspector:
            try:
                requirements = self.tool_introspector.analyze_task_requirements(task)
                
                # Score tools based on introspection
                scored_tools = []
                for tool in available_tools:
                    if hasattr(tool, 'name'):
                        score = self.tool_introspector.score_tool_fit(tool, requirements)
                        scored_tools.append((tool, score))
                
                if scored_tools:
                    # Return tool with highest score
                    return max(scored_tools, key=lambda x: x[1])[0]
                    
            except Exception as e:
                logger.warning("Tool introspection failed: {}", extra={"e": e})
        
        # Fallback to reliability-based selection
        if self.unified_registry:
            reliable_tools = self.unified_registry.get_tools_by_reliability(min_success_rate=0.7)
            if reliable_tools:
                # Return first reliable tool
                return reliable_tools[0]
        
        # Final fallback: return first available tool
        return available_tools[0]
    
    def get_tools_for_task(self, task_description: str) -> List[BaseTool]:
        """Get tools suitable for a specific task using unified registry."""
        if self.unified_registry:
            # This would use role-based filtering in a more sophisticated implementation
            return self.unified_registry.get_tools_for_role("general")
        else:
            return self.tools
    
    async def execute_workflow(self, workflow_steps: List[Dict[str, Any]], query: str = None) -> Dict[str, Any]:
        """Execute a workflow using the workflow orchestrator"""
        if not self.workflow_orchestrator:
            raise ValueError("Workflow orchestrator not available")
        
        try:
            # Create a workflow ID
            workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
            
            # Create workflow from steps
            await self.workflow_orchestrator.create_workflow_from_fsm(
                workflow_id, self, workflow_steps
            )
            
            # Execute the workflow
            input_data = {'query': query} if query else {}
            execution = await self.workflow_orchestrator.execute_workflow(
                workflow_id, input_data
            )
            
            return {
                'workflow_id': workflow_id,
                'execution_id': execution.execution_id,
                'status': execution.status.value,
                'result': execution.output_data,
                'success': execution.status == WorkflowStatus.COMPLETED
            }
            
        except Exception as e:
            logger.error("Workflow execution failed: {}", extra={"e": e})
            return {
                'workflow_id': None,
                'execution_id': None,
                'status': 'failed',
                'result': None,
                'success': False,
                'error': str(e)
            }

# --- RETRIEVAL-AUGMENTED TOOL USE (RA-TU) IMPLEMENTATION ---
class ToolDocumentation(BaseModel):
    """Schema for tool documentation used in RA-TU"""
    tool_name: str = Field(description="Name of the tool")
    description: str = Field(description="Natural language description of the tool's purpose")
    parameters: Dict[str, Any] = Field(description="Parameter specifications")
    examples: List[Dict[str, Any]] = Field(description="Example usage patterns")
    error_handling: Dict[str, str] = Field(description="Common error scenarios and resolutions")

class ToolRegistry:
    """Central registry for tool documentation and discovery"""
    def __init__(self):
        self.tool_docs: Dict[str, ToolDocumentation] = {}
        self.tool_embeddings: Dict[str, List[float]] = {}
        
    def register_tool(self, tool_doc: ToolDocumentation):
        """Register a tool with its documentation"""
        self.tool_docs[tool_doc.tool_name] = tool_doc
        # TODO: Generate and store embeddings for semantic search
        
    def get_tool_doc(self, tool_name: str) -> Optional[ToolDocumentation]:
        """Retrieve tool documentation by name"""
        return self.tool_docs.get(tool_name)
        
    def find_relevant_tools(self, task_description: str, top_k: int = 3) -> List[ToolDocumentation]:
        """Find relevant tools for a given task using semantic search"""
        # TODO: Implement semantic search using embeddings
        return list(self.tool_docs.values())[:top_k]

# --- MODEL CONTEXT PROTOCOL (MCP) IMPLEMENTATION ---
class ToolCapability(BaseModel):
    """MCP-compliant tool capability specification"""
    name: str = Field(description="Name of the capability")
    description: str = Field(description="Natural language description")
    input_schema: Dict[str, Any] = Field(description="JSON Schema for inputs")
    output_schema: Dict[str, Any] = Field(description="JSON Schema for outputs")
    examples: List[Dict[str, Any]] = Field(description="Example inputs and outputs")

class ToolAnnouncement(BaseModel):
    """MCP-compliant tool announcement"""
    tool_id: str = Field(description="Unique identifier for the tool")
    version: str = Field(description="Tool version")
    capabilities: List[ToolCapability] = Field(description="List of tool capabilities")
    authentication: Dict[str, Any] = Field(description="Authentication requirements")
    rate_limits: Dict[str, Any] = Field(description="Rate limiting specifications")

class MCPToolRegistry:
    """Registry for MCP-compliant tools"""
    def __init__(self):
        self.tools: Dict[str, ToolAnnouncement] = {}
        
    def register_tool(self, announcement: ToolAnnouncement):
        """Register an MCP-compliant tool"""
        self.tools[announcement.tool_id] = announcement
        
    def discover_tools(self, capability_filter: Optional[str] = None) -> List[ToolAnnouncement]:
        """Discover tools matching capability requirements"""
        if capability_filter:
            return [
                tool for tool in self.tools.values()
                if any(cap.name == capability_filter for cap in tool.capabilities)
            ]
        return list(self.tools.values())

# --- PERFORMANCE MONITORING AND HEALTH TRACKING ---
class PerformanceMonitor:
    """Comprehensive performance monitoring and health tracking system."""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: {
            'count': 0,
            'success': 0,
            'avg_time': 0,
            'errors': [],
            'last_execution': None,
            'min_time': float('inf'),
            'max_time': 0
        })
        self.health_status = {
            'overall_health': 'healthy',
            'last_check': time.time(),
            'issues': []
        }
    
    def track_execution(self, operation: str, success: bool, duration: float, error: str = None):
        """Track execution metrics for an operation."""
        metric = self.metrics[operation]
        metric['count'] += 1
        
        if success:
            metric['success'] += 1
        else:
            metric['errors'].append(error)
        
        # Update timing statistics
        metric['avg_time'] = (
            (metric['avg_time'] * (metric['count'] - 1) + duration) / 
            metric['count']
        )
        metric['min_time'] = min(metric['min_time'], duration)
        metric['max_time'] = max(metric['max_time'], duration)
        metric['last_execution'] = time.time()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health metrics."""
        current_time = time.time()
        issues = []
        
        # Calculate success rates
        total_operations = 0
        total_successes = 0
        
        for operation, metric in self.metrics.items():
            total_operations += metric['count']
            total_successes += metric['success']
            
            # Check for operation-specific issues
            if metric['count'] > 0:
                success_rate = metric['success'] / metric['count']
                if success_rate < 0.8:
                    issues.append(f"Low success rate for {operation}: {success_rate:.2%}")
                
                # Check for stale operations
                if metric['last_execution'] and (current_time - metric['last_execution']) > 3600:
                    issues.append(f"Operation {operation} hasn't been used in over an hour")
        
        overall_success_rate = total_successes / total_operations if total_operations > 0 else 1.0
        avg_response_time = self.calculate_avg_response_time()
        
        # Determine overall health
        if overall_success_rate >= 0.95 and len(issues) == 0:
            health = 'healthy'
        elif overall_success_rate >= 0.8 and len(issues) <= 2:
            health = 'degraded'
        else:
            health = 'unhealthy'
        
        self.health_status.update({
            'overall_health': health,
            'last_check': current_time,
            'issues': issues,
            'success_rate': overall_success_rate,
            'avg_response_time': avg_response_time,
            'total_operations': total_operations
        })
        
        return self.health_status
    
    def calculate_success_rate(self) -> float:
        """Calculate overall success rate across all operations."""
        total_operations = sum(metric['count'] for metric in self.metrics.values())
        total_successes = sum(metric['success'] for metric in self.metrics.values())
        return total_successes / total_operations if total_operations > 0 else 1.0
    
    def calculate_avg_response_time(self) -> float:
        """Calculate average response time across all operations."""
        total_time = sum(metric['avg_time'] * metric['count'] for metric in self.metrics.values())
        total_operations = sum(metric['count'] for metric in self.metrics.values())
        return total_time / total_operations if total_operations > 0 else 0.0
    
    def get_error_distribution(self) -> Dict[str, int]:
        """Get distribution of errors by type."""
        error_counts = defaultdict(int)
        for metric in self.metrics.values():
            for error in metric['errors']:
                error_counts[error] += 1
        return dict(error_counts)
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on performance data."""
        recommendations = []
        
        for operation, metric in self.metrics.items():
            if metric['count'] > 0:
                success_rate = metric['success'] / metric['count']
                
                if success_rate < 0.7:
                    recommendations.append(f"Investigate failures in {operation} (success rate: {success_rate:.2%})")
                
                if metric['avg_time'] > 10.0:
                    recommendations.append(f"Optimize {operation} performance (avg time: {metric['avg_time']:.2f}s)")
        
        return recommendations
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        self.metrics.clear()
        self.health_status = {
            'overall_health': 'healthy',
            'last_check': time.time(),
            'issues': []
        }

# --- REASONING VALIDATOR FOR LOGICAL COHERENCE ---
class ReasoningValidator:
    """Validate reasoning coherence and logical flow."""
    
    def __init__(self):
        self.logical_patterns = {
            'cause_effect': r'(because|therefore|thus|hence|as a result)',
            'comparison': r'(however|but|although|while|whereas)',
            'sequence': r'(first|second|then|next|finally)',
            'condition': r'(if|when|unless|provided that)'
        }
    
    def validate_reasoning_path(self, path: ReasoningPath) -> ValidationResult:
        """Validate reasoning coherence and logical flow."""
        issues = []
        confidence = 1.0
        
        # Check logical flow
        for i, step in enumerate(path.steps[:-1]):
            next_step = path.steps[i + 1]
            if not self.is_logical_transition(step, next_step):
                issues.append(f"Illogical transition at step {i}")
                confidence -= 0.2
        
        # Check evidence support
        if not self.has_sufficient_evidence(path):
            issues.append("Insufficient evidence for conclusion")
            confidence -= 0.3
        
        # Check for circular reasoning
        if self.has_circular_reasoning(path):
            issues.append("Circular reasoning detected")
            confidence -= 0.4
        
        # Check for contradictions
        if self.has_contradictions(path):
            issues.append("Contradictory statements detected")
            confidence -= 0.3
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            validation_errors=issues,
            confidence_score=max(0.0, confidence)
        )
    
    def is_logical_transition(self, current_step, next_step) -> bool:
        """Check if transition between reasoning steps is logical."""
        # Basic logical flow validation
        # This could be enhanced with more sophisticated NLP analysis
        
        # Check for reasonable progression
        if hasattr(current_step, 'conclusion') and hasattr(next_step, 'premise'):
            # Next step should build on current conclusion
            return True
        
        # Check for logical connectors
        if hasattr(current_step, 'reasoning') and hasattr(next_step, 'reasoning'):
            current_text = str(current_step.reasoning).lower()
            next_text = str(next_step.reasoning).lower()
            
            # Look for logical connectors
            for pattern in self.logical_patterns.values():
                if re.search(pattern, current_text) or re.search(pattern, next_text):
                    return True
        
        return True  # Default to True for now
    
    def has_sufficient_evidence(self, path: ReasoningPath) -> bool:
        """Check if reasoning path has sufficient evidence."""
        evidence_count = 0
        
        for step in path.steps:
            if hasattr(step, 'evidence') and step.evidence:
                evidence_count += 1
            elif hasattr(step, 'source') and step.source:
                evidence_count += 1
        
        # Require at least some evidence for complex reasoning
        return evidence_count >= max(1, len(path.steps) // 3)
    
    def has_circular_reasoning(self, path: ReasoningPath) -> bool:
        """Detect circular reasoning patterns."""
        conclusions = []
        
        for step in path.steps:
            if hasattr(step, 'conclusion'):
                conclusion = str(step.conclusion).lower()
                if conclusion in conclusions:
                    return True
                conclusions.append(conclusion)
        
        return False
    
    def has_contradictions(self, path: ReasoningPath) -> bool:
        """Detect contradictory statements in reasoning path."""
        statements = []
        
        for step in path.steps:
            if hasattr(step, 'conclusion'):
                statements.append(str(step.conclusion).lower())
        
        # Simple contradiction detection
        # This could be enhanced with more sophisticated NLP
        for i, stmt1 in enumerate(statements):
            for j, stmt2 in enumerate(statements[i+1:], i+1):
                if self._are_contradictory(stmt1, stmt2):
                    return True
        
        return False
    
    def _are_contradictory(self, stmt1: str, stmt2: str) -> bool:
        """Check if two statements are contradictory."""
        # Simple contradiction detection
        # This is a basic implementation - could be enhanced with NLP
        
        # Check for direct negations
        negations = [
            ('is', 'is not'), ('are', 'are not'), ('was', 'was not'),
            ('true', 'false'), ('correct', 'incorrect'), ('valid', 'invalid')
        ]
        
        for pos, neg in negations:
            if pos in stmt1 and neg in stmt2:
                return True
            if neg in stmt1 and pos in stmt2:
                return True
        
        return False

# --- ANSWER SYNTHESIZER WITH VERIFICATION ---
class AnswerSynthesizer:
    """Synthesize answers with cross-verification and quality assurance."""
    
    def __init__(self, api_client: ResilientAPIClient = None):
        self.api_client = api_client
        self.verification_threshold = 0.8
    
    def synthesize_with_verification(self, results: List[Dict], question: str) -> str:
        """Synthesize answer with cross-verification and quality checks."""
        try:
            # Extract key facts from each result
            facts = []
            for result in results:
                extracted = self.extract_facts(result)
                facts.extend(extracted)
            
            # Cross-verify facts
            verified_facts = self.cross_verify_facts(facts)
            
            # Build coherent answer
            answer = self.build_answer(verified_facts, question)
            
            # Verify answer addresses the question
            if not self.answers_question(answer, question):
                # Retry with different approach
                answer = self.fallback_synthesis(results, question)
            
            return answer
            
        except Exception as e:
            logger.error("Answer synthesis failed: {}", extra={"e": e})
            return self.fallback_synthesis(results, question)
    
    def extract_facts(self, result: Dict) -> List[str]:
        """Extract key facts from a result."""
        facts = []
        
        if isinstance(result, dict):
            # Extract from common result fields
            for field in ['answer', 'result', 'output', 'content']:
                if field in result and result[field]:
                    content = str(result[field])
                    # Extract factual statements
                    sentences = re.split(r'[.!?]+', content)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if len(sentence) > 10 and self._is_factual(sentence):
                            facts.append(sentence)
        
        return facts
    
    def _is_factual(self, sentence: str) -> bool:
        """Check if a sentence appears to be factual."""
        # Simple heuristic for factual statements
        factual_indicators = [
            'is', 'are', 'was', 'were', 'has', 'have', 'had',
            'contains', 'includes', 'consists', 'located', 'found',
            'number', 'count', 'total', 'amount', 'percentage'
        ]
        
        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in factual_indicators)
    
    def cross_verify_facts(self, facts: List[str]) -> List[str]:
        """Cross-verify facts for consistency."""
        if len(facts) <= 1:
            return facts
        
        verified_facts = []
        
        for fact in facts:
            # Check for consistency with other facts
            consistent_count = 0
            total_checks = 0
            
            for other_fact in facts:
                if fact != other_fact:
                    total_checks += 1
                    if self._are_consistent(fact, other_fact):
                        consistent_count += 1
            
            # Fact is verified if it's consistent with majority of other facts
            if total_checks == 0 or consistent_count / total_checks >= 0.5:
                verified_facts.append(fact)
        
        return verified_facts
    
    def _are_consistent(self, fact1: str, fact2: str) -> bool:
        """Check if two facts are consistent."""
        # Simple consistency check
        # This could be enhanced with more sophisticated NLP
        
        # Check for direct contradictions
        if self._are_contradictory(fact1, fact2):
            return False
        
        # Check for similar entities
        entities1 = self._extract_entities(fact1)
        entities2 = self._extract_entities(fact2)
        
        # If they mention the same entities, they should be consistent
        common_entities = set(entities1) & set(entities2)
        return len(common_entities) == 0 or not self._are_contradictory(fact1, fact2)
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        # Simple entity extraction
        # This could be enhanced with proper NER
        entities = []
        
        # Look for capitalized words (simple heuristic)
        words = text.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                entities.append(word.lower())
        
        return entities
    
    def _are_contradictory(self, text1: str, text2: str) -> bool:
        """Check if two texts are contradictory."""
        # Simple contradiction detection
        negations = [
            ('is', 'is not'), ('are', 'are not'), ('was', 'was not'),
            ('true', 'false'), ('correct', 'incorrect'), ('valid', 'invalid')
        ]
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        for pos, neg in negations:
            if pos in text1_lower and neg in text2_lower:
                return True
            if neg in text1_lower and pos in text2_lower:
                return True
        
        return False
    
    def build_answer(self, facts: List[str], question: str) -> str:
        """Build a coherent answer from verified facts."""
        if not facts:
            return "I don't have enough information to answer this question."
        
        # Simple answer construction
        # This could be enhanced with more sophisticated text generation
        
        # Group related facts
        grouped_facts = self._group_related_facts(facts)
        
        # Build answer based on question type
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['how many', 'count', 'number']):
            return self._build_numeric_answer(grouped_facts)
        elif any(word in question_lower for word in ['who', 'person', 'name']):
            return self._build_person_answer(grouped_facts)
        elif any(word in question_lower for word in ['when', 'date', 'time']):
            return self._build_temporal_answer(grouped_facts)
        else:
            return self._build_general_answer(grouped_facts)
    
    def _group_related_facts(self, facts: List[str]) -> Dict[str, List[str]]:
        """Group facts by related topics."""
        groups = defaultdict(list)
        
        for fact in facts:
            # Simple grouping by key terms
            key_terms = self._extract_key_terms(fact)
            if key_terms:
                primary_term = key_terms[0]
                groups[primary_term].append(fact)
        
        return dict(groups)
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        # Simple key term extraction
        # This could be enhanced with proper NLP
        words = text.lower().split()
        key_terms = []
        
        for word in words:
            if len(word) > 3 and word.isalpha():
                key_terms.append(word)
        
        return key_terms[:3]  # Return top 3 terms
    
    def _build_numeric_answer(self, grouped_facts: Dict[str, List[str]]) -> str:
        """Build answer for numeric questions."""
        numbers = []
        
        for facts in grouped_facts.values():
            for fact in facts:
                # Extract numbers
                number_matches = re.findall(r'\d+', fact)
                numbers.extend([int(n) for n in number_matches])
        
        if numbers:
            return f"The answer is {numbers[0]}."  # Return first number found
        else:
            return "I couldn't find a specific number in the available information."
    
    def _build_person_answer(self, grouped_facts: Dict[str, List[str]]) -> str:
        """Build answer for person-related questions."""
        names = []
        
        for facts in grouped_facts.values():
            for fact in facts:
                # Extract capitalized words (simple name detection)
                name_matches = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', fact)
                names.extend(name_matches)
        
        if names:
            return f"The answer is {names[0]}."  # Return first name found
        else:
            return "I couldn't find a specific person's name in the available information."
    
    def _build_temporal_answer(self, grouped_facts: Dict[str, List[str]]) -> str:
        """Build answer for time-related questions."""
        dates = []
        
        for facts in grouped_facts.values():
            for fact in facts:
                # Extract date patterns
                date_patterns = [
                    r'\d{4}',  # Year
                    r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
                    r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY
                ]
                
                for pattern in date_patterns:
                    date_matches = re.findall(pattern, fact)
                    dates.extend(date_matches)
        
        if dates:
            return f"The answer is {dates[0]}."  # Return first date found
        else:
            return "I couldn't find a specific date in the available information."
    
    def _build_general_answer(self, grouped_facts: Dict[str, List[str]]) -> str:
        """Build general answer from facts."""
        if not grouped_facts:
            return "I don't have enough information to answer this question."
        
        # Combine facts from the largest group
        largest_group = max(grouped_facts.values(), key=len)
        
        if len(largest_group) == 1:
            return largest_group[0]
        else:
            # Combine multiple facts
            return " ".join(largest_group[:2])  # Use first two facts
    
    def answers_question(self, answer: str, question: str) -> bool:
        """Check if answer properly addresses the question."""
        # Simple check - could be enhanced with more sophisticated NLP
        
        # Check if answer is not empty
        if not answer or answer.strip() == "":
            return False
        
        # Check if answer contains relevant information
        question_terms = set(self._extract_key_terms(question))
        answer_terms = set(self._extract_key_terms(answer))
        
        # Answer should contain some of the question terms
        overlap = question_terms & answer_terms
        return len(overlap) > 0
    
    def fallback_synthesis(self, results: List[Dict], question: str) -> str:
        """Fallback synthesis when primary method fails."""
        try:
            # Simple fallback - combine all results
            combined_text = ""
            
            for result in results:
                if isinstance(result, dict):
                    for field in ['answer', 'result', 'output', 'content']:
                        if field in result and result[field]:
                            combined_text += str(result[field]) + " "
            
            if combined_text.strip():
                # Return first sentence or first 200 characters
                sentences = re.split(r'[.!?]+', combined_text)
                if sentences:
                    return sentences[0].strip()
                else:
                    return combined_text[:200].strip()
            else:
                return "I couldn't find a satisfactory answer to your question."
                
        except Exception as e:
            logger.error("Fallback synthesis failed: {}", extra={"e": e})
            return "I encountered an error while processing your question. Please try again."

class GAIAAgentState(EnhancedAgentState):
    """GAIA-specific state extensions"""
    question_type: str = "unknown"  # factual, calculation, analysis, etc.
    verification_required: bool = True
    sources_consulted: List[str] = []
    calculation_verified: bool = False
    gaia_confidence_threshold: float = 0.95

def analyze_question_type(query: str) -> Dict[str, str]:
    """Analyze GAIA question type for optimized handling"""
    query_lower = query.lower()
    
    # Question type patterns
    patterns = {
        "counting": ["how many", "count", "number of", "total"],
        "calculation": ["calculate", "compute", "what is", "result of", "divided by", "multiply"],
        "factual_lookup": ["what is", "who is", "when did", "where is"],
        "date_extraction": ["when", "date", "year", "month", "day"],
        "coordinate": ["latitude", "longitude", "coordinates", "location"],
        "chess": ["chess", "move", "game", "board"],
        "music": ["album", "song", "artist", "release", "discography"],
        "country_code": ["country code", "iso", "alpha", "calling code"],
        "multi_step": ["if", "then", "between", "difference", "combined"]
    }
    
    for qtype, keywords in patterns.items():
        if any(keyword in query_lower for keyword in keywords):
            return {"type": qtype, "confidence": 0.8}
    
    return {"type": "general", "confidence": 0.5}

def should_verify_calculation(state: GAIAAgentState) -> bool:
    """Determine if calculation needs verification"""
    return (state.question_type == "calculation" and 
            not state.calculation_verified and 
            state.confidence < state.gaia_confidence_threshold)

def create_gaia_optimized_plan(query: str) -> Dict[str, any]:
    """Create GAIA-optimized execution plan"""
    question_analysis = analyze_question_type(query)
    
    # Select strategy based on type
    strategies = {
        "counting": ["search_primary_source", "extract_data", "count_items", "verify_count"],
        "calculation": ["extract_numbers", "perform_calculation", "verify_result", "format_answer"],
        "factual_lookup": ["search_authoritative_source", "extract_fact", "cross_verify"],
        "date_extraction": ["search_timeline", "extract_date", "validate_format"],
        "coordinate": ["search_location", "extract_coordinates", "validate_format"],
        "chess": ["analyze_position", "validate_moves", "extract_notation"],
        "music": ["search_discography", "extract_metadata", "cross_verify"],
        "country_code": ["lookup_iso_database", "extract_code", "validate_format"],
        "multi_step": ["decompose_question", "solve_subproblems", "combine_results", "validate"]
    }
    
    selected_strategy = strategies.get(question_analysis["type"], strategies["multi_step"])
    
    return {
        "question_type": question_analysis["type"],
        "strategy": selected_strategy,
        "confidence_threshold": 0.95 if question_analysis["type"] == "calculation" else 0.8,
        "verification_required": question_analysis["type"] in ["calculation", "factual_lookup"]
    }

# Complete FSMReActAgent implementation with GAIA integration
def _create_fsm_graph(self):
    """Create the FSM graph with GAIA component integration"""
    # Initialize GAIA components
    self._initialize_gaia_components()
    
    # Create state graph
    workflow = StateGraph(EnhancedAgentState)
    
    # Add nodes
    workflow.add_node("planning", self._planning_node)
    workflow.add_node("awaiting_plan_response", self._awaiting_plan_response_node)
    workflow.add_node("validating_plan", self._validating_plan_node)
    workflow.add_node("tool_execution", self._tool_execution_node)
    workflow.add_node("synthesizing", self._synthesizing_node)
    workflow.add_node("verifying", self._verifying_node)
    workflow.add_node("finished", self._finished_node)
    workflow.add_node("error_recovery", self._error_recovery_node)
    workflow.add_node("human_approval", self._human_approval_node)
    
    # Add edges
    workflow.add_edge("planning", "awaiting_plan_response")
    workflow.add_edge("awaiting_plan_response", "validating_plan")
    workflow.add_edge("validating_plan", "tool_execution")
    workflow.add_edge("tool_execution", "synthesizing")
    workflow.add_edge("synthesizing", "verifying")
    workflow.add_edge("verifying", "finished")
    workflow.add_edge("verifying", "error_recovery")
    workflow.add_edge("error_recovery", "tool_execution")
    workflow.add_edge("error_recovery", "human_approval")
    workflow.add_edge("human_approval", "finished")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "planning",
        self._check_plan_response,
        {
            "awaiting_plan_response": "awaiting_plan_response",
            "error_recovery": "error_recovery"
        }
    )
    
    workflow.add_conditional_edges(
        "validating_plan",
        self._check_plan_validation,
        {
            "tool_execution": "tool_execution",
            "error_recovery": "error_recovery"
        }
    )
    
    workflow.add_conditional_edges(
        "tool_execution",
        self._check_tool_execution,
        {
            "synthesizing": "synthesizing",
            "error_recovery": "error_recovery"
        }
    )
    
    workflow.add_conditional_edges(
        "synthesizing",
        self._check_synthesis,
        {
            "verifying": "verifying",
            "error_recovery": "error_recovery"
        }
    )
    
    workflow.add_conditional_edges(
        "verifying",
        self._check_verification,
        {
            "finished": "finished",
            "error_recovery": "error_recovery"
        }
    )
    
    workflow.add_conditional_edges(
        "error_recovery",
        self._check_error_recovery,
        {
            "tool_execution": "tool_execution",
            "human_approval": "human_approval",
            "finished": "finished"
        }
    )
    
    workflow.add_conditional_edges(
        "human_approval",
        self._check_human_approval,
        {
            "finished": "finished",
            "tool_execution": "tool_execution"
        }
    )
    
    # Set entry point
    workflow.set_entry_point("planning")
    
    return workflow.compile()


def _initialize_gaia_components(self):
    """Initialize GAIA components with proper error handling"""
    try:
        # Initialize reasoning engine
        self.reasoning_engine = AdvancedReasoningEngine(
            model_name=self.model_name
        )
        
        # Initialize memory system
        memory_path = Path("data/agent_memories")
        memory_path.mkdir(parents=True, exist_ok=True)
        self.memory_system = EnhancedMemorySystem(
            persist_directory=str(memory_path)
        )
        
        # Initialize adaptive tool system
        tool_learning_path = Path("data/tool_learning")
        tool_learning_path.mkdir(parents=True, exist_ok=True)
        self.adaptive_tools = AdaptiveToolSystem(
            persist_directory=str(tool_learning_path)
        )
        
        # Initialize multi-agent system
        orchestrator = MultiAgentOrchestrator(max_agents=10)
        self.multi_agent = MultiAgentGAIASystem(orchestrator)
        
        logger.info("GAIA components initialized successfully")
        
    except Exception as e:
        logger.warning(f"Failed to initialize GAIA components: {e}")
        # Set components to None if initialization fails
        self.reasoning_engine = None
        self.memory_system = None
        self.adaptive_tools = None
        self.multi_agent = None


def _planning_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
    """Planning node with GAIA reasoning engine"""
    correlation_id = state.get("correlation_id", str(uuid.uuid4()))
    
    with correlation_context(correlation_id):
        try:
            query = state.get("query", "")
            
            # Use GAIA reasoning engine if available
            if self.reasoning_engine:
                reasoning_path = self.reasoning_engine.generate_reasoning_path(query)
                state["reasoning_path"] = reasoning_path
                
                # Create plan based on reasoning
                plan_steps = []
                for step in reasoning_path.steps:
                    plan_steps.append(PlanStep(
                        step_name=step.step_type,
                        parameters={"content": step.content},
                        reasoning=step.content,
                        expected_output="reasoning_result"
                    ))
                
                plan = PlanResponse(
                    steps=plan_steps,
                    total_steps=len(plan_steps),
                    confidence=reasoning_path.confidence
                )
            else:
                # Fallback to original planning
                plan = self.planner.create_structured_plan(query, correlation_id=correlation_id)
            
            state["validated_plan"] = plan
            state["current_fsm_state"] = "AWAITING_PLAN_RESPONSE"
            
            logger.info("Planning completed", extra={"correlation_id": correlation_id})
            
        except Exception as e:
            logger.error(f"Planning failed: {e}", extra={"correlation_id": correlation_id})
            state["errors"].append(f"Planning error: {e}")
            state["current_fsm_state"] = "ERROR_RECOVERY"
        
        return state


def _awaiting_plan_response_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
    """Await plan response node"""
    state["current_fsm_state"] = "VALIDATING_PLAN"
    return state


def _validating_plan_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
    """Validate plan node"""
    plan = state.get("validated_plan")
    if plan and plan.confidence > 0.5:
        state["current_fsm_state"] = "TOOL_EXECUTION"
    else:
        state["current_fsm_state"] = "ERROR_RECOVERY"
    return state


def _tool_execution_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
    """Tool execution node with production tool executor"""
    correlation_id = state.get("correlation_id", str(uuid.uuid4()))
    
    with correlation_context(correlation_id):
        try:
            plan = state.get("validated_plan")
            if not plan:
                raise ValueError("No plan available for execution")
            
            # Initialize production tool executor if not already done
            if not hasattr(self, 'tool_executor'):
                self.tool_executor = create_production_tool_executor(max_workers=10)
                
                # Register built-in tools
                for tool in self.tools:
                    if hasattr(tool, 'name'):
                        self.tool_executor.register_tool(
                            tool.name, 
                            tool, 
                            tool_type=getattr(tool, 'tool_type', 'custom')
                        )
            
            tool_calls = []
            
            # Execute tools synchronously within the FSM context
            for step in plan.steps:
                # Use adaptive tool system if available
                if self.adaptive_tools:
                    # Get tool recommendations
                    recommendations = self.adaptive_tools.recommend_tools_for_task(
                        step.reasoning, max_recommendations=3
                    )
                    
                    if recommendations:
                        # Use the best recommended tool
                        best_tool, confidence = recommendations[0]
                        
                        if confidence > 0.5:
                            # Execute tool with production executor (sync wrapper)
                            try:
                                # Create a sync wrapper for async execution
                                loop = asyncio.get_event_loop()
                                result = loop.run_until_complete(
                                    self.tool_executor.execute(best_tool.id, **step.parameters)
                                )
                                
                                tool_calls.append({
                                    "tool": best_tool.name,
                                    "input": step.parameters,
                                    "output": result.result if result.status.value == "success" else None,
                                    "success": result.status.value == "success",
                                    "confidence": confidence,
                                    "execution_time": result.execution_time,
                                    "error": result.error if result.status.value != "success" else None
                                })
                            except Exception as e:
                                tool_calls.append({
                                    "tool": best_tool.name,
                                    "input": step.parameters,
                                    "output": None,
                                    "success": False,
                                    "confidence": confidence,
                                    "error": str(e)
                                })
                else:
                    # Fallback to direct tool execution
                    tool_name = step.step_name
                    if tool_name in [tool.name for tool in self.tools]:
                        # Execute with production executor (sync wrapper)
                        try:
                            loop = asyncio.get_event_loop()
                            result = loop.run_until_complete(
                                self.tool_executor.execute(tool_name, **step.parameters)
                            )
                            
                            tool_calls.append({
                                "tool": tool_name,
                                "input": step.parameters,
                                "output": result.result if result.status.value == "success" else None,
                                "success": result.status.value == "success",
                                "execution_time": result.execution_time,
                                "error": result.error if result.status.value != "success" else None
                            })
                        except Exception as e:
                            tool_calls.append({
                                "tool": tool_name,
                                "input": step.parameters,
                                "output": None,
                                "success": False,
                                "error": str(e)
                            })
                    else:
                        # Handle unknown tool
                        tool_calls.append({
                            "tool": tool_name,
                            "input": step.parameters,
                            "output": None,
                            "success": False,
                            "error": f"Tool {tool_name} not found"
                        })
            
            state["tool_calls"].extend(tool_calls)
            state["current_fsm_state"] = "SYNTHESIZING"
            
            # Log execution statistics
            if hasattr(self.tool_executor, 'get_execution_stats'):
                stats = self.tool_executor.get_execution_stats()
                logger.info(f"Tool execution completed with {len(tool_calls)} calls. Stats: {stats}", 
                           extra={"correlation_id": correlation_id})
            else:
                logger.info(f"Tool execution completed with {len(tool_calls)} calls", 
                           extra={"correlation_id": correlation_id})
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}", extra={"correlation_id": correlation_id})
            state["errors"].append(f"Tool execution error: {e}")
            state["current_fsm_state"] = "ERROR_RECOVERY"
        
        return state


def _synthesizing_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
    """Synthesizing node with memory system integration"""
    correlation_id = state.get("correlation_id", str(uuid.uuid4()))
    
    with correlation_context(correlation_id):
        try:
            query = state.get("query", "")
            tool_calls = state.get("tool_calls", [])
            
            # Extract results from tool calls
            results = [call["output"] for call in tool_calls if call.get("success")]
            
            # Use memory system if available
            if self.memory_system:
                # Store episodic memory
                self.memory_system.store_episodic(
                    content=f"Query: {query}, Results: {str(results)}",
                    event_type="query_processing",
                    metadata={
                        "query": query,
                        "tools_used": [call["tool"] for call in tool_calls],
                        "success_count": len(results)
                    }
                )
                
                # Retrieve relevant memories
                relevant_memories = self.memory_system.search_memories(query, k=3)
                
                # Enhance synthesis with memory context
                memory_context = ""
                if relevant_memories:
                    memory_context = "\n\nRelevant past experiences:\n"
                    for memory in relevant_memories:
                        memory_context += f"- {memory.content}\n"
                
                # Create enhanced synthesis
                synthesis_input = f"Query: {query}\nResults: {str(results)}{memory_context}"
                
                # Store in working memory
                self.memory_system.store_working(
                    content=synthesis_input,
                    priority=MemoryPriority.HIGH
                )
            else:
                synthesis_input = f"Query: {query}\nResults: {str(results)}"
            
            # Use answer synthesizer
            final_answer = self.synthesizer.synthesize_with_verification(results, query)
            
            state["final_answer"] = final_answer
            state["current_fsm_state"] = "VERIFYING"
            
            logger.info("Synthesis completed", extra={"correlation_id": correlation_id})
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}", extra={"correlation_id": correlation_id})
            state["errors"].append(f"Synthesis error: {e}")
            state["current_fsm_state"] = "ERROR_RECOVERY"
        
        return state


def _verifying_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
    """Verifying node"""
    # Simple verification - check if we have a final answer
    if state.get("final_answer"):
        state["current_fsm_state"] = "FINISHED"
    else:
        state["current_fsm_state"] = "ERROR_RECOVERY"
    return state


def _finished_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
    """Finished node"""
    state["end_time"] = time.time()
    state["current_fsm_state"] = "FINISHED"
    return state


def _error_recovery_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
    """Error recovery node"""
    state["retry_count"] = state.get("retry_count", 0) + 1
    
    if state["retry_count"] < state.get("max_retries", 3):
        state["current_fsm_state"] = "PLANNING"
    else:
        state["current_fsm_state"] = "FINISHED"
    
    return state


def _human_approval_node(self, state: EnhancedAgentState) -> EnhancedAgentState:
    """Human approval node"""
    state["requires_human_approval"] = True
    state["current_fsm_state"] = "FINISHED"
    return state


# Conditional edge functions
def _check_plan_response(self, state: EnhancedAgentState) -> str:
    """Check if plan response is ready"""
    if state.get("validated_plan"):
        return "awaiting_plan_response"
    return "error_recovery"


def _check_plan_validation(self, state: EnhancedAgentState) -> str:
    """Check if plan validation passed"""
    plan = state.get("validated_plan")
    if plan and plan.confidence > 0.5:
        return "tool_execution"
    return "error_recovery"


def _check_tool_execution(self, state: EnhancedAgentState) -> str:
    """Check if tool execution completed"""
    tool_calls = state.get("tool_calls", [])
    if tool_calls and any(call.get("success") for call in tool_calls):
        return "synthesizing"
    return "error_recovery"


def _check_synthesis(self, state: EnhancedAgentState) -> str:
    """Check if synthesis completed"""
    if state.get("final_answer"):
        return "verifying"
    return "error_recovery"


def _check_verification(self, state: EnhancedAgentState) -> str:
    """Check if verification passed"""
    if state.get("final_answer"):
        return "finished"
    return "error_recovery"


def _check_error_recovery(self, state: EnhancedAgentState) -> str:
    """Check error recovery strategy"""
    retry_count = state.get("retry_count", 0)
    if retry_count < 3:
        return "tool_execution"
    elif retry_count < 5:
        return "human_approval"
    else:
        return "finished"


def _check_human_approval(self, state: EnhancedAgentState) -> str:
    """Check human approval status"""
    if state.get("requires_human_approval"):
        return "finished"
    return "tool_execution"


def _calculate_complexity(self, query: str) -> float:
    """Calculate query complexity"""
    # Simple complexity calculation
    words = query.split()
    complexity = len(words) * 0.1
    
    # Add complexity for special patterns
    if any(word in query.lower() for word in ["calculate", "compute", "analyze"]):
        complexity += 0.3
    if any(word in query.lower() for word in ["compare", "difference", "between"]):
        complexity += 0.2
    
    return min(complexity, 1.0)


def _is_valid_tool(self, tool_name: str) -> bool:
    """Check if tool is valid"""
    return any(tool.name == tool_name for tool in self.tools)


def _mock_tool_wrapper(self, tool_name: str, **kwargs):
    """Mock tool wrapper for testing"""
    return {"result": f"Mock result from {tool_name}", "success": True}


async def run(self, query: str, correlation_id: str = None) -> Dict[str, Any]:
    """Run the GAIA-enhanced FSM agent"""
    if not correlation_id:
        correlation_id = str(uuid.uuid4())
    
    with correlation_context(correlation_id):
        try:
            # Validate input
            validation_result = validate_user_prompt(query)
            if not validation_result.is_valid:
                return {
                    "success": False,
                    "error": f"Invalid input: {validation_result.validation_errors}",
                    "correlation_id": correlation_id
                }
            
            # Initialize state
            initial_state = {
                "correlation_id": correlation_id,
                "query": query,
                "input_query": validation_result,
                "plan": "",
                "master_plan": [],
                "validated_plan": None,
                "tool_calls": [],
                "step_outputs": {},
                "final_answer": "",
                "current_fsm_state": "PLANNING",
                "stagnation_counter": 0,
                "max_stagnation": 5,
                "retry_count": 0,
                "max_retries": 3,
                "failure_history": [],
                "circuit_breaker_status": "closed",
                "last_api_error": None,
                "verification_level": "thorough",
                "confidence": 0.0,
                "cross_validation_sources": [],
                "messages": [],
                "errors": [],
                "step_count": 0,
                "start_time": time.time(),
                "end_time": 0.0,
                "tool_reliability": {},
                "tool_preferences": {},
                "turn_count": 0,
                "action_history": [],
                "stagnation_score": 0,
                "error_log": [],
                "error_counts": {},
                "remaining_loops": 15,
                "last_state_hash": "",
                "force_termination": False,
                "tool_errors": [],
                "recovery_attempts": 0,
                "fallback_level": 0,
                "draft_answer": "",
                "reflection_passed": False,
                "reflection_issues": [],
                "requires_human_approval": False,
                "approval_request": None,
                "execution_paused": False
            }
            
            # Execute FSM
            result = await self.graph.ainvoke(initial_state)
            
            # Extract final answer
            final_answer = result.get("final_answer", "")
            
            # Clean up answer
            if final_answer:
                final_answer = self._extract_clean_answer(final_answer)
            
            # Calculate confidence
            confidence = self._calculate_confidence(result)
            
            # Prepare response
            response = {
                "success": True,
                "answer": final_answer,
                "confidence": confidence,
                "correlation_id": correlation_id,
                "tool_calls": result.get("tool_calls", []),
                "errors": result.get("errors", []),
                "start_time": result.get("start_time", 0),
                "end_time": result.get("end_time", 0),
                "execution_time": result.get("end_time", 0) - result.get("start_time", 0)
            }
            
            # Store in memory if available
            if self.memory_system:
                asyncio.create_task(
                    self.memory_system.store_episodic(
                        query=query,
                        response=final_answer,
                        tools_used=[call["tool"] for call in response["tool_calls"]],
                        success=response["success"]
                    )
                )
            
            logger.info("GAIA agent execution completed", extra={"correlation_id": correlation_id, "confidence": confidence})
            
            return response
            
        except Exception as e:
            logger.error(f"GAIA agent execution failed: {e}", extra={"correlation_id": correlation_id})
            
            return {
                "success": False,
                "error": str(e),
                "correlation_id": correlation_id,
                "answer": "",
                "confidence": 0.0
            }


def _extract_clean_answer(self, answer: str) -> str:
    """Extract clean answer without formatting artifacts"""
    # Remove LaTeX boxing
    answer = re.sub(r'\\boxed\{([^}]+)\}', r'\1', answer)
    
    # Remove "final answer" prefixes
    answer = re.sub(r'^[Tt]he\s+final\s+answer\s+is\s*:?\s*', '', answer)
    answer = re.sub(r'^[Ff]inal\s+answer\s*:?\s*', '', answer)
    
    # Remove tool call artifacts
    answer = re.sub(r'<tool_call>.*?</tool_call>', '', answer, flags=re.DOTALL)
    
    # Clean up whitespace
    answer = answer.strip()
    
    return answer


def _calculate_confidence(self, result: Dict[str, Any]) -> float:
    """Calculate confidence score"""
    base_confidence = 0.5
    
    # Increase confidence for successful tool calls
    tool_calls = result.get("tool_calls", [])
    if tool_calls:
        successful_calls = sum(1 for call in tool_calls if call.get("success"))
        base_confidence += (successful_calls / len(tool_calls)) * 0.3
    
    # Decrease confidence for errors
    errors = result.get("errors", [])
    if errors:
        base_confidence -= len(errors) * 0.1
    
    # Increase confidence for longer answers
    final_answer = result.get("final_answer", "")
    if len(final_answer) > 10:
        base_confidence += 0.1
    
    return max(0.0, min(1.0, base_confidence))