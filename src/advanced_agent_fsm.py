import operator
import logging
import time
import random
import json
import re
import uuid
import requests
import os
import hashlib
from typing import Annotated, List, TypedDict, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager

from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, ValidationError, field_validator
from src.tools.base_tool import BaseTool
from src.reasoning.reasoning_path import ReasoningPath, ReasoningType, AdvancedReasoning
from src.errors.error_category import ErrorCategory, ErrorHandler
from src.data_quality import DataQualityLevel, DataQualityValidator, ValidationResult

# Import resilience patterns
from src.langgraph_resilience_patterns import (
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

# --- INPUT VALIDATION LAYER (Layer B) ---
class ValidatedQuery(BaseModel):
    """Pydantic model for validated user queries with comprehensive sanitization"""
    query: str
    
    @field_validator('query')
    @classmethod
    def query_must_be_valid_and_sanitized(cls, v: str) -> str:
        """
        Ensures the input query is not empty, a placeholder, or potentially malicious.
        This is the first layer of the sanitization gateway.
        """
        if not v or not v.strip():
            raise ValueError("Query cannot be empty.")
        
        # Explicitly block the placeholder literals that caused the original error
        if v.strip() in ["{{", "}}", "{{}}", "{", "}", "{{user_question}}", "{{user_query}}"]:
            raise ValueError(f"Invalid placeholder query received: '{v}'")
        
        # Block control characters except newlines and tabs
        control_chars = ''.join(chr(i) for i in range(32) if i not in [9, 10, 13])
        if any(char in v for char in control_chars):
            raise ValueError("Query contains invalid control characters")
        
        # Basic prompt injection detection
        injection_patterns = [
            r"ignore\s+previous\s+instructions",
            r"disregard\s+all\s+prior",
            r"system\s*:\s*you\s+are",
            r"<\|im_start\|>",
            r"<\|im_end\|>",
            r"\[INST\]",
            r"\[/INST\]"
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f"Query contains potential prompt injection pattern")
        
        # Length constraints
        if len(v.strip()) < 3:
            raise ValueError("Query must be at least 3 characters long")
        
        if len(v) > 10000:
            raise ValueError("Query exceeds maximum length of 10000 characters")
        
        # Strip and normalize whitespace
        return v.strip()

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
                logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
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
            logger.info(f"Initiating API call to {self.base_url}/chat/completions", 
                       extra={'model': model, 'message_count': len(messages)})
            
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
                payload["response_format"] = {"type": "json_object"}
                logger.debug("JSON response format enforced")
                
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
                           extra={'status_code': response.status_code, 
                                 'response_time': response.elapsed.total_seconds()})
                
                return result
                
            except requests.exceptions.HTTPError as http_err:
                logger.error(f"HTTP error occurred: {http_err}", 
                           extra={'status_code': response.status_code, 'response_body': response.text[:500]})
                raise
            except requests.exceptions.ConnectionError as conn_err:
                logger.error(f"Connection error occurred: {conn_err}")
                raise
            except requests.exceptions.Timeout as timeout_err:
                logger.error(f"Request timed out: {timeout_err}")
                raise
            except requests.exceptions.RequestException as req_err:
                logger.error(f"Unexpected request error: {req_err}")
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
            logger.info("Creating structured plan", extra={'query_length': len(query)})
        
            # Create the prompt with explicit JSON requirement
            system_prompt = """You are a strategic planning engine. You MUST respond with a valid JSON object.

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
- web_researcher: {"query": "search query", "source": "wikipedia"}
- semantic_search_tool: {"query": "search query", "filename": "knowledge_base.csv", "top_k": 3}
- python_interpreter: {"code": "python code"}
- tavily_search: {"query": "search query", "max_results": 3}
- file_reader: {"filename": "path/to/file", "lines": -1}
- video_analyzer: {"url": "video_url", "action": "download_info"}
- gaia_video_analyzer: {"video_url": "googleusercontent_url"}

Use EXACT parameter names. Respond ONLY with the JSON object, no markdown or explanations."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Create a plan to answer: {query}"}
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
                
                logger.debug("Raw plan response received", extra={'response_length': len(plan_json)})
                
                # Parse and validate using Pydantic
                try:
                    plan_response = PlanResponse.model_validate_json(plan_json)
                    logger.info("Plan validation successful", 
                               extra={'total_steps': plan_response.total_steps, 
                                     'confidence': plan_response.confidence})
                    return plan_response
                    
                except ValidationError as e:
                    logger.error("Plan validation failed", extra={'validation_errors': str(e), 'raw_response': plan_json[:500]})
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
    """Enhanced FSM-based ReAct agent with improved error handling, reasoning, and data quality."""
    
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
        if not tools:
            raise ValueError("Tools list cannot be empty")
            
        self.tools = tools
        self.tool_names = [tool.name for tool in tools]
        self.model_name = model_name
        self.model_preference = model_preference
        self.use_crew = use_crew
        
        # Set up logging if handler provided
        if log_handler:
            logger.addHandler(log_handler)
            logger.info("Custom log handler added to FSMReActAgent")
        
        # Initialize components with error handling
        try:
            self.error_handler = ErrorHandler()
            self.reasoning = AdvancedReasoning()
            self.quality_validator = DataQualityValidator(quality_level)
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise
        
        # Set reasoning type
        self.reasoning_type = reasoning_type
        
        # Initialize tool registries
        self.tool_registry = ToolRegistry()
        self.mcp_registry = MCPToolRegistry()
        
        # Register tools with documentation
        for tool in tools:
            try:
                tool_doc = ToolDocumentation(
                    tool_name=tool.name,
                    description=tool.description,
                    parameters=tool.args_schema.schema() if hasattr(tool, 'args_schema') else {},
                    examples=[],
                    error_handling={}
                )
                self.tool_registry.register_tool(tool_doc)
                
                announcement = ToolAnnouncement(
                    tool_id=f"tool_{tool.name}",
                    version="1.0.0",
                    capabilities=[
                        ToolCapability(
                            name=tool.name,
                            description=tool.description,
                            input_schema=tool.args_schema.schema() if hasattr(tool, 'args_schema') else {},
                            output_schema={},
                            examples=[]
                        )
                    ],
                    authentication={},
                    rate_limits={"requests_per_minute": 60}
                )
                self.mcp_registry.register_tool(announcement)
            except Exception as e:
                logger.warning(f"Failed to register tool {tool.name}: {str(e)}")
                continue

    def run(self, input_text: str) -> Dict[str, Any]:
        """Run the agent with enhanced error handling, reasoning, and data quality."""
        # Initialize state
        state = AgentState(
            input=input_text,
            tools=self.tools,
            tool_names=self.tool_names,
            tool_results=[],
            current_step=0
        )
        
        # Validate input
        state.validation_result = self.quality_validator.validate_input(input_text)
        if not state.validation_result.is_valid:
            return {
                "status": "error",
                "error": "Invalid input",
                "issues": state.validation_result.issues,
                "suggestions": state.validation_result.suggestions
            }
        
        # Generate reasoning plan
        state.reasoning_path = self.reasoning.generate_plan(
            input_text,
            self.reasoning_type
        )
        
        # Execute reasoning steps
        while state.current_step < state.max_steps:
            try:
                # Get current step
                current_step = state.reasoning_path.steps[state.current_step]
                
                # Execute tool
                tool_result = self._execute_tool_with_retry(
                    current_step.tool_name,
                    current_step.tool_input,
                    state
                )
                
                # Update state
                state.tool_results.append(tool_result)
                state.current_step += 1
                
                # Verify step
                if not self.reasoning.verify_step(current_step, tool_result):
                    state.error_state = ErrorCategory.LOGIC_ERROR
                    break
                
                # Check for completion
                if self._is_complete(tool_result):
                    break
                
            except Exception as e:
                # Handle error
                error_category = self.error_handler.categorize_error(e)
                state.error_state = error_category
                
                # Get retry strategy
                retry_strategy = self.error_handler.get_retry_strategy(
                    error_category,
                    state.current_step
                )
                
                if not retry_strategy.should_retry:
                    break
                
                # Wait for backoff
                time.sleep(retry_strategy.backoff_factor)
        
        # Return final result
        return self._format_result(state)
    
    def _execute_tool_with_retry(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        state: AgentState
    ) -> Dict[str, Any]:
        """Execute a tool with retry logic."""
        # Find tool
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")
        
        # Execute with retry
        result = self.error_handler.execute_with_retry(
            tool.run,
            tool_input,
            max_retries=3,
            backoff_factor=1.0
        )
        
        return result
    
    def _is_complete(self, tool_result: Dict[str, Any]) -> bool:
        """Check if the current step is complete."""
        # Check for final answer
        if "final_answer" in tool_result:
            return True
        
        # Check for error
        if "error" in tool_result:
            return True
        
        return False
    
    def _format_result(self, state: AgentState) -> Dict[str, Any]:
        """Format the final result."""
        if state.error_state:
            return {
                "status": "error",
                "error": state.error_state.value,
                "step": state.current_step,
                "tool_results": state.tool_results
            }
        
        # Get final answer
        final_result = state.tool_results[-1]
        if "final_answer" in final_result:
            return {
                "status": "success",
                "answer": final_result["final_answer"],
                "reasoning_path": state.reasoning_path,
                "tool_results": state.tool_results
            }
        
        return {
            "status": "incomplete",
            "step": state.current_step,
            "tool_results": state.tool_results
        }

def validate_user_prompt(prompt: str) -> bool:
    stripped = prompt.strip()
    return len(stripped) >= 3 and any(c.isalnum() for c in stripped) 

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