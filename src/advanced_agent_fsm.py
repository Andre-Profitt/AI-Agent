import operator
import logging
import time
import random
import json
import re
import uuid
import requests
import os
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
from pydantic import BaseModel, Field, ValidationError

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
    
    # User input
    query: str
    
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

# --- Model Configuration ---

class ModelConfig:
    """Configuration for different Groq models optimized for specific tasks."""
    
    # Reasoning models - for complex logical thinking
    REASONING_MODELS = {
        "primary": "llama-3.3-70b-versatile",
        "fast": "llama-3.1-8b-instant",
        "deep": "deepseek-r1-distill-llama-70b"
    }
    
    # Function calling models - for tool use
    FUNCTION_CALLING_MODELS = {
        "primary": "llama-3.3-70b-versatile",
        "fast": "llama-3.1-8b-instant",
        "versatile": "llama3-groq-70b-8192-tool-use-preview"
    }
    
    # Text generation models - for final answers
    TEXT_GENERATION_MODELS = {
        "primary": "llama-3.3-70b-versatile",
        "fast": "llama-3.1-8b-instant",
        "creative": "gemma2-9b-it"
    }

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

class FSMReActAgent:
    """
    Finite State Machine based ReAct agent with deterministic control flow,
    proactive state-passing, and production-ready reliability.
    """
    
    def __init__(self, tools: list, log_handler: logging.Handler = None, model_preference: str = "balanced", use_crew: bool = False):
        self.tools = tools
        self.log_handler = log_handler
        self.tool_registry = {tool.name: tool for tool in tools}
        self.model_preference = model_preference
        self.use_crew = use_crew
        
        # Initialize resilient communication components
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.warning("GROQ_API_KEY not found, some features may be limited")
            # Create a mock client for testing
            self.api_client = None
            self.planner = None
        else:
            self.api_client = ResilientAPIClient(groq_api_key)
            self.planner = EnhancedPlanner(self.api_client)
        
        # Add stub tools for missing ones
        self._add_stub_tools()
        
        # FSM Configuration
        self.max_stagnation = 3
        self.max_steps = 20
        
        try:
            logger.info(f"Initializing FSMReActAgent with {len(tools)} tools")
            with correlation_context(str(uuid.uuid4())):
                logger.info("Building resilient FSM graph", extra={'tools_count': len(tools), 'use_crew': use_crew})
                self.graph = self._build_fsm_graph()
                logger.info("FSMReActAgent graph built successfully")
        except Exception as e:
            logger.error(f"Failed to build FSM graph: {e}", exc_info=True)
            raise RuntimeError(f"FSMReActAgent initialization failed: {e}")
    
    def _add_stub_tools(self):
        """FIX 3: Add stub implementations for missing tools referenced in logs."""
        stub_tools = {
            "video_analyzer": {
                "description": "Analyze video content and extract information",
                "params": {"url": str, "action": str},
                "response": "Video analyzer not yet implemented. Please use gaia_video_analyzer or video_analyzer_production instead."
            },
            "programming_language_identifier": {
                "description": "Identify the programming language of code",
                "params": {"code": str},
                "response": "Language identifier not yet implemented. Please use python_interpreter to analyze code."
            }
        }
        
        for tool_name, config in stub_tools.items():
            if tool_name not in self.tool_registry:
                # Create a stub tool
                from langchain_core.tools import Tool
                
                def make_stub_func(name, resp):
                    def stub_func(**kwargs):
                        logger.warning(f"Stub tool '{name}' called with args: {kwargs}")
                        return resp
                    return stub_func
                
                stub_tool = Tool(
                    name=tool_name,
                    description=config["description"],
                    func=make_stub_func(tool_name, config["response"])
                )
                
                self.tools.append(stub_tool)
                self.tool_registry[tool_name] = stub_tool
                logger.info(f"Added stub tool: {tool_name}")
    
    def _get_llm(self, task_type: str = "reasoning", temperature: float = None):
        """Get appropriate LLM based on task type and preference."""
        model_configs = {
            "reasoning": ModelConfig.REASONING_MODELS,
            "function_calling": ModelConfig.FUNCTION_CALLING_MODELS,
            "text_generation": ModelConfig.TEXT_GENERATION_MODELS,
        }
        
        models = model_configs.get(task_type, ModelConfig.REASONING_MODELS)
        
        # Select model based on preference
        if self.model_preference == "fast":
            model_name = models.get("fast", models["primary"])
        elif self.model_preference == "quality":
            model_name = models.get("primary")
        else:  # balanced
            model_name = models.get("fast", models["primary"])
        
        # Set temperature if not provided
        if temperature is None:
            temperature = 0.1 if task_type == "reasoning" else 0.0
        
        max_tokens = 4096 if "70b" in model_name else 2048
            
        try:
            return ChatGroq(
                temperature=temperature,
                model_name=model_name,
                max_tokens=max_tokens,
                max_retries=1,
                request_timeout=90
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChatGroq with model {model_name}: {e}")
            # Fallback to a simpler model
            fallback_model = "llama-3.1-8b-instant"
            logger.warning(f"Falling back to {fallback_model}")
            return ChatGroq(
                temperature=0.1,
                model_name=fallback_model,
                max_tokens=2048,
                max_retries=1,
                request_timeout=60
            )
    
    def _build_fsm_graph(self):
        """Build the FSM graph with strict state transitions."""
        
        # --- Node Functions ---
        
        def planning_node(state: EnhancedAgentState) -> dict:
            """Strategic planning node that creates or updates the execution plan."""
            logger.info(f"--- FSM STATE: PLANNING (Step {state.get('step_count', 0)}) ---")
            
            try:
                # Check if this is a complex query that should be delegated to crew
                if self.use_crew and state.get("step_count", 0) == 0:
                    # Analyze complexity - delegate if query involves multiple steps or research
                    complexity_indicators = [
                        "analyze", "compare", "research", "find and explain",
                        "multiple", "various", "comprehensive", "detailed"
                    ]
                    query_lower = state["query"].lower()
                    
                    is_complex = any(indicator in query_lower for indicator in complexity_indicators)
                    
                    if is_complex:
                        logger.info("Complex query detected - delegating to CrewAI workflow")
                        try:
                            from src.crew_workflow import run_crew_workflow
                            crew_result = run_crew_workflow(state["query"], self.tool_registry)
                            
                            # Convert crew results to FSM format
                            crew_steps = crew_result.get("intermediate_steps", {})
                            tool_calls = []
                            
                            for step_id, step_data in crew_steps.items():
                                if isinstance(step_data, dict):
                                    tool_calls.append({
                                        "tool_name": "crew_agent",
                                        "tool_input": {"task": step_id},
                                        "output": step_data,
                                        "step": len(tool_calls) + 1,
                                        "timestamp": datetime.now().isoformat()
                                    })
                            
                            return {
                                "final_answer": crew_result.get("output", ""),
                                "tool_calls": tool_calls,
                                "current_fsm_state": FSMState.VERIFYING,  # Still verify crew results
                                "stagnation_counter": 0,
                                "confidence": 0.85  # Crew results have good baseline confidence
                            }
                        except ImportError:
                            logger.warning("CrewAI not available, falling back to standard planning")
                        except Exception as e:
                            logger.error(f"Crew workflow failed: {e}, falling back to standard planning")
                
                # Standard planning flow
                llm = self._get_llm(task_type="reasoning")
                
                # Hydrate prompt with previous results (Directive 2)
                prompt = self._hydrate_planning_prompt(state)
                
                # Make LLM call with rate limiting
                def make_planning_call():
                    response = llm.invoke(prompt)
                    return response
                
                rate_limiter.wait_if_needed()
                response = make_planning_call()
                
                # Parse the plan into structured steps
                plan_text = response.content
                structured_plan = self._parse_plan_into_steps(plan_text, state)
                
                # FIX 1: Validate and correct tool parameters before returning
                validated_plan = self._validate_and_correct_plan(structured_plan)
                
                # Determine next tool to execute
                next_tool = self._get_next_tool_from_plan(validated_plan, state)
                
                return {
                    "plan": plan_text,
                    "master_plan": validated_plan,
                    "current_fsm_state": FSMState.TOOL_EXECUTION if next_tool else FSMState.SYNTHESIZING,
                    "messages": [response],
                    "stagnation_counter": 0  # Reset on successful planning
                }
                
            except Exception as e:
                logger.error(f"Error in planning node: {e}")
                return {
                    "current_fsm_state": FSMState.FINAL_FAILURE,
                    "errors": [str(e)]
                }
        
        def tool_execution_node(state: EnhancedAgentState) -> dict:
            """Execute tools and persist outputs (Directive 2)."""
            logger.info(f"--- FSM STATE: TOOL_EXECUTION (Step {state.get('step_count', 0)}) ---")
            
            # FIX 5: Initialize tool_reliability at the beginning to avoid UnboundLocalError
            tool_reliability = state.get("tool_reliability", {})
            
            try:
                # Get the next tool to execute from the plan
                next_tool_info = self._get_next_tool_from_plan(
                    state.get("master_plan", []), 
                    state
                )
                
                if not next_tool_info:
                    # No more tools to execute
                    return {"current_fsm_state": FSMState.SYNTHESIZING}
                
                tool_name = next_tool_info["tool"]
                tool_input = next_tool_info["input"]
                step_number = next_tool_info["step"]
                
                # --- NEW: LOOP-DETECTION & STAGNATION GUARDRAIL ---
                # If we have already made this exact tool call with the same input, treat as stagnation and re-plan
                for previous_call in state.get("tool_calls", []):
                    if previous_call.get("tool_name") == tool_name and previous_call.get("tool_input") == tool_input:
                        logger.warning("Duplicate tool call detected – triggering re-planning to avoid infinite loop")
                        return {
                            "current_fsm_state": FSMState.PLANNING,
                            "stagnation_counter": state.get("stagnation_counter", 0) + 1,
                            "errors": ["Duplicate tool call detected"],
                            "step_count": state.get("step_count", 0) + 1
                        }
                
                # FIX 2: Apply parameter translation layer before execution
                translated_input = self._translate_tool_parameters(tool_name, tool_input)
                
                # Execute the tool
                tool_output = None
                execution_start = time.time()
                execution_success = False
                
                if tool_name in self.tool_registry:
                    logger.info(f"Executing tool: {tool_name} with input: {translated_input}")
                    tool = self.tool_registry[tool_name]
                    
                    # --- ADAPTIVE TOOL SELECTION ---
                    # Check if we should use an alternative tool based on reliability
                    tool_stats = tool_reliability.get(tool_name, {"successes": 0, "failures": 0})
                    
                    if tool_stats["failures"] > 2 and tool_stats["successes"] < tool_stats["failures"]:
                        # This tool has been failing, try to find an alternative
                        alternative_tool = self._find_alternative_tool(tool_name, state)
                        if alternative_tool:
                            logger.warning(f"Tool {tool_name} has high failure rate, switching to {alternative_tool}")
                            tool_name = alternative_tool
                            tool = self.tool_registry[tool_name]
                    
                    # Execute with error handling
                    try:
                        tool_output = tool.invoke(translated_input)
                        execution_success = True
                    except Exception as tool_error:
                        logger.error(f"Tool execution failed: {tool_error}")
                        execution_success = False
                        
                        # Update tool reliability
                        tool_stats = tool_reliability.get(tool_name, {"successes": 0, "failures": 0})
                        tool_stats["failures"] += 1
                        tool_stats["last_error"] = str(tool_error)
                        tool_stats["last_error_time"] = datetime.now().isoformat()
                        tool_reliability[tool_name] = tool_stats
                        
                        # Try alternative tool if available
                        alternative_tool = self._find_alternative_tool(tool_name, state)
                        if alternative_tool and alternative_tool != tool_name:
                            logger.info(f"Trying alternative tool: {alternative_tool}")
                            try:
                                alt_tool = self.tool_registry[alternative_tool]
                                tool_output = alt_tool.invoke(translated_input)
                                execution_success = True
                                tool_name = alternative_tool  # Update for record
                            except Exception as alt_error:
                                logger.error(f"Alternative tool also failed: {alt_error}")
                                # Return to planning with error
                                return {
                                    "current_fsm_state": FSMState.PLANNING,
                                    "stagnation_counter": state.get("stagnation_counter", 0) + 1,
                                    "errors": [f"Tool execution failed: {str(tool_error)}, Alternative also failed: {str(alt_error)}"],
                                    "step_count": state.get("step_count", 0) + 1,
                                    "tool_reliability": tool_reliability
                                }
                        else:
                            # No alternative, return to planning
                            return {
                                "current_fsm_state": FSMState.PLANNING,
                                "stagnation_counter": state.get("stagnation_counter", 0) + 1,
                                "errors": [f"Tool execution failed: {str(tool_error)}"],
                                "step_count": state.get("step_count", 0) + 1,
                                "tool_reliability": tool_reliability
                            }
                else:
                    # Tool not found - will use stub if available
                    tool_output = f"Error: Tool '{tool_name}' not found"
                    logger.error(tool_output)
                    execution_success = False
                    
                    # Return to planning
                    return {
                        "current_fsm_state": FSMState.PLANNING,
                        "stagnation_counter": state.get("stagnation_counter", 0) + 1,
                        "errors": [tool_output],
                        "step_count": state.get("step_count", 0) + 1,
                        "tool_reliability": tool_reliability
                    }
                
                # Update tool reliability for successful execution
                if execution_success:
                    tool_stats = tool_reliability.get(tool_name, {"successes": 0, "failures": 0})
                    tool_stats["successes"] += 1
                    tool_stats["last_success_time"] = datetime.now().isoformat()
                    tool_stats["avg_execution_time"] = (
                        (tool_stats.get("avg_execution_time", 0) * tool_stats["successes"] + 
                         (time.time() - execution_start)) / (tool_stats["successes"] + 1)
                    )
                    tool_reliability[tool_name] = tool_stats
                
                # --- CRITICAL STATE UPDATE (Directive 2) ---
                # Persist the tool output in step_outputs
                current_step_outputs = state.get("step_outputs", {})
                current_step_outputs[step_number] = tool_output
                
                # Create tool call record
                new_tool_call = {
                    "tool_name": tool_name,
                    "tool_input": translated_input,
                    "output": tool_output,
                    "step": step_number,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Determine next state
                # Check if we have more tools to execute
                remaining_tools = self._count_remaining_tools(state.get("master_plan", []), state)
                next_state = FSMState.TOOL_EXECUTION if remaining_tools > 0 else FSMState.SYNTHESIZING
                
                return {
                    "step_outputs": current_step_outputs,
                    "tool_calls": [new_tool_call],
                    "current_fsm_state": next_state,
                    "stagnation_counter": 0,  # Reset on successful execution
                    "step_count": state.get("step_count", 0) + 1,
                    "tool_reliability": tool_reliability  # Include updated reliability
                }
                
            except Exception as e:
                logger.error(f"Error in tool execution node: {e}")
                return {
                    "current_fsm_state": FSMState.TOOL_EXECUTION_FAILURE,
                    "errors": [str(e)],
                    "tool_reliability": tool_reliability  # Always include tool_reliability
                }
        
        def synthesizing_node(state: EnhancedAgentState) -> dict:
            """Synthesize final answer with structured output (Directive 3)."""
            logger.info(f"--- FSM STATE: SYNTHESIZING ---")
            
            # FIX 4: Guard-rail - check if we have any successful tool outputs
            step_outputs = state.get("step_outputs", {})
            tool_calls = state.get("tool_calls", [])
            
            if not step_outputs and not tool_calls:
                logger.warning("No tool outputs available for synthesis - returning to planning")
                return {
                    "current_fsm_state": FSMState.PLANNING,
                    "stagnation_counter": state.get("stagnation_counter", 0) + 1,
                    "errors": ["No tool outputs available for synthesis"],
                    "step_count": state.get("step_count", 0) + 1
                }
            
            try:
                # Determine answer type from query
                answer_schema = self._determine_answer_schema(state["query"])
                
                # Get synthesis LLM with structured output
                llm = self._get_llm(task_type="text_generation", temperature=0.0)
                
                # Bind structured output schema
                if answer_schema:
                    structured_llm = llm.with_structured_output(answer_schema)
                else:
                    structured_llm = llm
                
                # Create synthesis prompt with all evidence
                synthesis_prompt = self._create_synthesis_prompt(state)
                
                # Generate structured answer
                rate_limiter.wait_if_needed()
                
                if answer_schema:
                    pydantic_output = structured_llm.invoke(synthesis_prompt)
                    # Extract the answer field from the Pydantic model
                    if hasattr(pydantic_output, 'answer'):
                        final_answer = str(pydantic_output.answer)
                    elif hasattr(pydantic_output, 'nominator_name'):
                        final_answer = pydantic_output.nominator_name
                    else:
                        final_answer = str(pydantic_output)
                else:
                    # Fallback for complex answers
                    response = structured_llm.invoke(synthesis_prompt)
                    final_answer = response.content.strip()
                
                return {
                    "final_answer": final_answer,
                    "current_fsm_state": FSMState.VERIFYING,
                    "confidence": 0.8  # Initial confidence before verification
                }
                
            except Exception as e:
                logger.error(f"Error in synthesizing node: {e}")
                return {
                    "current_fsm_state": FSMState.FINAL_FAILURE,
                    "errors": [str(e)]
                }
        
        def verifying_node(state: EnhancedAgentState) -> dict:
            """Verify the final answer for accuracy and formatting."""
            logger.info(f"--- FSM STATE: VERIFYING ---")
            
            # FIX 4: Guard-rail - check if we have a final answer to verify
            if not state.get("final_answer"):
                logger.warning("No final answer to verify - returning to planning")
                return {
                    "current_fsm_state": FSMState.PLANNING,
                    "stagnation_counter": state.get("stagnation_counter", 0) + 1,
                    "errors": ["No final answer to verify"],
                    "step_count": state.get("step_count", 0) + 1
                }
            
            try:
                # Get verification LLM
                llm = self._get_llm(task_type="reasoning")
                structured_llm = llm.with_structured_output(VerificationResult)
                
                # Create verification prompt
                verification_prompt = self._create_verification_prompt(state)
                
                # Verify the answer
                rate_limiter.wait_if_needed()
                verification_result = structured_llm.invoke(verification_prompt)
                
                if verification_result.is_valid and verification_result.confidence >= 0.8:
                    # Answer is verified
                    return {
                        "current_fsm_state": FSMState.FINISHED,
                        "confidence": verification_result.confidence
                    }
                else:
                    # Need to retry or refine
                    logger.warning(f"Verification failed: {verification_result.issues}")
                    
                    # Check stagnation
                    stagnation_count = state.get("stagnation_counter", 0) + 1
                    if stagnation_count >= self.max_stagnation:
                        return {
                            "current_fsm_state": FSMState.FINAL_FAILURE,
                            "errors": ["Max verification attempts reached"]
                        }
                    
                    # Go back to planning with verification feedback
                    return {
                        "current_fsm_state": FSMState.PLANNING,
                        "stagnation_counter": stagnation_count,
                        "errors": verification_result.issues
                    }
                    
            except Exception as e:
                logger.error(f"Error in verifying node: {e}")
                return {
                    "current_fsm_state": FSMState.FINISHED  # Accept answer as-is
                }
        
        def error_node(state: EnhancedAgentState) -> dict:
            """Handle errors gracefully."""
            logger.error(f"--- FSM STATE: ERROR ---")
            errors = state.get("errors", ["Unknown error"])
            logger.error(f"Errors: {errors}")
            
            # Set a default error message as final answer
            return {
                "final_answer": f"I encountered an error while processing your request: {'; '.join(errors)}",
                "current_fsm_state": FSMState.FINISHED
            }
        
        # --- FSM Router ---
        def fsm_router(state: EnhancedAgentState) -> str:
            """
            Central FSM router that determines next state transitions.
            Implements stagnation detection and termination guarantees.
            """
            current_state = state.get("current_fsm_state", FSMState.PLANNING)
            stagnation_count = state.get("stagnation_counter", 0)
            step_count = state.get("step_count", 0)
            
            # Absolute termination conditions
            if step_count >= self.max_steps:
                logger.warning(f"Max steps ({self.max_steps}) reached, terminating")
                return "error_node"
            
            if stagnation_count >= self.max_stagnation:
                logger.warning(f"Stagnation detected ({stagnation_count}), terminating")
                return "error_node"
            
            # Route based on current state
            if current_state == FSMState.PLANNING:
                return "planning_node"
            elif current_state == FSMState.TOOL_EXECUTION:
                return "tool_execution_node"
            elif current_state == FSMState.SYNTHESIZING:
                return "synthesizing_node"
            elif current_state == FSMState.VERIFYING:
                return "verifying_node"
            elif current_state in [FSMState.FINAL_FAILURE, FSMState.TOOL_EXECUTION_FAILURE, 
                                  FSMState.PERMANENT_API_FAILURE, FSMState.INVALID_PLAN_FAILURE]:
                return "error_node"
            elif current_state == FSMState.FINISHED:
                return END
            else:
                # Unknown state, go to error
                return "error_node"
        
        # --- Build the Graph ---
        workflow = StateGraph(EnhancedAgentState)
        
        # Add nodes
        workflow.add_node("planning_node", planning_node)
        workflow.add_node("tool_execution_node", tool_execution_node)
        workflow.add_node("synthesizing_node", synthesizing_node)
        workflow.add_node("verifying_node", verifying_node)
        workflow.add_node("error_node", error_node)
        
        # Set entry point
        workflow.set_entry_point("planning_node")
        
        # Add conditional edges from each node to the router
        for node_name in ["planning_node", "tool_execution_node", 
                         "synthesizing_node", "verifying_node", "error_node"]:
            workflow.add_conditional_edges(
                node_name,
                fsm_router,
                {
                    "planning_node": "planning_node",
                    "tool_execution_node": "tool_execution_node",
                    "synthesizing_node": "synthesizing_node",
                    "verifying_node": "verifying_node",
                    "error_node": "error_node",
                    END: END
                }
            )
        
        # Compile the graph
        return workflow.compile()
    
    # --- Helper Methods ---
    
    def _hydrate_planning_prompt(self, state: EnhancedAgentState) -> str:
        """Create planning prompt with context from previous steps (Directive 2)."""
        base_prompt = f"""You are a strategic planning engine. Create a step-by-step plan to answer this query:

Query: {state['query']}

"""
        
        # Add previous results if available
        step_outputs = state.get('step_outputs', {})
        if step_outputs:
            base_prompt += "Previous Step Results:\n"
            for step_num, output in sorted(step_outputs.items()):
                base_prompt += f"Step {step_num}: {output[:200]}...\n"
            base_prompt += "\n"
        
        # Add any errors from verification
        errors = state.get('errors', [])
        if errors:
            base_prompt += f"Previous attempt had issues: {', '.join(errors)}\n\n"
        
        base_prompt += """Create a detailed plan with specific tool calls. Each tool has SPECIFIC parameter names.

Available tools and their EXACT parameter requirements:
- web_researcher: {{"query": "search query", "source": "wikipedia" or "search"}}
- semantic_search_tool: {{"query": "search query", "filename": "knowledge_base.csv", "top_k": 3}}
- python_interpreter: {{"code": "python code to execute"}}
- tavily_search: {{"query": "search query", "max_results": 3}}
- file_reader: {{"filename": "path/to/file.txt", "lines": -1}}
- advanced_file_reader: {{"filename": "path/to/file"}}
- image_analyzer: {{"filename": "path/to/image.jpg", "task": "describe"}}
- image_analyzer_enhanced: {{"filename": "path/to/image.jpg", "task": "describe" or "chess"}}
- video_analyzer: {{"url": "video_url", "action": "download_info" or "transcribe"}}
- gaia_video_analyzer: {{"video_url": "googleusercontent_url"}}
- audio_transcriber: {{"filename": "path/to/audio.mp3"}}
- chess_logic_tool: {{"fen_string": "chess_position_in_FEN", "analysis_time_seconds": 2.0}}

Format your plan EXACTLY like this:
Step 1: [Description] - Tool: tool_name with parameters: {{"param1": "value1", "param2": value2}}
Step 2: [Description] - Tool: tool_name with parameters: {{"param": "value based on Step 1"}}

CRITICAL: Use the EXACT parameter names shown above. Do NOT use generic "query" for tools that require specific parameters."""
        
        return base_prompt
    
    def _validate_and_correct_plan(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """FIX 1: Validate tool parameters and correct common mistakes."""
        validated_plan = []
        
        for step in plan:
            if not step.get("tool"):
                validated_plan.append(step)
                continue
                
            tool_name = step["tool"]
            tool_input = step.get("input", {})
            
            # Skip if tool not in registry
            if tool_name not in self.tool_registry:
                logger.warning(f"Tool {tool_name} not found in registry, will use stub")
                validated_plan.append(step)
                continue
            
            tool = self.tool_registry[tool_name]
            
            # Try to get the tool's input schema
            try:
                if hasattr(tool, 'args_schema'):
                    # Validate against Pydantic schema
                    try:
                        validated_input = tool.args_schema.model_validate(tool_input)
                        step["input"] = validated_input.model_dump()
                    except ValidationError as e:
                        logger.warning(f"Validation error for {tool_name}: {e}")
                        # Try to fix common issues
                        corrected_input = self._correct_tool_input(tool_name, tool_input, e)
                        if corrected_input:
                            step["input"] = corrected_input
                        else:
                            # Mark step as invalid
                            step["error"] = f"Invalid parameters: {str(e)}"
                else:
                    # No schema available, apply heuristic corrections
                    step["input"] = self._correct_tool_input_heuristic(tool_name, tool_input)
            except Exception as e:
                logger.error(f"Error validating tool {tool_name}: {e}")
                # Use input as-is
            
            validated_plan.append(step)
        
        return validated_plan
    
    def _correct_tool_input(self, tool_name: str, tool_input: Dict[str, Any], validation_error: ValidationError) -> Optional[Dict[str, Any]]:
        """Try to correct tool input based on validation errors."""
        corrected = tool_input.copy()
        
        # Extract missing fields from validation error
        for error in validation_error.errors():
            if error["type"] == "missing":
                field_name = error["loc"][0]
                
                # Common corrections for missing fields
                if field_name == "filename" and "query" in tool_input:
                    # User provided 'query' instead of 'filename'
                    corrected["filename"] = tool_input["query"]
                    del corrected["query"]
                elif field_name == "code" and "query" in tool_input:
                    # User provided 'query' instead of 'code'
                    corrected["code"] = tool_input["query"]
                    del corrected["query"]
                elif field_name == "video_url" and "url" in tool_input:
                    # User provided 'url' instead of 'video_url'
                    corrected["video_url"] = tool_input["url"]
                    del corrected["url"]
                elif field_name == "fen_string" and "query" in tool_input:
                    # User provided 'query' instead of 'fen_string'
                    corrected["fen_string"] = tool_input["query"]
                    del corrected["query"]
        
        # Try validation again with corrected input
        try:
            tool = self.tool_registry[tool_name]
            if hasattr(tool, 'args_schema'):
                validated = tool.args_schema.model_validate(corrected)
                return validated.model_dump()
        except:
            pass
        
        return None
    
    def _correct_tool_input_heuristic(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Apply heuristic corrections when no schema is available."""
        # Define expected parameters for each tool
        tool_params = {
            "file_reader": ["filename", "lines"],
            "advanced_file_reader": ["filename"],
            "audio_transcriber": ["filename"],
            "image_analyzer": ["filename", "task"],
            "image_analyzer_enhanced": ["filename", "task"],
            "video_analyzer": ["url", "action"],
            "gaia_video_analyzer": ["video_url"],
            "chess_logic_tool": ["fen_string", "analysis_time_seconds"],
            "python_interpreter": ["code"],
            "web_researcher": ["query", "source", "date_range", "search_type"],
            "semantic_search_tool": ["query", "filename", "top_k"],
            "tavily_search": ["query", "max_results"]
        }
        
        expected_params = tool_params.get(tool_name, ["query"])
        corrected = {}
        
        # Map common mismatches
        if "query" in tool_input and "query" not in expected_params:
            # Figure out where to map 'query'
            if "filename" in expected_params:
                corrected["filename"] = tool_input["query"]
            elif "code" in expected_params:
                corrected["code"] = tool_input["query"]
            elif "fen_string" in expected_params:
                corrected["fen_string"] = tool_input["query"]
            elif "url" in expected_params:
                corrected["url"] = tool_input["query"]
            elif "video_url" in expected_params:
                corrected["video_url"] = tool_input["query"]
        else:
            # Keep query if it's expected
            if "query" in tool_input and "query" in expected_params:
                corrected["query"] = tool_input["query"]
        
        # Copy over other valid parameters
        for param in expected_params:
            if param in tool_input and param not in corrected:
                corrected[param] = tool_input[param]
        
        # Add default values for missing required params
        if tool_name == "file_reader" and "lines" not in corrected:
            corrected["lines"] = -1
        elif tool_name == "semantic_search_tool":
            if "filename" not in corrected:
                corrected["filename"] = "knowledge_base.csv"
            if "top_k" not in corrected:
                corrected["top_k"] = 3
        elif tool_name == "tavily_search" and "max_results" not in corrected:
            corrected["max_results"] = 3
        elif tool_name == "chess_logic_tool" and "analysis_time_seconds" not in corrected:
            corrected["analysis_time_seconds"] = 2.0
        elif tool_name in ["image_analyzer", "image_analyzer_enhanced"] and "task" not in corrected:
            corrected["task"] = "describe"
        elif tool_name == "video_analyzer" and "action" not in corrected:
            corrected["action"] = "download_info"
        
        return corrected
    
    def _translate_tool_parameters(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """FIX 2: Translate common parameter mismatches at execution time."""
        # This provides a second layer of defense in case planning validation missed something
        
        # Quick translations based on tool name
        translations = {
            "file_reader": {"query": "filename"},
            "advanced_file_reader": {"query": "filename"},
            "audio_transcriber": {"query": "filename"},
            "image_analyzer": {"query": "filename"},
            "image_analyzer_enhanced": {"query": "filename"},
            "python_interpreter": {"query": "code"},
            "video_analyzer": {"query": "url"},
            "gaia_video_analyzer": {"query": "video_url", "url": "video_url"},
            "chess_logic_tool": {"query": "fen_string"}
        }
        
        if tool_name not in translations:
            return tool_input
        
        translated = tool_input.copy()
        
        for old_key, new_key in translations[tool_name].items():
            if old_key in translated and new_key not in translated:
                translated[new_key] = translated[old_key]
                del translated[old_key]
                logger.info(f"Translated parameter '{old_key}' to '{new_key}' for tool {tool_name}")
        
        return translated
    
    def _parse_plan_into_steps(self, plan_text: str, state: EnhancedAgentState) -> List[Dict[str, Any]]:
        """Parse LLM plan into structured steps."""
        steps = []
        # Updated pattern to capture tool parameters
        step_pattern = r"Step (\d+):\s*(.+?)(?:\n|$)"
        matches = re.findall(step_pattern, plan_text, re.MULTILINE | re.DOTALL)
        
        for step_num, step_desc in matches:
            # Extract tool name
            tool_match = re.search(r"Tool:\s*(\w+)", step_desc, re.IGNORECASE)
            tool_name = tool_match.group(1) if tool_match else None
            
            # Extract parameters - look for JSON-like structure
            params_match = re.search(r"parameters:\s*(\{[^}]+\})", step_desc, re.IGNORECASE)
            
            if params_match:
                try:
                    # Parse the JSON parameters
                    params_str = params_match.group(1)
                    # Handle single quotes by replacing them with double quotes
                    params_str = params_str.replace("'", '"')
                    tool_params = json.loads(params_str)
                except json.JSONDecodeError:
                    # Fallback to extracting parameters manually
                    logger.warning(f"Failed to parse JSON parameters for step {step_num}, using fallback parser")
                    tool_params = self._fallback_parse_params(step_desc, tool_name)
            else:
                # Try to extract parameters from the description
                tool_params = self._fallback_parse_params(step_desc, tool_name)
            
            # Check if this step has already been completed
            completed = int(step_num) in state.get("step_outputs", {})
            
            steps.append({
                "step": int(step_num),
                "description": step_desc,
                "tool": tool_name,
                "input": tool_params if tool_name else None,
                "completed": completed
            })
        
        return steps
    
    def _fallback_parse_params(self, step_desc: str, tool_name: str) -> Dict[str, Any]:
        """Fallback parameter extraction when JSON parsing fails."""
        # Default parameters for each tool
        tool_defaults = {
            "python_interpreter": {"code": ""},
            "file_reader": {"filename": "", "lines": -1},
            "advanced_file_reader": {"filename": ""},
            "image_analyzer": {"filename": "", "task": "describe"},
            "image_analyzer_enhanced": {"filename": "", "task": "describe"},
            "video_analyzer": {"url": "", "action": "download_info"},
            "gaia_video_analyzer": {"video_url": ""},
            "audio_transcriber": {"filename": ""},
            "chess_logic_tool": {"fen_string": "", "analysis_time_seconds": 2.0},
            "web_researcher": {"query": "", "source": "wikipedia"},
            "semantic_search_tool": {"query": "", "filename": "knowledge_base.csv", "top_k": 3},
            "tavily_search": {"query": "", "max_results": 3}
        }
        
        # Get default parameters for this tool
        params = tool_defaults.get(tool_name, {"query": ""})
        
        # Try to extract values from the description
        # Look for quoted strings after "input:" or "parameters:"
        input_match = re.search(r'(?:input|parameters):\s*["\']?([^"\']+)["\']?', step_desc, re.IGNORECASE)
        if input_match:
            input_value = input_match.group(1).strip()
            
            # Assign to the appropriate parameter based on tool
            if tool_name == "python_interpreter":
                params["code"] = input_value
            elif tool_name in ["file_reader", "advanced_file_reader", "image_analyzer", "image_analyzer_enhanced", "audio_transcriber"]:
                params["filename"] = input_value
            elif tool_name in ["video_analyzer"]:
                params["url"] = input_value
            elif tool_name == "gaia_video_analyzer":
                params["video_url"] = input_value
            elif tool_name == "chess_logic_tool":
                params["fen_string"] = input_value
            else:
                # For search tools, use "query"
                params["query"] = input_value
        
        return params
    
    def _get_next_tool_from_plan(self, plan: List[Dict], state: EnhancedAgentState) -> Optional[Dict]:
        """Get the next uncompleted tool from the plan."""
        for step in plan:
            if not step.get("completed") and step.get("tool"):
                return step
        return None
    
    def _count_remaining_tools(self, plan: List[Dict], state: EnhancedAgentState) -> int:
        """Count remaining tools to execute."""
        return sum(1 for step in plan if not step.get("completed") and step.get("tool"))
    
    def _determine_answer_schema(self, query: str) -> Optional[BaseModel]:
        """Determine the appropriate answer schema based on query type."""
        query_lower = query.lower()
        
        if any(indicator in query_lower for indicator in ["how many", "count", "number of"]):
            return FinalIntegerAnswer
        elif any(indicator in query_lower for indicator in ["who", "name", "person"]):
            return FinalNameAnswer
        else:
            return FinalStringAnswer
    
    def _create_synthesis_prompt(self, state: EnhancedAgentState) -> str:
        """Create synthesis prompt with all evidence."""
        prompt = f"""Based on the following evidence, provide the final answer to the user's query.

Query: {state['query']}

Evidence:
"""
        
        # Add all tool outputs
        for tool_call in state.get("tool_calls", []):
            prompt += f"\n{tool_call['tool_name']} output: {tool_call['output'][:500]}..."
        
        prompt += """

CRITICAL: Provide ONLY the direct answer requested. Do not include explanations, prefixes like "The answer is", or any formatting. Just the answer itself.

Examples:
- For "How many?": Just the number (e.g., "7")
- For "Who?": Just the name (e.g., "John Smith")
- For "What?": Just the thing itself"""
        
        return prompt
    
    def _create_verification_prompt(self, state: EnhancedAgentState) -> str:
        """Create verification prompt."""
        return f"""Verify the following answer for accuracy and correct formatting.

Query: {state['query']}
Proposed Answer: {state['final_answer']}

Evidence used:
{json.dumps(state.get('step_outputs', {}), indent=2)}

Check:
1. Does the answer directly address the query?
2. Is it supported by the evidence?
3. Is it in the correct format (just the answer, no explanations)?
4. Is it factually accurate based on the evidence?

Provide a verification result."""
    
    def _find_alternative_tool(self, failed_tool: str, state: EnhancedAgentState) -> Optional[str]:
        """Find an alternative tool based on the failed tool and query context."""
        # Define tool alternatives mapping
        tool_alternatives = {
            "web_researcher": ["tavily_search", "semantic_search_tool"],
            "tavily_search": ["web_researcher", "semantic_search_tool"],
            "gaia_video_analyzer": ["video_analyzer_production", "audio_transcriber"],
            "video_analyzer_production": ["gaia_video_analyzer", "audio_transcriber"],
            "chess_logic_tool": ["chess_analyzer_production", "python_interpreter"],
            "chess_analyzer_production": ["chess_logic_tool", "python_interpreter"],
            "file_reader": ["advanced_file_reader", "python_interpreter"],
            "image_analyzer_enhanced": ["image_analyzer", "python_interpreter"],
            "image_analyzer": ["image_analyzer_enhanced", "python_interpreter"],
            # Add alternatives for stub tools
            "video_analyzer": ["gaia_video_analyzer", "video_analyzer_production"],
            "programming_language_identifier": ["python_interpreter"]
        }
        
        # Get alternatives for the failed tool
        alternatives = tool_alternatives.get(failed_tool, [])
        
        # Filter alternatives that exist in the registry
        available_alternatives = [alt for alt in alternatives if alt in self.tool_registry]
        
        if not available_alternatives:
            return None
        
        # Check tool reliability to pick the best alternative
        tool_reliability = state.get("tool_reliability", {})
        best_alternative = None
        best_score = -1
        
        for alt in available_alternatives:
            stats = tool_reliability.get(alt, {"successes": 0, "failures": 0})
            # Calculate a simple reliability score
            total_calls = stats["successes"] + stats["failures"]
            if total_calls == 0:
                # Untested tool, give it a chance
                score = 0.5
            else:
                score = stats["successes"] / total_calls
            
            if score > best_score:
                best_score = score
                best_alternative = alt
        
        return best_alternative
    
    def run(self, inputs: dict):
        """Run the FSM agent with comprehensive resilience and correlation tracking."""
        correlation_id = str(uuid.uuid4())
        
        with correlation_context(correlation_id):
            try:
                query = inputs.get("input", inputs.get("query", ""))
                logger.info(f"Starting resilient FSM execution", extra={'query_length': len(query), 'query_preview': query[:100]})
                
                # Initialize state with full resilience context
                initial_state = {
                    "correlation_id": correlation_id,
                    "query": query,
                    "messages": [HumanMessage(content=query)],
                    "current_fsm_state": FSMState.PLANNING,
                    "stagnation_counter": 0,
                    "max_stagnation": self.max_stagnation,
                    "retry_count": 0,
                    "max_retries": 3,
                    "step_count": 0,
                    "tool_calls": [],
                    "step_outputs": {},
                    "errors": [],
                    "failure_history": [],
                    "circuit_breaker_status": "closed",
                    "last_api_error": None,
                    "start_time": time.time(),
                    "verification_level": "thorough",  # Default to thorough per directives
                    "confidence": 0.0,
                    "cross_validation_sources": [],
                    "tool_reliability": {},
                    "tool_preferences": {},
                    "validated_plan": None,
                    "plan": "",
                    "master_plan": [],
                    "final_answer": "",
                    "end_time": 0.0
                }
                
                # Log state transition
                logger.info("FSM STATE: INITIAL", extra={'next_state': FSMState.PLANNING})
                
                # Run the graph with resilience
                result = self.graph.invoke(initial_state)
                
                # Extract final answer with fallback
                final_answer = result.get("final_answer", "Unable to determine answer")
                confidence = result.get("confidence", 0.0)
                total_steps = result.get("step_count", 0)
                
                # Log successful completion
                execution_time = time.time() - initial_state["start_time"]
                logger.info("FSM execution completed successfully", 
                           extra={
                               'final_answer_length': len(final_answer),
                               'confidence': confidence,
                               'total_steps': total_steps,
                               'execution_time': execution_time
                           })
                
                return {
                    "output": final_answer,
                    "intermediate_steps": result.get("tool_calls", []),
                    "confidence": confidence,
                    "total_steps": total_steps,
                    "correlation_id": correlation_id,
                    "execution_time": execution_time
                }
                
            except Exception as e:
                logger.error("FSM agent execution failed", extra={'error': str(e)}, exc_info=True)
                
                # Return graceful error response
                return {
                    "output": f"I encountered an error while processing your request. Please try again or rephrase your question.",
                    "intermediate_steps": [],
                    "error": str(e),
                    "correlation_id": correlation_id,
                    "confidence": 0.0,
                    "total_steps": 0
                }
    
    def stream(self, inputs: dict):
        """Stream the FSM agent execution."""
        # For now, just run normally
        # Streaming can be implemented later if needed
        yield self.run(inputs) 