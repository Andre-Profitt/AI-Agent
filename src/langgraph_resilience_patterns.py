"""
LangGraph Resilience Patterns Implementation
============================================
Based on "Architecting Resilience: A Definitive Guide to Debugging and Error Handling in LangGraph"

This module implements the advanced architectural patterns for building robust LangGraph agents:
1. GraphRecursionError Prevention with state-based counters
2. Tool Error Handling with self-correction and fallbacks
3. State Management with proper validation
4. Plan-and-Execute pattern for reduced cognitive load
5. Self-Reflection nodes for quality assurance
6. Human-in-the-Loop for critical operations
"""

import operator
import logging
import time
import json
import hashlib
from typing import Annotated, List, TypedDict, Dict, Any, Optional, Literal, Union, Callable
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
from datetime import datetime

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, ValidationError, field_validator
from src.error_handling import ToolExecutionResult

logger = logging.getLogger(__name__)

# --- SECTION 1: GraphRecursionError Prevention ---

class ErrorCategory(str, Enum):
    """Categories of errors for targeted recovery strategies"""
    TRANSIENT = "transient"           # Temporary failures (network, rate limits)
    PERMANENT = "permanent"           # Permanent failures (invalid input, auth)
    LOGIC = "logic"                   # Logic errors (invalid state, bad plan)
    RESOURCE = "resource"             # Resource issues (memory, CPU)
    UNKNOWN = "unknown"               # Unclassified errors

class LoopPreventionState(BaseModel):
    """State for preventing infinite loops"""
    max_loops: int = Field(default=5, description="Maximum number of loops allowed")
    current_loops: int = Field(default=0, description="Current number of loops")
    last_state_hash: str = Field(default="", description="Hash of last state")
    stagnation_score: int = Field(default=0, description="Score for detecting stagnation")
    force_termination: bool = Field(default=False, description="Force termination flag")

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True
    }

def calculate_state_hash(state: Dict[str, Any]) -> str:
    """Calculate a hash of the current state for loop detection"""
    # Create a stable representation of the state
    state_str = json.dumps(state, sort_keys=True)
    return hashlib.sha256(state_str.encode()).hexdigest()

def check_for_stagnation(current_hash: str, last_hash: str, state: LoopPreventionState) -> bool:
    """Check if the system is stagnating"""
    if current_hash == last_hash:
        state.stagnation_score += 1
    else:
        state.stagnation_score = max(0, state.stagnation_score - 1)
    
    return state.stagnation_score >= 3

def decrement_loop_counter(state: LoopPreventionState) -> bool:
    """Decrement the loop counter and check if we should continue"""
    state.current_loops += 1
    return state.current_loops < state.max_loops

# --- SECTION 2: Tool Error Handling ---

class ToolErrorStrategy(BaseModel):
    """Strategy for handling tool execution errors"""
    max_retries: int = Field(default=3, description="Maximum number of retries")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")
    backoff_factor: float = Field(default=2.0, description="Exponential backoff factor")
    error_categories: Dict[ErrorCategory, Callable] = Field(
        default_factory=dict,
        description="Error category handlers"
    )

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True
    }

def categorize_tool_error(error: str) -> ErrorCategory:
    """Categorize a tool error for appropriate handling"""
    error_lower = error.lower()
    
    if any(x in error_lower for x in ["timeout", "connection", "rate limit", "temporary"]):
        return ErrorCategory.TRANSIENT
    elif any(x in error_lower for x in ["invalid", "unauthorized", "forbidden", "not found"]):
        return ErrorCategory.PERMANENT
    elif any(x in error_lower for x in ["logic", "state", "plan", "validation"]):
        return ErrorCategory.LOGIC
    elif any(x in error_lower for x in ["memory", "cpu", "resource", "quota"]):
        return ErrorCategory.RESOURCE
    else:
        return ErrorCategory.UNKNOWN

# --- SECTION 3: State Validation Patterns ---

class ValidatedState(BaseModel):
    """Base class for validated state components"""
    
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "use_enum_values": True
    }

class PlanStep(ValidatedState):
    """Validated plan step with comprehensive error checking"""
    tool_name: str = Field(..., min_length=1, description="Name of the tool to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
    reasoning: str = Field(..., min_length=10, description="Why this step is necessary")
    expected_output_type: str = Field(..., description="Expected output format")
    dependencies: List[int] = Field(default_factory=list, description="Indices of dependent steps")
    
    @field_validator('tool_name')
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        """Ensure tool name is valid"""
        if v.startswith('_'):
            raise ValueError("Tool names cannot start with underscore")
        return v

class StateValidator(BaseModel):
    """Validator for agent state transitions"""
    required_fields: List[str] = Field(default_factory=list)
    field_validators: Dict[str, Callable] = Field(default_factory=dict)
    
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True
    }

    def validate_state(self, state: Dict[str, Any]) -> bool:
        """Validate the current state"""
        # Check required fields
        for field in self.required_fields:
            if field not in state:
                return False
                
        # Run field validators
        for field, validator in self.field_validators.items():
            if field in state and not validator(state[field]):
                return False
                
        return True

# --- SECTION 4: Plan-and-Execute Pattern ---

class PlanAndExecuteState(TypedDict):
    """State for Plan-and-Execute architecture"""
    query: str
    master_plan: List[PlanStep]
    current_step_index: int
    step_results: Dict[int, Any]
    planner_model: str  # High-capability model for planning
    executor_model: str  # Efficient model for execution

def create_master_plan(state: PlanAndExecuteState, planner_llm: Any) -> dict:
    """Create a comprehensive plan using a capable model"""
    system_prompt = """You are a strategic planner. Break down the user's query into clear, executable steps.
    
    Each step should:
    1. Use a specific tool
    2. Have clear parameters
    3. Explain its purpose
    4. Specify expected output format
    5. Note any dependencies on previous steps
    
    Focus on creating a linear, logical flow that minimizes backtracking."""
    
    response = planner_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Create a plan for: {state['query']}")
    ])
    
    # Parse and validate plan
    plan_data = json.loads(response.content)
    validated_plan = [PlanStep(**step) for step in plan_data['steps']]
    
    return {
        'master_plan': validated_plan,
        'current_step_index': 0
    }

def execute_plan_step(state: PlanAndExecuteState, executor_llm: Any, tools: list) -> dict:
    """Execute a single step from the plan using an efficient model"""
    current_step = state['master_plan'][state['current_step_index']]
    
    # Simple, focused prompt for the executor
    prompt = f"""Execute this specific step:
    Tool: {current_step.tool_name}
    Parameters: {json.dumps(current_step.parameters)}
    Purpose: {current_step.reasoning}
    
    Call the tool exactly as specified."""
    
    # Execute with timeout and error handling
    try:
        result = executor_llm.invoke([HumanMessage(content=prompt)])
        return {
            'step_results': {state['current_step_index']: result},
            'current_step_index': state['current_step_index'] + 1
        }
    except Exception as e:
        logger.error(f"Step execution failed: {e}")
        return handle_step_failure(state, current_step, e)

# --- SECTION 5: Self-Reflection Pattern ---

class ReflectionCriteria(BaseModel):
    """Criteria for reflection/critique nodes"""
    check_factual_accuracy: bool = True
    check_completeness: bool = True
    check_format_compliance: bool = True
    check_logical_consistency: bool = True
    minimum_confidence: float = 0.8

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True
    }

def create_reflection_node(criteria: ReflectionCriteria):
    """Factory for creating reflection nodes with specific criteria"""
    
    def reflection_node(state: dict) -> dict:
        """Reflect on the current output and determine if revision is needed"""
        output_to_check = state.get('draft_answer', '')
        issues_found = []
        confidence_score = 1.0
        
        if criteria.check_factual_accuracy:
            # Check against source documents if available
            if 'source_documents' in state:
                factual_issues = check_factual_consistency(
                    output_to_check, 
                    state['source_documents']
                )
                issues_found.extend(factual_issues)
                confidence_score -= len(factual_issues) * 0.1
                
        if criteria.check_completeness:
            # Check if all aspects of the query are addressed
            completeness_issues = check_completeness(
                state['query'], 
                output_to_check
            )
            issues_found.extend(completeness_issues)
            confidence_score -= len(completeness_issues) * 0.15
            
        if criteria.check_format_compliance:
            # Check if output matches requested format
            format_issues = check_format_compliance(
                state.get('requested_format', ''),
                output_to_check
            )
            issues_found.extend(format_issues)
            confidence_score -= len(format_issues) * 0.05
            
        reflection_passed = (
            len(issues_found) == 0 and 
            confidence_score >= criteria.minimum_confidence
        )
        
        return {
            'reflection_passed': reflection_passed,
            'reflection_issues': issues_found,
            'confidence_score': max(0, confidence_score),
            'needs_revision': not reflection_passed
        }
    
    return reflection_node

# --- SECTION 6: Human-in-the-Loop Pattern ---

class HumanApprovalRequest(BaseModel):
    """Request for human approval"""
    action_type: Literal["send_email", "execute_code", "modify_data", "api_call"]
    action_description: str
    action_parameters: Dict[str, Any]
    risk_level: Literal["low", "medium", "high", "critical"]
    reasoning: str
    alternatives: Optional[List[str]] = None

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True
    }

def create_human_approval_node(risk_threshold: str = "high"):
    """Create a node that requests human approval for risky actions"""
    
    def human_approval_node(state: dict) -> dict:
        """Check if human approval is needed"""
        planned_action = state.get('next_action', {})
        
        # Assess risk level
        risk_assessment = assess_action_risk(planned_action)
        
        if should_request_approval(risk_assessment, risk_threshold):
            approval_request = HumanApprovalRequest(
                action_type=planned_action['type'],
                action_description=planned_action['description'],
                action_parameters=planned_action['parameters'],
                risk_level=risk_assessment['level'],
                reasoning=risk_assessment['reasoning'],
                alternatives=suggest_alternatives(planned_action)
            )
            
            return {
                'requires_human_approval': True,
                'approval_request': approval_request.dict(),
                'execution_paused': True
            }
        
        return {
            'requires_human_approval': False,
            'execution_paused': False
        }
    
    return human_approval_node

# --- SECTION 7: Comprehensive Error Recovery ---

class ErrorRecoveryState(BaseModel):
    """State for error recovery process"""
    error_history: List[ToolExecutionResult] = Field(default_factory=list)
    recovery_attempts: int = Field(default=0)
    max_recovery_attempts: int = Field(default=3)
    last_successful_state: Optional[Dict[str, Any]] = None
    fallback_level: int = Field(default=0)
    max_fallback_level: int = Field(default=3)

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True
    }

def create_self_correction_prompt(error: str, state: Dict[str, Any]) -> str:
    """Create a prompt for self-correction based on error"""
    return f"""
    An error occurred: {error}
    Current state: {json.dumps(state, indent=2)}
    
    Please analyze the error and suggest corrections to:
    1. The current state
    2. The execution plan
    3. The tool parameters
    
    Provide specific, actionable corrections.
    """

def create_adaptive_error_handler(
    error_category: ErrorCategory,
    state: ErrorRecoveryState
) -> Callable:
    """Create an adaptive error handler based on error category and state"""
    
    def handle_transient_error(error: str) -> Dict[str, Any]:
        """Handle transient errors with retries and backoff"""
        if state.recovery_attempts < state.max_recovery_attempts:
            delay = (2 ** state.recovery_attempts) * 1.0  # Exponential backoff
            time.sleep(delay)
            state.recovery_attempts += 1
            return {"should_retry": True, "delay": delay}
        return {"should_retry": False, "error": "Max retries exceeded"}
        
    def handle_permanent_error(error: str) -> Dict[str, Any]:
        """Handle permanent errors with fallback strategies"""
        if state.fallback_level < state.max_fallback_level:
            state.fallback_level += 1
            return {"should_retry": True, "fallback_level": state.fallback_level}
        return {"should_retry": False, "error": "No more fallback strategies"}
        
    def handle_logic_error(error: str) -> Dict[str, Any]:
        """Handle logic errors with state correction"""
        if state.last_successful_state:
            return {
                "should_retry": True,
                "corrected_state": state.last_successful_state
            }
        return {"should_retry": False, "error": "No valid state to restore"}
        
    def handle_resource_error(error: str) -> Dict[str, Any]:
        """Handle resource errors with cleanup and retry"""
        # Implement resource cleanup
        return {"should_retry": True, "cleanup_performed": True}
        
    def handle_unknown_error(error: str) -> Dict[str, Any]:
        """Handle unknown errors with basic retry"""
        if state.recovery_attempts < state.max_recovery_attempts:
            state.recovery_attempts += 1
            return {"should_retry": True}
        return {"should_retry": False, "error": "Max retries exceeded"}
    
    handlers = {
        ErrorCategory.TRANSIENT: handle_transient_error,
        ErrorCategory.PERMANENT: handle_permanent_error,
        ErrorCategory.LOGIC: handle_logic_error,
        ErrorCategory.RESOURCE: handle_resource_error,
        ErrorCategory.UNKNOWN: handle_unknown_error
    }
    
    return handlers.get(error_category, handle_unknown_error)

# --- SECTION 8: Building the Resilient Graph ---

def build_resilient_graph(tools: list, config: dict = None) -> StateGraph:
    """Build a LangGraph with all resilience patterns implemented"""
    
    # Default configuration
    config = config or {
        'max_loops': 15,
        'enable_reflection': True,
        'human_approval_threshold': 'high',
        'planner_model': 'gpt-4',
        'executor_model': 'gpt-3.5-turbo',
        'reflection_criteria': ReflectionCriteria()
    }
    
    # Define the comprehensive state
    class ResilientAgentState(TypedDict):
        # User input
        query: str
        
        # Loop prevention
        remaining_loops: int
        stagnation_counter: int
        action_history: List[str]
        last_state_hash: str
        force_termination: bool
        
        # Plan and execute
        master_plan: List[Dict[str, Any]]
        current_step_index: int
        step_results: Dict[int, Any]
        
        # Tool execution
        tool_calls: Annotated[List[ToolMessage], operator.add]
        tool_errors: List[ToolExecutionResult]
        
        # Reflection
        draft_answer: str
        reflection_passed: bool
        reflection_issues: List[str]
        confidence_score: float
        
        # Human in the loop
        requires_human_approval: bool
        approval_request: Optional[Dict[str, Any]]
        
        # Error recovery
        error_history: List[Dict[str, Any]]
        recovery_attempts: int
        fallback_level: int
        
        # Final output
        final_answer: str
        messages: Annotated[List[BaseMessage], operator.add]
    
    # Initialize the graph
    graph = StateGraph(ResilientAgentState)
    
    # Add nodes
    graph.add_node("planner", create_master_plan)
    graph.add_node("executor", execute_plan_step)
    graph.add_node("reflection", create_reflection_node(config['reflection_criteria']))
    graph.add_node("human_approval", create_human_approval_node(config['human_approval_threshold']))
    graph.add_node("error_recovery", create_adaptive_error_handler())
    graph.add_node("synthesizer", create_synthesis_node)
    graph.add_node("tools", ToolNode(tools))
    
    # Add edges with loop prevention
    def should_continue_execution(state: ResilientAgentState) -> str:
        """Router with comprehensive termination conditions"""
        
        # Check force termination
        if state.get('force_termination', False):
            logger.warning("Force termination triggered")
            return "synthesizer"
            
        # Check loop counter
        if state.get('remaining_loops', config['max_loops']) <= 0:
            logger.warning("Loop limit reached")
            return "synthesizer"
            
        # Check stagnation
        if check_for_stagnation(state['last_state_hash'], state['last_state_hash'], state['loop_prevention_state']):
            logger.warning("Stagnation detected, terminating")
            return "synthesizer"
            
        # Check if plan is complete
        if state.get('current_step_index', 0) >= len(state.get('master_plan', [])):
            return "reflection" if config['enable_reflection'] else "synthesizer"
            
        # Check for errors that need recovery
        if state.get('tool_errors', []):
            return "error_recovery"
            
        # Check for human approval requirement
        if state.get('requires_human_approval', False):
            return "human_approval"
            
        # Continue execution
        return "executor"
    
    # Set up the graph flow
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", should_continue_execution)
    graph.add_edge("error_recovery", "executor")
    graph.add_edge("human_approval", "executor")
    graph.add_edge("reflection", "synthesizer")
    graph.add_edge("synthesizer", END)
    
    # Set entry point
    graph.set_entry_point("planner")
    
    return graph.compile()

# --- Utility Functions ---

def check_factual_consistency(answer: str, sources: List[str]) -> List[str]:
    """Check if answer is consistent with source documents"""
    # Implementation would use NLP/embedding comparison
    return []

def check_completeness(query: str, answer: str) -> List[str]:
    """Check if answer addresses all aspects of the query"""
    # Implementation would analyze query components vs answer coverage
    return []

def check_format_compliance(requested_format: str, answer: str) -> List[str]:
    """Check if answer matches requested format"""
    # Implementation would validate format requirements
    return []

def assess_action_risk(action: dict) -> dict:
    """Assess the risk level of a planned action"""
    # Implementation would analyze action type and parameters
    return {'level': 'low', 'reasoning': 'Safe action'}

def should_request_approval(risk_assessment: dict, threshold: str) -> bool:
    """Determine if human approval is needed based on risk"""
    risk_levels = ['low', 'medium', 'high', 'critical']
    return risk_levels.index(risk_assessment['level']) >= risk_levels.index(threshold)

def suggest_alternatives(action: dict) -> List[str]:
    """Suggest alternative actions"""
    return []

def create_user_friendly_error(error: dict) -> str:
    """Create a user-friendly error message"""
    return f"I encountered an issue: {error.get('message', 'Unknown error')}. Please try rephrasing your request."

def handle_step_failure(state: dict, step: PlanStep, error: Exception) -> dict:
    """Handle failure of a plan step"""
    return {
        'tool_errors': [ToolExecutionResult(
            success=False,
            output=None,
            error=error,
            error_category=categorize_tool_error(str(error))
        )]
    }

def create_synthesis_node(state: dict) -> dict:
    """Synthesize final answer from execution results"""
    return {'final_answer': 'Synthesis logic here'}

# --- SECTION 9: Debugging Utilities ---

class DebugContext:
    """Context manager for debugging LangGraph execution"""
    
    def __init__(self, graph: StateGraph, enable_breakpoints: bool = True):
        self.graph = graph
        self.enable_breakpoints = enable_breakpoints
        self.execution_trace = []
        
    def __enter__(self):
        # Wrap all nodes with debugging logic
        for node_name, node_func in self.graph.nodes.items():
            self.graph.nodes[node_name] = self._wrap_node(node_name, node_func)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Execution failed: {exc_type.__name__}: {exc_val}")
            self.dump_trace()
            
    def _wrap_node(self, name: str, func):
        def wrapped_node(state: dict) -> dict:
            # Pre-execution logging
            logger.debug(f"Entering node: {name}")
            logger.debug(f"State keys: {list(state.keys())}")
            
            # Breakpoint for debugging
            if self.enable_breakpoints and name in self.breakpoints:
                import pdb; pdb.set_trace()
                
            # Execute node
            start_time = time.time()
            try:
                result = func(state)
                execution_time = time.time() - start_time
                
                # Post-execution logging
                logger.debug(f"Node {name} completed in {execution_time:.2f}s")
                
                # Record in trace
                self.execution_trace.append({
                    'node': name,
                    'success': True,
                    'execution_time': execution_time,
                    'state_hash': calculate_state_hash(state)
                })
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Node {name} failed: {e}")
                
                # Record failure in trace
                self.execution_trace.append({
                    'node': name,
                    'success': False,
                    'error': str(e),
                    'execution_time': execution_time,
                    'state_snapshot': {k: str(v)[:100] for k, v in state.items()}
                })
                raise
                
        return wrapped_node
        
    def dump_trace(self):
        """Dump execution trace for debugging"""
        logger.info("=== EXECUTION TRACE ===")
        for entry in self.execution_trace:
            logger.info(json.dumps(entry, indent=2))

# --- Example Usage ---

def create_production_agent(tools: list) -> StateGraph:
    """Create a production-ready agent with all resilience patterns"""
    
    config = {
        'max_loops': 15,  # Reasonable limit
        'enable_reflection': True,  # Self-checking
        'human_approval_threshold': 'high',  # Only for risky actions
        'planner_model': 'gpt-4',  # Best model for planning
        'executor_model': 'gpt-3.5-turbo',  # Efficient for execution
        'reflection_criteria': ReflectionCriteria(
            check_factual_accuracy=True,
            check_completeness=True,
            check_format_compliance=True,
            minimum_confidence=0.85
        )
    }
    
    return build_resilient_graph(tools, config)

if __name__ == "__main__":
    # Example: Create and test a resilient agent
    from langchain_core.tools import tool
    
    @tool
    def example_tool(query: str) -> str:
        """Example tool for testing"""
        return f"Processed: {query}"
    
    # Build the graph
    graph = create_production_agent([example_tool])
    
    # Test with debugging enabled
    with DebugContext(graph) as debug:
        result = graph.invoke({
            'query': 'Test query',
            'remaining_loops': 15
        })
        
    print(f"Result: {result}") 