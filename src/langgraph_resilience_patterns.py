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
from typing import Annotated, List, TypedDict, Dict, Any, Optional, Literal, Union
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

# --- SECTION 1: GraphRecursionError Prevention ---

class LoopPreventionState(TypedDict):
    """State keys for preventing infinite loops"""
    remaining_loops: int  # State-based counter for guaranteed termination
    stagnation_counter: int  # Tracks lack of progress
    action_history: List[str]  # Hashes of recent actions for cycle detection
    last_state_hash: str  # Hash of previous state to detect stagnation

def calculate_state_hash(state: dict) -> str:
    """Calculate a hash of the current state for stagnation detection"""
    # Only hash relevant fields that indicate progress
    relevant_fields = {
        'tool_calls': state.get('tool_calls', []),
        'final_answer': state.get('final_answer', ''),
        'current_step': state.get('current_step', 0)
    }
    state_str = json.dumps(relevant_fields, sort_keys=True)
    return hashlib.md5(state_str.encode()).hexdigest()

def check_for_stagnation(state: dict) -> bool:
    """Check if the agent is making no progress (stagnation)"""
    current_hash = calculate_state_hash(state)
    last_hash = state.get('last_state_hash', '')
    
    if current_hash == last_hash:
        state['stagnation_counter'] = state.get('stagnation_counter', 0) + 1
        logger.warning(f"Stagnation detected. Counter: {state['stagnation_counter']}")
        return state['stagnation_counter'] >= 3  # Stagnant for 3 iterations
    else:
        state['stagnation_counter'] = 0
        state['last_state_hash'] = current_hash
        return False

def decrement_loop_counter(state: dict) -> dict:
    """Decrement the loop counter and check for termination"""
    remaining = state.get('remaining_loops', 15)
    remaining -= 1
    
    logger.info(f"Loop counter: {remaining} remaining")
    
    return {
        'remaining_loops': remaining,
        'force_termination': remaining <= 0
    }

# --- SECTION 2: Tool Error Handling ---

class ToolErrorStrategy(Enum):
    """Strategies for handling tool errors"""
    SIMPLE_RETRY = "simple_retry"
    SELF_CORRECTION = "self_correction"
    MODEL_FALLBACK = "model_fallback"

@dataclass
class ToolExecutionResult:
    """Result of tool execution with error context"""
    success: bool
    output: Any
    error: Optional[Exception] = None
    error_category: Optional[str] = None
    retry_suggestions: Optional[List[str]] = None

def categorize_tool_error(error: Exception) -> tuple[str, ToolErrorStrategy]:
    """Categorize tool errors and recommend handling strategy"""
    error_str = str(error).lower()
    
    if isinstance(error, ValidationError):
        return "validation_error", ToolErrorStrategy.SELF_CORRECTION
    elif "rate limit" in error_str or "429" in error_str:
        return "rate_limit", ToolErrorStrategy.SIMPLE_RETRY
    elif "timeout" in error_str:
        return "timeout", ToolErrorStrategy.SIMPLE_RETRY
    elif "not found" in error_str or "404" in error_str:
        return "not_found", ToolErrorStrategy.SELF_CORRECTION
    else:
        return "unknown", ToolErrorStrategy.MODEL_FALLBACK

def create_self_correction_prompt(tool_name: str, tool_input: dict, error: Exception) -> str:
    """Create a prompt for self-correcting tool usage"""
    return f"""Your previous tool call failed. Please correct it based on the error:

Tool: {tool_name}
Your Input: {json.dumps(tool_input, indent=2)}
Error: {str(error)}

Common fixes:
1. Check parameter names match exactly
2. Ensure required parameters are provided
3. Verify data types (string vs int vs list)
4. Check for typos in parameter values

Please provide the corrected tool call."""

# --- SECTION 3: State Validation Patterns ---

class ValidatedState(BaseModel):
    """Base class for validated state components"""
    
    class Config:
        validate_assignment = True  # Validate on every assignment
        use_enum_values = True

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

class StateValidator:
    """Utilities for state validation and debugging"""
    
    @staticmethod
    def trace_state_corruption(state: dict, error: ValidationError) -> dict:
        """Trace which node corrupted the state"""
        error_details = {
            'failed_field': None,
            'bad_value': None,
            'expected_type': None,
            'corrupting_node': None
        }
        
        # Extract error details from Pydantic
        for err in error.errors():
            error_details['failed_field'] = err['loc'][0] if err['loc'] else 'unknown'
            error_details['bad_value'] = err['input']
            error_details['expected_type'] = err['type']
            
        # Trace back through state history to find corrupting node
        if 'node_history' in state:
            # The last node is likely the culprit
            error_details['corrupting_node'] = state['node_history'][-1]
            
        return error_details

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

class ErrorRecoveryState(TypedDict):
    """State for error recovery flow"""
    error_history: List[Dict[str, Any]]
    recovery_attempts: int
    fallback_level: int  # 0=retry, 1=self-correct, 2=model upgrade, 3=human
    circuit_breaker_trips: Dict[str, int]

def create_adaptive_error_handler(max_attempts: int = 3):
    """Create an error handler that escalates through recovery strategies"""
    
    def error_recovery_node(state: ErrorRecoveryState) -> dict:
        """Attempt to recover from errors with escalating strategies"""
        latest_error = state['error_history'][-1] if state['error_history'] else None
        
        if not latest_error:
            return {'error_resolved': True}
            
        error_category, strategy = categorize_tool_error(
            Exception(latest_error['message'])
        )
        
        recovery_result = {
            'error_resolved': False,
            'recovery_strategy': None,
            'should_terminate': False
        }
        
        # Escalate through strategies based on failure count
        if state['recovery_attempts'] < max_attempts:
            if state['fallback_level'] == 0:
                # Level 0: Simple retry with backoff
                recovery_result['recovery_strategy'] = 'retry_with_backoff'
                recovery_result['wait_seconds'] = 2 ** state['recovery_attempts']
                
            elif state['fallback_level'] == 1:
                # Level 1: Self-correction with error context
                recovery_result['recovery_strategy'] = 'self_correction'
                recovery_result['correction_prompt'] = create_self_correction_prompt(
                    latest_error.get('tool_name', 'unknown'),
                    latest_error.get('tool_input', {}),
                    Exception(latest_error['message'])
                )
                
            elif state['fallback_level'] == 2:
                # Level 2: Upgrade to more capable model
                recovery_result['recovery_strategy'] = 'model_fallback'
                recovery_result['fallback_model'] = 'gpt-4'
                
            else:
                # Level 3: Request human intervention
                recovery_result['recovery_strategy'] = 'human_intervention'
                recovery_result['should_terminate'] = True
                
            recovery_result['fallback_level'] = state['fallback_level'] + 1
            recovery_result['recovery_attempts'] = state['recovery_attempts'] + 1
            
        else:
            # Max attempts reached
            recovery_result['should_terminate'] = True
            recovery_result['final_error'] = create_user_friendly_error(latest_error)
            
        return recovery_result
    
    return error_recovery_node

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
        if check_for_stagnation(state):
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
            error_category=categorize_tool_error(error)[0]
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