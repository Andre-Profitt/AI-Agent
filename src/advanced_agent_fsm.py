import operator
import logging
import time
import random
import json
from typing import Annotated, List, TypedDict, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger(__name__)

# --- FSM States Definition ---
class FSMState(str, Enum):
    """Enumeration of all possible FSM states"""
    PLANNING = "PLANNING"
    TOOL_EXECUTION = "TOOL_EXECUTION"
    SYNTHESIZING = "SYNTHESIZING"
    VERIFYING = "VERIFYING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"

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
    Enhanced agent state with FSM control fields and proactive state-passing
    """
    # User input
    query: str
    
    # Planning and execution
    plan: str
    master_plan: List[Dict[str, Any]]  # List of planned steps
    
    # Tool execution history with state-passing
    tool_calls: Annotated[List[Dict[str, Any]], operator.add]
    step_outputs: Dict[int, Any]  # Key: step number, Value: output
    
    # Final output
    final_answer: str
    
    # --- FSM Control Fields ---
    current_fsm_state: str
    stagnation_counter: int
    max_stagnation: int
    
    # Verification and confidence
    verification_level: str  # "basic", "thorough", "exhaustive"
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
    tool_reliability: Dict[str, Dict[str, Any]]  # Tool performance metrics
    tool_preferences: Dict[str, List[str]]  # Preferred tools for query types

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
    """Enhanced rate limiter with burst handling."""
    
    def __init__(self, max_requests_per_minute=60, burst_allowance=10):
        self.max_requests = max_requests_per_minute
        self.burst_allowance = burst_allowance
        self.requests = []
        self.burst_used = 0
        
    def wait_if_needed(self):
        """Advanced rate limiting with burst capacity."""
        now = time.time()
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        if len(self.requests) >= self.max_requests:
            if self.burst_used < self.burst_allowance:
                self.burst_used += 1
                logger.info(f"Using burst capacity ({self.burst_used}/{self.burst_allowance})")
            else:
                sleep_time = 60 - (now - self.requests[0]) + 1
                if sleep_time > 0:
                    logger.warning(f"Rate limit hit, sleeping for {sleep_time:.1f}s")
                    time.sleep(sleep_time)
        
        # Reset burst counter periodically
        if len(self.requests) < self.max_requests * 0.5:
            self.burst_used = max(0, self.burst_used - 1)
        
        self.requests.append(now)

rate_limiter = RateLimiter(max_requests_per_minute=60, burst_allowance=15)

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
        
        # FSM Configuration
        self.max_stagnation = 3  # Maximum allowed stagnation before error
        self.max_steps = 20  # Maximum total steps to prevent runaway execution
        
        try:
            logger.info(f"Initializing FSMReActAgent with {len(tools)} tools")
            self.graph = self._build_fsm_graph()
            logger.info("FSMReActAgent graph built successfully")
        except Exception as e:
            logger.error(f"Failed to build FSM graph: {e}", exc_info=True)
            raise RuntimeError(f"FSMReActAgent initialization failed: {e}")
    
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
                
                # Determine next tool to execute
                next_tool = self._get_next_tool_from_plan(structured_plan, state)
                
                return {
                    "plan": plan_text,
                    "master_plan": structured_plan,
                    "current_fsm_state": FSMState.TOOL_EXECUTION if next_tool else FSMState.SYNTHESIZING,
                    "messages": [response],
                    "stagnation_counter": 0  # Reset on successful planning
                }
                
            except Exception as e:
                logger.error(f"Error in planning node: {e}")
                return {
                    "current_fsm_state": FSMState.ERROR,
                    "errors": [str(e)]
                }
        
        def tool_execution_node(state: EnhancedAgentState) -> dict:
            """Execute tools and persist outputs (Directive 2)."""
            logger.info(f"--- FSM STATE: TOOL_EXECUTION (Step {state.get('step_count', 0)}) ---")
            
            # Initialize tool_reliability at the beginning to avoid UnboundLocalError
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
                
                # Execute the tool
                tool_output = None
                execution_start = time.time()
                execution_success = False
                
                if tool_name in self.tool_registry:
                    logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
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
                        tool_output = tool.invoke(tool_input)
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
                                tool_output = alt_tool.invoke(tool_input)
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
                    tool_output = f"Error: Tool '{tool_name}' not found"
                    logger.error(tool_output)
                
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
                    "tool_input": tool_input,
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
                    "current_fsm_state": FSMState.ERROR,
                    "errors": [str(e)]
                }
        
        def synthesizing_node(state: EnhancedAgentState) -> dict:
            """Synthesize final answer with structured output (Directive 3)."""
            logger.info(f"--- FSM STATE: SYNTHESIZING ---")
            
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
                    "current_fsm_state": FSMState.ERROR,
                    "errors": [str(e)]
                }
        
        def verifying_node(state: EnhancedAgentState) -> dict:
            """Verify the final answer for accuracy and formatting."""
            logger.info(f"--- FSM STATE: VERIFYING ---")
            
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
                            "current_fsm_state": FSMState.ERROR,
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
            elif current_state == FSMState.ERROR:
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
    
    def _parse_plan_into_steps(self, plan_text: str, state: EnhancedAgentState) -> List[Dict[str, Any]]:
        """Parse LLM plan into structured steps."""
        import re
        import json
        
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
            "image_analyzer": ["image_analyzer_enhanced", "python_interpreter"]
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
        """Run the FSM agent with the given inputs."""
        try:
            # Initialize state
            initial_state = {
                "query": inputs.get("input", inputs.get("query", "")),
                "messages": [HumanMessage(content=inputs.get("input", ""))],
                "current_fsm_state": FSMState.PLANNING,
                "stagnation_counter": 0,
                "max_stagnation": self.max_stagnation,
                "step_count": 0,
                "tool_calls": [],
                "step_outputs": {},
                "errors": [],
                "start_time": time.time(),
                "verification_level": "thorough",  # Default to thorough per directives
                "confidence": 0.0,
                "tool_reliability": {},
                "tool_preferences": {}
            }
            
            # Run the graph
            logger.info(f"Starting FSM execution for query: {initial_state['query'][:100]}...")
            result = self.graph.invoke(initial_state, config={"recursion_limit": 50})
            
            # Extract final answer
            final_answer = result.get("final_answer", "Unable to determine answer")
            
            logger.info(f"FSM execution completed. Answer: {final_answer}")
            
            return {
                "output": final_answer,
                "intermediate_steps": result.get("tool_calls", []),
                "confidence": result.get("confidence", 0.0),
                "total_steps": result.get("step_count", 0)
            }
            
        except Exception as e:
            logger.error(f"Error in FSM agent execution: {e}", exc_info=True)
            return {
                "output": f"Error: {str(e)}",
                "intermediate_steps": [],
                "error": str(e)
            }
    
    def stream(self, inputs: dict):
        """Stream the FSM agent execution."""
        # For now, just run normally
        # Streaming can be implemented later if needed
        yield self.run(inputs) 