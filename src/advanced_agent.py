import operator
import logging
import time
import random
import json
from typing import Annotated, List, TypedDict, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
from dataclasses import dataclass

from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Configure logging
logger = logging.getLogger(__name__)

# --- Enhanced State Tracking ---

@dataclass
class PlanStep:
    """Represents a single step in the reasoning plan."""
    step_id: int
    description: str
    tool_needed: Optional[str]
    expected_outcome: str
    completed: bool = False
    success: bool = False
    attempts: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

@dataclass
class ReflectionNote:
    """Captures reflection insights during reasoning."""
    step: int
    confidence_before: float
    confidence_after: float
    insight: str
    decision: str  # "continue", "pivot", "verify", "conclude"
    timestamp: datetime

@dataclass
class ToolResult:
    """Enhanced tool result with validation metadata."""
    tool_name: str
    success: bool
    content: str
    confidence: float
    validation_checks: Dict[str, bool]
    cross_references: List[str]
    errors: List[str]

class EnhancedAgentState(TypedDict):
    """
    Advanced agent state with comprehensive planning and reflection capabilities.
    """
    messages: Annotated[List[AnyMessage], operator.add]
    run_id: UUID
    log_to_db: bool
    
    # Strategic Planning
    master_plan: List[PlanStep]
    current_step: int
    plan_revisions: int
    
    # Reflection & Learning
    reflections: List[ReflectionNote]
    confidence_history: List[float]
    error_recovery_attempts: int
    
    # Adaptive Intelligence
    step_count: int
    confidence: float
    reasoning_complete: bool
    verification_level: str  # "basic", "thorough", "exhaustive"
    
    # Tool Performance Tracking
    tool_success_rates: Dict[str, float]
    tool_results: List[ToolResult]
    cross_validation_sources: List[str]

# --- Multi-Model Configuration ---

class ModelConfig:
    """Configuration for different Groq models optimized for specific tasks."""
    
    # Reasoning models - for complex logical thinking
    REASONING_MODELS = {
        "primary": "llama-3.3-70b-versatile",  # Best for complex reasoning
        "fast": "llama-3.1-8b-instant",        # Fast reasoning
        "deep": "deepseek-r1-distill-llama-70b"  # Deep analytical reasoning
    }
    
    # Function calling models - for tool use
    FUNCTION_CALLING_MODELS = {
        "primary": "llama-3.3-70b-versatile",  # Best function calling
        "fast": "llama-3.1-8b-instant",        # Fast tool use
        "versatile": "llama3-groq-70b-8192-tool-use-preview"  # Specialized for tools
    }
    
    # Text generation models - for final answers
    TEXT_GENERATION_MODELS = {
        "primary": "llama-3.3-70b-versatile",  # High quality text
        "fast": "llama-3.1-8b-instant",        # Fast generation
        "creative": "gemma2-9b-it"             # Creative responses
    }
    
    # Vision models - for image analysis  
    VISION_MODELS = {
        "primary": "meta-llama/llama-4-scout-17b-16e-instruct",  # Latest vision capabilities
        "fast": "llama-3.1-8b-instant"                          # Fast fallback for simple tasks
    }
    
    # Grading/evaluation models - for answer validation
    GRADING_MODELS = {
        "primary": "gemma-7b-it",               # Good for evaluation
        "fast": "llama-3.1-8b-instant"          # Fast grading
    }

# --- Rate Limiting & Error Handling ---

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

def advanced_retry_with_recovery(func, max_retries=3, recovery_strategies=None):
    """Enhanced retry with intelligent recovery strategies."""
    if recovery_strategies is None:
        recovery_strategies = ["exponential_backoff", "alternative_approach", "simplified_query"]
    
    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed()
            return func()
        except Exception as e:
            error_str = str(e).lower()
            
            if "429" in error_str or "rate_limit" in error_str:
                strategy = "exponential_backoff"
                wait_time = (2 ** attempt) + random.uniform(0, 2)
                logger.warning(f"Rate limit: waiting {wait_time:.1f}s (attempt {attempt + 1})")
                time.sleep(wait_time)
                
            elif "context_length" in error_str:
                strategy = "simplified_query"
                logger.error(f"Context length exceeded: {e}")
                raise  # Can't recover from context length
                
            elif "timeout" in error_str:
                strategy = "alternative_approach"
                wait_time = 5 + attempt * 2
                logger.warning(f"Timeout: waiting {wait_time}s before retry")
                time.sleep(wait_time)
                
            else:
                logger.error(f"Unrecoverable error: {e}")
                raise
            
            if attempt == max_retries - 1:
                logger.error(f"Max retries reached with strategy {strategy}: {e}")
                raise
    
    return None

# --- World-Class ReAct Agent with Advanced Capabilities ---

class AdvancedReActAgent:
    """
    Next-generation ReAct agent with sophisticated planning, reflection,
    cross-validation, and adaptive reasoning capabilities.
    """
    
    def __init__(self, tools: list, log_handler: logging.Handler = None, model_preference: str = "balanced"):
        self.tools = tools
        self.log_handler = log_handler
        self.max_reasoning_steps = 15  # Reduced to prevent recursion issues
        self.tool_registry = {tool.name: tool for tool in tools}
        self.model_preference = model_preference  # "fast", "balanced", "quality"
        
        try:
            logger.info(f"Initializing AdvancedReActAgent with {len(tools)} tools")
            self.graph = self._build_advanced_graph()
            logger.info("AdvancedReActAgent graph built successfully")
        except Exception as e:
            logger.error(f"Failed to build agent graph: {e}", exc_info=True)
            raise RuntimeError(f"AdvancedReActAgent initialization failed during graph building: {e}")
    
    def _get_llm(self, task_type: str = "reasoning"):
        """Get appropriate LLM based on task type and preference."""
        model_configs = {
            "reasoning": ModelConfig.REASONING_MODELS,
            "function_calling": ModelConfig.FUNCTION_CALLING_MODELS,
            "text_generation": ModelConfig.TEXT_GENERATION_MODELS,
            "vision": ModelConfig.VISION_MODELS,
            "grading": ModelConfig.GRADING_MODELS
        }
        
        models = model_configs.get(task_type, ModelConfig.REASONING_MODELS)
        
        # Select model based on preference
        if self.model_preference == "fast":
            model_name = models.get("fast", models["primary"])
        elif self.model_preference == "quality":
            model_name = models.get("primary")
        else:  # balanced
            # Use fast for simple tasks, quality for complex
            model_name = models.get("fast", models["primary"])
        
        # Special handling for certain models
        temperature = 0.05
        max_tokens = 3072
        
        if "deepseek" in model_name:
            temperature = 0.1  # Slightly higher for deep reasoning
            max_tokens = 4096  # More tokens for complex analysis
        elif "gemma" in model_name:
            temperature = 0.0  # Very deterministic for grading
            max_tokens = 2048
        elif "70b" in model_name:
            max_tokens = 4096  # Larger models can handle more
            
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
            # Fallback to a simpler model if the preferred one fails
            try:
                fallback_model = "llama-3.1-8b-instant"
                logger.warning(f"Falling back to {fallback_model}")
                return ChatGroq(
                    temperature=0.1,
                    model_name=fallback_model,
                    max_tokens=2048,
                    max_retries=1,
                    request_timeout=60
                )
            except Exception as fallback_error:
                logger.critical(f"Failed to initialize fallback ChatGroq model: {fallback_error}")
                raise RuntimeError(f"Unable to initialize any ChatGroq model. Original error: {e}, Fallback error: {fallback_error}")
    
    def _get_planning_llm(self):
        """Get LLM optimized for strategic planning."""
        try:
            # Use the deep reasoning model for planning
            return ChatGroq(
                temperature=0.1,
                model_name=ModelConfig.REASONING_MODELS.get("deep", ModelConfig.REASONING_MODELS["primary"]),
                max_tokens=4096,
                max_retries=1,
                request_timeout=120
            )
        except Exception as e:
            logger.error(f"Failed to initialize planning LLM: {e}")
            # Fallback to basic reasoning model
            try:
                return ChatGroq(
                    temperature=0.1,
                    model_name=ModelConfig.REASONING_MODELS["primary"],
                    max_tokens=3072,
                    max_retries=1,
                    request_timeout=90
                )
            except Exception as fallback_error:
                logger.critical(f"Failed to initialize fallback planning LLM: {fallback_error}")
                raise RuntimeError(f"Unable to initialize planning LLM. Original error: {e}, Fallback error: {fallback_error}")
    
    def _get_advanced_system_prompt(self):
        """World-class system prompt with sophisticated reasoning guidance."""
        return """You are Orion-X, an advanced AI research assistant with world-class reasoning capabilities.

🎯 MISSION: Solve complex problems through strategic planning, systematic execution, and intelligent reflection.

🧠 ADVANCED REASONING FRAMEWORK:

**PHASE 1: STRATEGIC PLANNING**
Before any action, create a comprehensive plan:
- Break the problem into logical sub-goals
- Identify required information and tools
- Plan verification and cross-validation steps
- Estimate confidence levels for each step
- Consider alternative approaches

**PHASE 2: SYSTEMATIC EXECUTION**
Execute your plan methodically:
- Follow the planned sequence intelligently
- Use tools strategically and purposefully
- Validate results from multiple angles
- Handle errors with alternative strategies
- Build evidence systematically

**PHASE 3: INTELLIGENT REFLECTION**
At regular intervals, assess progress:
- Evaluate confidence in current findings
- Identify gaps or inconsistencies
- Consider if plan revision is needed
- Assess when sufficient evidence is gathered
- Determine optimal verification level

**PHASE 4: CROSS-VALIDATION**
For critical information:
- Verify through multiple independent sources
- Cross-reference findings for consistency
- Check for contradictory evidence
- Validate using different tool approaches
- Assess reliability of each source

**PHASE 5: ADAPTIVE DECISION MAKING**
Based on accumulated evidence:
- Adjust confidence levels intelligently
- Revise strategy if needed
- Determine when sufficient certainty is reached
- Choose appropriate verification depth
- Conclude with optimal timing

🛠️ ADVANCED TOOL ORCHESTRATION:
- **Primary Research**: web_researcher, semantic_search_tool
- **Document Analysis**: file_reader, advanced_file_reader
- **Multimedia Processing**: audio_transcriber, video_analyzer, image_analyzer
- **Computation**: python_interpreter
- **Real-time Search**: tavily_search
- **Cross-validation**: Use multiple tools for verification

🔍 REFLECTION TRIGGERS:
Reflect when:
- Confidence drops below 70%
- Contradictory information is found
- Tool errors occur multiple times
- Step count reaches multiples of 4
- Evidence seems insufficient

⚡ ADAPTIVE STRATEGIES:
- **High-confidence path**: Direct execution with basic validation
- **Medium-confidence path**: Enhanced verification with 2+ sources
- **Low-confidence path**: Exhaustive cross-validation and alternative approaches
- **Error recovery**: Switch tools, simplify queries, or change approach

🎯 ENHANCED ACCURACY REQUIREMENTS:
- For factual questions: Verify through 2+ authoritative sources - NEVER rely on single source
- For calculations: Cross-check with python_interpreter AND manual verification
- For multimedia: Use appropriate specialized tools AND verify content carefully
- For temporal data: Pay special attention to dates, timeframes, and ensure current information
- For complex reasoning: Break into verifiable sub-components AND validate each step
- For reverse text: Decode character by character to ensure accuracy
- For mathematical problems: Verify logic step-by-step before concluding
- For historical data: Cross-reference multiple sources for accuracy

🏆 EXCELLENCE PRINCIPLES:
- Plan before acting, reflect during execution
- Validate critical findings through multiple sources
- Handle errors gracefully with intelligent recovery
- Build confidence systematically through evidence
- Adapt strategy based on problem complexity
- Conclude only when appropriately confident

🎯 CRITICAL ANSWER FORMAT REQUIREMENT:
When providing your final answer:
- Give ONLY the direct answer (number, word, phrase, name, etc.)  
- DO NOT include "final answer", "the answer is", or any prefixes
- DO NOT use mathematical notation like $\\boxed{...}$ or LaTeX formatting
- DO NOT include explanations or reasoning in your final response
- Examples of correct format: "2", "PaleoNeon", "HON", "Vladimir", "Moscow"

🚨 STRICT AMBIGUITY HANDLING (DIRECTIVE 3):
- NEVER hallucinate or guess when information is missing
- If a tool fails or context is incomplete, state what is missing
- Default to "Unable to determine answer" rather than inventing an answer
- Remember: It's better to admit uncertainty than provide incorrect information

Remember: Think like a world-class researcher - plan strategically, execute systematically, validate thoroughly, but provide only the clean, direct answer when complete."""

    def _create_initial_plan(self, query: str) -> List[PlanStep]:
        """Create an intelligent initial plan based on query analysis."""
        # Analyze query type and complexity
        query_lower = query.lower()
        
        steps = []
        step_id = 1
        
        # Determine query type and create appropriate plan
        if any(indicator in query_lower for indicator in ["how many", "count", "number of"]):
            # Numerical/counting question
            steps.extend([
                PlanStep(step_id, "Research primary sources for counting data", "web_researcher", "Find authoritative sources"),
                PlanStep(step_id + 1, "Cross-validate with secondary sources", "semantic_search_tool", "Confirm numbers"),
                PlanStep(step_id + 2, "Verify calculations if needed", "python_interpreter", "Accurate count")
            ])
            
        elif any(indicator in query_lower for indicator in ["chess", "move", "position"]):
            # Chess analysis question
            steps.extend([
                PlanStep(step_id, "Analyze chess position", "image_analyzer", "Understand position"),
                PlanStep(step_id + 1, "Evaluate candidate moves", None, "Strategic analysis"),
                PlanStep(step_id + 2, "Cross-validate with chess principles", "web_researcher", "Confirm best move")
            ])
            
        elif any(indicator in query_lower for indicator in ["code", "country", "iso"]):
            # Country/code lookup question
            steps.extend([
                PlanStep(step_id, "Search official standards", "web_researcher", "Find ISO/official codes"),
                PlanStep(step_id + 1, "Verify through multiple sources", "semantic_search_tool", "Confirm accuracy"),
            ])
            
        elif any(indicator in query_lower for indicator in ["album", "song", "music", "artist"]):
            # Music/discography question - enhanced accuracy
            steps.extend([
                PlanStep(step_id, "Research artist discography", "web_researcher", "Find complete discography"),
                PlanStep(step_id + 1, "Cross-reference release dates", "semantic_search_tool", "Verify timeframes"),
                PlanStep(step_id + 2, "Verify with secondary sources", "tavily_search", "Double-check discography"),
                PlanStep(step_id + 3, "Count relevant releases", "python_interpreter", "Accurate count")
            ])
            
        elif any(indicator in query_lower for indicator in ["video", "watch", "youtube"]):
            # Video analysis question - enhanced verification
            steps.extend([
                PlanStep(step_id, "Analyze video content", "video_analyzer", "Extract video information"),
                PlanStep(step_id + 1, "Cross-reference video details", "web_researcher", "Verify video content"),
                PlanStep(step_id + 2, "Double-check specific details", "semantic_search_tool", "Confirm accuracy")
            ])
            
        elif any(indicator in query_lower for indicator in ["reverse", "backwards", "decode"]):
            # Text reversal/decoding question
            steps.extend([
                PlanStep(step_id, "Decode reversed text carefully", "python_interpreter", "Character-by-character reversal"),
                PlanStep(step_id + 1, "Verify decoding accuracy", None, "Manual verification"),
                PlanStep(step_id + 2, "Confirm final answer", None, "Double-check result")
            ])
            
        elif any(indicator in query_lower for indicator in ["olympics", "athletes", "competition"]):
            # Olympic/sports data question - enhanced verification
            steps.extend([
                PlanStep(step_id, "Research Olympic data", "web_researcher", "Find official Olympic records"),
                PlanStep(step_id + 1, "Cross-reference statistics", "semantic_search_tool", "Verify athlete counts"),
                PlanStep(step_id + 2, "Confirm with secondary sources", "tavily_search", "Double-check data"),
                PlanStep(step_id + 3, "Calculate final answer", "python_interpreter", "Accurate calculation")
            ])
            
        else:
            # General research question
            steps.extend([
                PlanStep(step_id, "Initial research", "web_researcher", "Gather basic information"),
                PlanStep(step_id + 1, "Deep analysis", "semantic_search_tool", "Find detailed data"),
                PlanStep(step_id + 2, "Cross-validation", "tavily_search", "Verify accuracy")
            ])
        
        return steps

    def _parse_enhanced_plan(self, plan_content: str, user_query: str) -> List[PlanStep]:
        """Parse the enhanced plan with explicit input/output specifications."""
        import re
        
        structured_plan = []
        
        # Try to parse the structured plan
        step_pattern = r"Step (\d+):(.*?)(?=Step \d+:|$)"
        matches = re.findall(step_pattern, plan_content, re.DOTALL)
        
        for step_num, step_content in matches:
            # Extract components
            desc_match = re.search(r"Step \d+:\s*(.+?)(?=\n|$)", plan_content)
            description = desc_match.group(1).strip() if desc_match else f"Step {step_num}"
            
            # Extract expected output
            output_match = re.search(r"Expected Output:\s*(.+?)(?=\n|$)", step_content)
            expected_output = output_match.group(1).strip() if output_match else "Result from this step"
            
            # Extract tool
            tool_match = re.search(r"Tool:\s*(.+?)(?=\n|$)", step_content)
            tool_text = tool_match.group(1).strip() if tool_match else ""
            
            # Map tool text to actual tool names
            tool_mapping = {
                "web_researcher": "web_researcher",
                "semantic_search": "semantic_search_tool",
                "tavily": "tavily_search",
                "python": "python_interpreter",
                "image": "image_analyzer",
                "video": "video_analyzer",
                "file": "file_reader",
                "audio": "audio_transcriber"
            }
            
            tool_needed = None
            for key, value in tool_mapping.items():
                if key in tool_text.lower():
                    tool_needed = value
                    break
            
            structured_plan.append(PlanStep(
                step_id=int(step_num),
                description=description,
                tool_needed=tool_needed,
                expected_outcome=expected_output,
                completed=False,
                success=False,
                attempts=0,
                errors=[]
            ))
        
        # If parsing failed or no steps found, use default plan
        if not structured_plan:
            logger.warning("Failed to parse enhanced plan, using default")
            structured_plan = self._create_initial_plan(user_query)
        
        return structured_plan

    def _assess_verification_level(self, state: EnhancedAgentState) -> str:
        """Determine verification level with bias toward thorough verification for accuracy."""
        confidence = state.get("confidence", 0.5)
        step_count = state.get("step_count", 0)
        error_count = state.get("error_recovery_attempts", 0)
        
        # Default to thorough verification for better accuracy
        if confidence > 0.85 and error_count == 0 and step_count >= 5:
            return "basic"  # Only use basic if very confident with multiple steps
        elif confidence > 0.6 and error_count < 2:
            return "thorough"  # Default level for most questions
        else:
            return "exhaustive"  # Use exhaustive for complex/problematic cases

    def _build_advanced_graph(self):
        """Build sophisticated graph with planning, execution, and reflection nodes."""
        # Get different LLMs for different tasks
        reasoning_llm = self._get_llm("reasoning")
        function_llm = self._get_llm("function_calling")
        planning_llm = self._get_planning_llm()
        
        # DIRECTIVE 1: Add synthesis LLM for precise answer extraction
        synthesis_llm = self._get_llm("text_generation")
        
        # DIRECTIVE 3: Add fact-checking LLM for verification
        fact_check_llm = self._get_llm("grading")
        
        # DIRECTIVE 4: Add logic review LLM
        logic_llm = self._get_llm("reasoning")
        
        # Bind tools to the function calling model
        model_with_tools = function_llm.bind_tools(self.tools)

        def strategic_planning_node(state: EnhancedAgentState):
            """DIRECTIVE 2: Enhanced planning with explicit outputs and debugging capability."""
            messages = state["messages"]
            
            # Extract user query for planning
            user_query = ""
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    user_query = msg.content
                    break
            
            # Check if we need to create or debug the plan
            if not state.get("master_plan") or state.get("plan_debugging_requested"):
                # Enhanced planning with explicit input/output specification
                planning_prompt = f"""
STRATEGIC PLANNING PHASE - ENHANCED WITH EXPLICIT OUTPUTS

Analyze this problem and create a comprehensive, step-by-step plan:
"{user_query}"

CRITICAL: For each step, you MUST define:
1. **Input**: What information/data this step needs (be specific)
2. **Action**: Exactly what will be done  
3. **Tool**: Which tool(s) to use (if any)
4. **Expected Output**: The EXACT format/type of output (e.g., "a number", "a name", "a list of dates")
5. **Next Step Input**: How this output will be used in the next step

Example format:
Step 1: Find the Yankee with most walks in 1977
- Input: Query about 1977 Yankees walks statistics
- Action: Search for 1977 MLB Yankees statistics
- Tool: web_researcher
- Expected Output: Player name (string, e.g., "Mickey Rivers")
- Next Step Input: Use player name to find their At Bats

Step 2: Find At Bats for [Player_Name]
- Input: Player name from Step 1
- Action: Search for [Player_Name] 1977 statistics
- Tool: web_researcher or tavily_search
- Expected Output: Number of At Bats (integer, e.g., "565")
- Next Step Input: This is the final answer

Create a plan following this EXACT format for the given question.
"""
                
                # Add debugging context if this is a plan revision
                if state.get("plan_debugging_requested"):
                    plan_failures = state.get("plan_failures", [])
                    debugging_context = f"""

PLAN DEBUGGING MODE:
Previous plan failed at Step {state.get('current_step', 1)}. 
Failures: {', '.join(plan_failures[-3:])}

Analyze what went wrong and create a REVISED plan that addresses these issues.
"""
                    planning_prompt += debugging_context
                
                # Execute enhanced planning
                def make_planning_call():
                    plan_messages = [SystemMessage(content=self._get_advanced_system_prompt())]
                    plan_messages.extend(messages[-5:])  # Recent context
                    plan_messages.append(HumanMessage(content=planning_prompt))
                    return planning_llm.invoke(plan_messages)
                
                plan_response = advanced_retry_with_recovery(make_planning_call, max_retries=2)
                
                # Parse the enhanced plan to create structured PlanStep objects
                master_plan = self._parse_enhanced_plan(plan_response.content, user_query)
                
                logger.info(f"Created enhanced plan with {len(master_plan)} steps")
                
                # Add planning context to messages
                planning_context = f"""
STRATEGIC PLANNING PHASE - ENHANCED:

Query Analysis: "{user_query}"
Plan: {len(master_plan)} steps with explicit input/output specifications
Current Step: {state.get('current_step', 1)}
Confidence: {state.get('confidence', 0.3):.0%}
Verification Level: {self._assess_verification_level(state)}

Execute the next step in your plan systematically, paying attention to expected outputs.
"""
                
                updated_messages = messages + [SystemMessage(content=planning_context)]
                
                return {
                    "messages": updated_messages,
                    "master_plan": master_plan,
                    "current_step": state.get("current_step", 1),
                    "plan_debugging_requested": False,  # Reset flag
                    "plan_revisions": state.get("plan_revisions", 0) + 1
                }
            else:
                # Use existing plan
                master_plan = state["master_plan"]
                
                # Add current step context
                current_step_idx = state.get('current_step', 1) - 1
                if 0 <= current_step_idx < len(master_plan):
                    current_step = master_plan[current_step_idx]
                    step_context = f"""
EXECUTING STEP {current_step.step_id}:
Description: {current_step.description}
Expected Output: {current_step.expected_outcome}
Tool Needed: {current_step.tool_needed or 'reasoning'}

Focus on achieving the expected output format.
"""
                    updated_messages = messages + [SystemMessage(content=step_context)]
                else:
                    updated_messages = messages
                
                return {
                    "messages": updated_messages,
                    "master_plan": master_plan,
                    "current_step": state.get("current_step", 1)
                }

        def advanced_reasoning_node(state: EnhancedAgentState):
            """Enhanced reasoning with adaptive strategies and reflection."""
            if state.get('log_to_db', False) and self.log_handler:
                self._log_reasoning_step(state)

            messages = state["messages"]
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self._get_advanced_system_prompt())] + messages

            step_count = state.get("step_count", 0)
            confidence = state.get("confidence", 0.3)
            
            # Determine which model to use based on context
            if any(tool_name in str(messages[-1]) if messages else False 
                   for tool_name in ["image_analyzer", "video_analyzer"]):
                # Use vision model if dealing with visual content
                current_llm = self._get_llm("vision")
            elif step_count > 10 or confidence < 0.4:
                # Use deep reasoning model for complex situations
                current_llm = reasoning_llm
            else:
                # Use function calling model for tool interactions
                current_llm = model_with_tools
            
            # Add adaptive prompting based on context
            if step_count == 0:
                guidance = self._get_initial_guidance()
                messages.append(SystemMessage(content=guidance))
            elif step_count % 4 == 0 and step_count > 0:
                reflection = self._get_reflection_prompt(state)
                messages.append(SystemMessage(content=reflection))
            elif confidence < 0.5:
                enhancement = self._get_confidence_enhancement_prompt(state)
                messages.append(SystemMessage(content=enhancement))

            # Intelligent context management
            if len(messages) > 20:
                messages = self._optimize_context(messages)

            def make_advanced_llm_call():
                return current_llm.invoke(messages)

            response = advanced_retry_with_recovery(make_advanced_llm_call, max_retries=3)
            
            # Advanced confidence tracking
            new_confidence = self._calculate_adaptive_confidence(state, response)
            new_step_count = step_count + 1
            
            # Enhanced completion detection
            reasoning_complete = self._assess_completion_readiness(state, response, new_confidence)
            
            return {
                "messages": [response],
                "step_count": new_step_count,
                "confidence": new_confidence,
                "reasoning_complete": reasoning_complete
            }

        def enhanced_tool_execution_node(state: EnhancedAgentState):
            """Advanced tool execution with cross-validation and error recovery."""
            if state.get('log_to_db', False) and self.log_handler:
                self._log_tool_execution(state)
            
            try:
                # Standard tool execution
                tool_node = ToolNode(self.tools)
                tool_output = tool_node.invoke(state)
                
                # Enhanced result validation
                validated_output = self._validate_tool_results(state, tool_output)
                
                # Update tool performance tracking
                self._update_tool_performance(state, validated_output, success=True)
                
                # Reset error recovery attempts after a successful tool call
                validated_output["error_recovery_attempts"] = 0
                
                if state.get('log_to_db', False) and self.log_handler:
                    self._log_tool_results(state, validated_output)

                return validated_output
                
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                
                # Intelligent error recovery
                recovery_guidance = self._generate_error_recovery_guidance(state, e)
                
                # Update error tracking
                error_attempts = state.get("error_recovery_attempts", 0) + 1
                
                return {
                    "messages": [SystemMessage(content=recovery_guidance)],
                    "error_recovery_attempts": error_attempts
                }

        def reflection_and_adaptation_node(state: EnhancedAgentState):
            """DIRECTIVE 2: Enhanced reflection with plan debugging capability."""
            step_count = state.get("step_count", 0)
            confidence = state.get("confidence", 0.3)
            master_plan = state.get("master_plan", [])
            current_step_idx = state.get("current_step", 1) - 1

            # Analyze current plan step failures
            plan_failures = []
            if 0 <= current_step_idx < len(master_plan):
                current_plan_step = master_plan[current_step_idx]
                if current_plan_step.attempts > 0 and not current_plan_step.success:
                    failure_msg = f"Step {current_plan_step.step_id} failed: Expected '{current_plan_step.expected_outcome}' but got errors"
                    plan_failures.append(failure_msg)
                    
                    # Mark step as having errors
                    if current_plan_step.errors:
                        plan_failures.extend(current_plan_step.errors[-2:])  # Last 2 errors

            # Build enhanced reflection prompt with plan debugging
            recent_messages = state["messages"][-8:]
            serialized_history = "\n".join([
                f"{type(m).__name__}: {getattr(m,'content','')[:200]}" for m in recent_messages
            ])
            
            # Extract any error messages
            last_error = ""
            for m in reversed(state["messages"]):
                if isinstance(m, SystemMessage) and ("RECOVERY" in m.content.upper() or "ERROR" in m.content.upper()):
                    last_error = m.content
                    break

            # Determine reflection type based on context
            if plan_failures and state.get("plan_revisions", 0) < 3:
                # DIRECTIVE 2: Plan debugging mode
                logger.info(f"Entering plan debugging mode. Failures: {plan_failures}")
                
                reflection = ReflectionNote(
                    step=step_count,
                    confidence_before=confidence,
                    confidence_after=confidence,
                    insight=f"Plan debugging triggered: {'; '.join(plan_failures)}",
                    decision="debug_plan",
                    timestamp=datetime.now()
                )
                
                reflections = state.get("reflections", [])
                reflections.append(reflection)
                
                # Request plan debugging
                return {
                    "messages": [SystemMessage(content="PLAN DEBUGGING REQUIRED: Revising plan based on failures")],
                    "plan_debugging_requested": True,
                    "plan_failures": plan_failures,
                    "reflections": reflections,
                    "error_recovery_attempts": 0  # reset after reflection
                }
            else:
                # Standard reflection for general issues
                meta_prompt = f"""
You are entering REFLECTION mode. Analyze the current situation and determine the best path forward.

Current Status:
- Step Count: {step_count}
- Confidence: {confidence:.0%}
- Verification Level: {state.get('verification_level', 'basic')}
- Plan Revisions: {state.get('plan_revisions', 0)}

--- Recent History (truncated) ---
{serialized_history}
--- End History ---

Last known issue:
{last_error or 'No specific error, but progress seems stalled'}

Analyze:
1. What progress has been made?
2. What is blocking further progress?
3. Should we continue, pivot, or conclude?

Provide your reflection and recommended next action.
"""

                # Call LLM for reflection
                reflection_response = advanced_retry_with_recovery(
                    lambda: planning_llm.invoke([SystemMessage(content=meta_prompt)]), 
                    max_retries=2
                )

                # Analyze reflection and make decision
                reflection_content = reflection_response.content.lower()
                if "conclude" in reflection_content or "sufficient" in reflection_content:
                    decision = "conclude"
                elif "pivot" in reflection_content or "alternative" in reflection_content:
                    decision = "pivot"
                else:
                    decision = "continue"

                # Record reflection note
                reflection = ReflectionNote(
                    step=step_count,
                    confidence_before=confidence,
                    confidence_after=confidence + 0.1,  # Slight boost after reflection
                    insight=reflection_response.content[:200],
                    decision=decision,
                    timestamp=datetime.now()
                )

                reflections = state.get("reflections", [])
                reflections.append(reflection)

                # Return updated state
                return {
                    "messages": [reflection_response],
                    "reflections": reflections,
                    "confidence": min(0.9, confidence + 0.1),
                    "error_recovery_attempts": 0  # reset after reflection
                }

        def advanced_decision_node(state: EnhancedAgentState):
            """Enhanced decision logic with new synthesis and verification nodes."""
            last_message = state["messages"][-1] if state["messages"] else None
            step_count = state.get("step_count", 0)
            confidence = state.get("confidence", 0.0)
            reasoning_complete = state.get("reasoning_complete", False)
            verification_level = state.get("verification_level", "thorough")  # Default to thorough
            error_attempts = state.get("error_recovery_attempts", 0)
            
            # Safety: Always terminate if no message or step count too high
            if not last_message or step_count >= self.max_reasoning_steps:
                logger.warning(f"Terminating: step_count={step_count}, max_steps={self.max_reasoning_steps}")
                return END
            
            # Continue if there are tool calls to execute
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                logger.debug(f"Step {step_count}: Executing tools")
                return "enhanced_tools"
            
            # Check if we need answer synthesis
            has_gathered_info = any(isinstance(msg, ToolMessage) for msg in state["messages"][-10:])
            answer_synthesized = state.get("answer_synthesized", False)
            
            # DIRECTIVE 1: Route to answer synthesis if we have info but haven't synthesized
            if has_gathered_info and not answer_synthesized and confidence >= 0.5:
                logger.info(f"Step {step_count}: Routing to answer synthesis")
                return "answer_synthesis"
            
            # Check if this is a logic/puzzle question
            original_question = ""
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    original_question = msg.content.lower()
                    break
            
            is_logic_puzzle = any(indicator in original_question for indicator in 
                                ["reverse", "backwards", "decode", "puzzle", "riddle", "trick"])
            
            # DIRECTIVE 4: Route to logic review for puzzle-type questions
            if is_logic_puzzle and not state.get("logic_reviewed", False) and confidence >= 0.4:
                logger.info(f"Step {step_count}: Routing to logic review for puzzle question")
                return "logic_review"
            
            # DIRECTIVE 3: Route to fact checking before final answer
            if reasoning_complete and not state.get("fact_checked", False) and confidence >= 0.6:
                logger.info(f"Step {step_count}: Routing to fact checking")
                return "fact_checking"
            
            # Enhanced completion logic
            if reasoning_complete or step_count >= 15:
                confidence_threshold = {
                    "basic": 0.7,       # Raised thresholds for accuracy
                    "thorough": 0.8,    
                    "exhaustive": 0.9
                }.get(verification_level, 0.8)
                
                # Ensure we've done synthesis and verification before terminating
                if (confidence >= confidence_threshold or step_count >= 15) and \
                   (answer_synthesized or not has_gathered_info):
                    logger.info(f"Terminating: reasoning_complete={reasoning_complete}, confidence={confidence:.2f}, verified={state.get('fact_checked', False)}")
                    return END
            
            # Smart reflection triggers
            if error_attempts >= 2:
                if step_count < self.max_reasoning_steps - 5:
                    logger.debug(f"Step {step_count}: Triggering reflection due to errors")
                    return "reflection"
                else:
                    logger.warning(f"Step {step_count}: Forcing termination due to error attempts near max steps")
                    return END
            
            # Periodic reflection
            if step_count % 6 == 0 and step_count > 0 and step_count < self.max_reasoning_steps - 3:
                logger.debug(f"Step {step_count}: Periodic reflection")
                return "reflection"
            
            # Check for stagnation
            if step_count >= 10 and confidence < 0.4:
                logger.warning(f"Step {step_count}: Low confidence ({confidence:.2f}), forcing termination")
                return END
            
            # Continue reasoning
            logger.debug(f"Step {step_count}: Continuing reasoning")
            return "advanced_reasoning"

        # Build the advanced graph
        graph_builder = StateGraph(EnhancedAgentState)
        
        # Add all nodes
        graph_builder.add_node("strategic_planning", strategic_planning_node)
        graph_builder.add_node("advanced_reasoning", advanced_reasoning_node)
        graph_builder.add_node("enhanced_tools", enhanced_tool_execution_node)
        graph_builder.add_node("reflection", reflection_and_adaptation_node)
        
        # DIRECTIVE 1: Add Answer Synthesis Node
        def answer_synthesis_node(state: EnhancedAgentState):
            """Synthesize gathered information into a precise, direct answer."""
            # Get the user's original question
            original_question = ""
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    original_question = msg.content
                    break
            
            # Gather all retrieved information
            retrieved_data = []
            for msg in state["messages"]:
                if isinstance(msg, ToolMessage) or (isinstance(msg, AIMessage) and "found" in msg.content.lower()):
                    retrieved_data.append(msg.content[:500])  # Limit each piece
            
            synthesis_prompt = f"""You have gathered the following information:
{chr(10).join(retrieved_data[:10])}  

The user's original question was: {original_question}

Synthesize the information to directly answer the user's question in the precise format requested. 
- If the user asked "how many," provide ONLY a number
- If they asked for a name, provide ONLY the name
- If they asked for a location, provide ONLY the location
- Be as concise and direct as possible

Do not include extraneous details. Just provide the direct answer."""

            try:
                synthesis_response = advanced_retry_with_recovery(
                    lambda: synthesis_llm.invoke([SystemMessage(content=synthesis_prompt)]),
                    max_retries=2
                )
                
                # Update state with synthesized answer
                return {
                    "messages": [synthesis_response],
                    "reasoning_complete": True,
                    "confidence": state.get("confidence", 0.8) + 0.1,  # Boost confidence after synthesis
                    "answer_synthesized": True  # Mark that synthesis was performed
                }
            except Exception as e:
                logger.error(f"Answer synthesis failed: {e}")
                return {"messages": [AIMessage(content="Unable to synthesize answer")]}
        
        # DIRECTIVE 3: Add Fact-Checking Node  
        def fact_checking_node(state: EnhancedAgentState):
            """Verify factual accuracy before finalizing answer."""
            # Get the proposed answer from the last message
            last_message = state["messages"][-1] if state["messages"] else None
            if not last_message or not hasattr(last_message, 'content'):
                return state
                
            proposed_answer = last_message.content
            
            # Find source information
            sources = []
            for msg in state["messages"]:
                if isinstance(msg, ToolMessage):
                    sources.append(msg.content[:300])
            
            fact_check_prompt = f"""I believe the answer is: {proposed_answer}

Based on these sources:
{chr(10).join(sources[:5])}

Re-read the source text and verify that it explicitly and unambiguously supports this answer. 
Pay close attention to subtle distinctions (e.g., 'nominator' vs. 'contributor', exact dates, specific numbers).

Does the source explicitly support this exact answer? If not, what is the correct answer based on the sources?
Respond with either:
1. "VERIFIED: [answer]" if the answer is correct
2. "CORRECTED: [correct answer]" if it needs correction
3. "INSUFFICIENT: Need more information" if sources don't contain the answer"""

            try:
                fact_check_response = advanced_retry_with_recovery(
                    lambda: fact_check_llm.invoke([SystemMessage(content=fact_check_prompt)]),
                    max_retries=2
                )
                
                content = fact_check_response.content
                if "CORRECTED:" in content:
                    # Extract corrected answer
                    corrected = content.split("CORRECTED:")[1].strip()
                    return {
                        "messages": [AIMessage(content=corrected)],
                        "confidence": 0.9,  # High confidence after verification
                        "fact_checked": True
                    }
                elif "INSUFFICIENT:" in content:
                    return {
                        "messages": [AIMessage(content="Unable to determine answer from available sources")],
                        "reasoning_complete": False,
                        "confidence": 0.3,
                        "fact_checked": True
                    }
                else:  # VERIFIED
                    return {
                        "confidence": min(0.95, state.get("confidence", 0.8) + 0.1),
                        "fact_checked": True  # Mark that fact checking was performed
                    }
                    
            except Exception as e:
                logger.error(f"Fact checking failed: {e}")
                return state
        
        # DIRECTIVE 4: Add Logic Review Node
        def logic_review_node(state: EnhancedAgentState):
            """Review answer for logical soundness and common sense."""
            # Get the original question
            original_question = ""
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    original_question = msg.content
                    break
                    
            # Get the proposed answer
            last_message = state["messages"][-1] if state["messages"] else None
            if not last_message or not hasattr(last_message, 'content'):
                return state
                
            proposed_answer = last_message.content
            
            logic_prompt = f"""I am about to answer '{proposed_answer}' to the question '{original_question}'.

Is this answer logically sound? Is there a trick or a nuance I might be missing? 
Re-read the question one more time to ensure my interpretation is correct.

Consider:
- Does the answer make logical sense?
- Am I answering what was actually asked?
- Is there wordplay, a riddle, or hidden meaning?
- For numerical answers: is the magnitude reasonable?
- For lists: did I provide the format requested (count vs list)?

Respond with either:
1. "LOGICAL: Proceed with answer" if it makes sense
2. "ILLOGICAL: [explanation and correct answer]" if there's an issue"""

            try:
                logic_response = advanced_retry_with_recovery(
                    lambda: logic_llm.invoke([SystemMessage(content=logic_prompt)]),
                    max_retries=2
                )
                
                content = logic_response.content
                if "ILLOGICAL:" in content:
                    # Extract the corrected reasoning
                    corrected = content.split("ILLOGICAL:")[1].strip()
                    return {
                        "messages": [AIMessage(content=corrected)],
                        "confidence": 0.8,
                        "logic_reviewed": True
                    }
                else:  # LOGICAL
                    return {
                        "confidence": min(0.95, state.get("confidence", 0.8) + 0.05),
                        "logic_reviewed": True  # Mark that logic review was performed
                    }
                    
            except Exception as e:
                logger.error(f"Logic review failed: {e}")
                return state
        
        graph_builder.add_node("answer_synthesis", answer_synthesis_node)
        graph_builder.add_node("fact_checking", fact_checking_node) 
        graph_builder.add_node("logic_review", logic_review_node)

        # Set entry point
        graph_builder.set_entry_point("strategic_planning")
        
        # Add edges
        graph_builder.add_edge("strategic_planning", "advanced_reasoning")
        graph_builder.add_edge("enhanced_tools", "advanced_reasoning")
        graph_builder.add_edge("reflection", "strategic_planning")  # Reflection can trigger replanning
        
        # DIRECTIVE edges: New nodes flow back to decision making
        graph_builder.add_edge("answer_synthesis", "fact_checking")  # Synthesis -> Fact Check
        graph_builder.add_edge("fact_checking", "advanced_reasoning")  # Fact Check -> Continue
        graph_builder.add_edge("logic_review", "advanced_reasoning")   # Logic Review -> Continue
        
        # Add conditional edges from reasoning with all new nodes
        graph_builder.add_conditional_edges(
            "advanced_reasoning",
            advanced_decision_node,
            {
                "enhanced_tools": "enhanced_tools",
                "reflection": "reflection", 
                "advanced_reasoning": "advanced_reasoning",
                "answer_synthesis": "answer_synthesis",     # DIRECTIVE 1
                "fact_checking": "fact_checking",           # DIRECTIVE 3
                "logic_review": "logic_review",             # DIRECTIVE 4
                END: END
            }
        )

        return graph_builder.compile()

    # --- Helper Methods for Advanced Features ---
    
    def _get_initial_guidance(self) -> str:
        return """
INITIAL EXECUTION: Begin systematic problem-solving:
1. Analyze the question type and complexity
2. Follow your strategic plan step by step
3. Use tools purposefully to gather evidence
4. Build confidence through systematic validation
"""

    def _get_reflection_prompt(self, state: EnhancedAgentState) -> str:
        confidence = state.get("confidence", 0.3)
        step_count = state.get("step_count", 0)
        
        return f"""
REFLECTION CHECKPOINT (Step {step_count}, Confidence: {confidence:.0%}):

Critical Questions:
- Have I made sufficient progress toward the answer?
- Is my current confidence level ({confidence:.0%}) appropriate?
- Do I need to verify findings through additional sources?
- Should I adjust my strategy or continue current approach?
- Am I ready to conclude or need more evidence?

Based on reflection, proceed with optimal strategy.
"""

    def _get_confidence_enhancement_prompt(self, state: EnhancedAgentState) -> str:
        return """
CONFIDENCE ENHANCEMENT MODE:

Current confidence is below optimal levels. Enhanced verification recommended:
- Seek additional authoritative sources
- Cross-validate findings through multiple approaches
- Use alternative tools if primary approach uncertain
- Break complex problems into verifiable components
- Build evidence systematically before concluding
"""

    def _optimize_context(self, messages: List[AnyMessage]) -> List[AnyMessage]:
        """Context window optimization with token-budget and summarization."""
        try:
            import tiktoken  # lightweight dependency
            enc = tiktoken.encoding_for_model("gpt-4o") if hasattr(tiktoken, 'encoding_for_model') else tiktoken.get_encoding("cl100k_base")
            def _num_tokens(msg: BaseMessage):
                return len(enc.encode(getattr(msg, 'content', '')))
        except Exception:
            # Fallback: rough char based count (1 token ≈ 4 chars)
            def _num_tokens(msg: BaseMessage):
                return max(1, len(getattr(msg, 'content', '')) // 4)

        TOKEN_LIMIT = 6000  # keep well below 8k model limit

        # Always preserve system prompt(s) and the most recent 6 messages
        preserved = [m for m in messages if isinstance(m, SystemMessage)] + messages[-6:]

        budget_used = sum(_num_tokens(m) for m in preserved)

        # If still within limit, return
        if budget_used <= TOKEN_LIMIT:
            return preserved

        # Otherwise summarize overflow
        overflow = []
        for m in messages:
            if m in preserved:
                continue
            overflow.append(m)
            budget_used += _num_tokens(m)
            if budget_used > TOKEN_LIMIT:
                break

        if overflow:
            summary_prompt = "\n".join([getattr(m, 'content', '')[:500] for m in overflow])
            summary_instruction = (
                "Summarize the following conversation chunks focusing on key facts and decisions."
            )
            try:
                summary_msg = advanced_retry_with_recovery(
                    lambda: self._get_llm("text_generation").invoke([
                        SystemMessage(content=summary_instruction),
                        HumanMessage(content=summary_prompt)
                    ]),
                    max_retries=2
                )
                preserved.insert(1, SystemMessage(content="CONTEXT SUMMARY:" + summary_msg.content))
            except Exception as e:
                logger.warning(f"Failed to summarize context: {e}")

        # final deduplicate but keep order
        seen = set()
        result = []
        for msg in preserved:
            key = (type(msg), getattr(msg, 'content', ''))
            if key not in seen:
                seen.add(key)
                result.append(msg)

        return result

    def _calculate_adaptive_confidence(self, state: EnhancedAgentState, response: AIMessage) -> float:
        """Advanced confidence calculation based on multiple factors."""
        current_confidence = state.get("confidence", 0.3)
        content = response.content.lower() if response.content else ""
        
        # Confidence adjustments based on content indicators
        confidence_boosts = [
            ("found", 0.1), ("confirmed", 0.15), ("verified", 0.2),
            ("according to", 0.08), ("based on", 0.05), ("research shows", 0.12)
        ]
        
        confidence_reductions = [
            ("uncertain", -0.1), ("unclear", -0.08), ("might be", -0.05),
            ("possibly", -0.06), ("error", -0.15), ("failed", -0.12)
        ]
        
        # Apply adjustments
        new_confidence = current_confidence
        for indicator, adjustment in confidence_boosts:
            if indicator in content:
                new_confidence += adjustment
        
        for indicator, adjustment in confidence_reductions:
            if indicator in content:
                new_confidence += adjustment  # adjustment is negative
        
        # Tool usage confidence boost
        if hasattr(response, 'tool_calls') and response.tool_calls:
            new_confidence += 0.1
        
        # Cross-validation bonus
        cross_refs = state.get("cross_validation_sources", [])
        if len(cross_refs) > 1:
            new_confidence += 0.15
        
        # Clamp to valid range
        return max(0.1, min(0.95, new_confidence))

    def _assess_completion_readiness(self, state: EnhancedAgentState, response: AIMessage, confidence: float) -> bool:
        """Enhanced completion assessment with stricter accuracy requirements."""
        content = response.content.lower() if response.content else ""
        
        # Strong completion indicators
        strong_indicators = [
            "final answer", "the answer is", "conclusion", "therefore",
            "result:", "definitively", "confirmed", "verified through"
        ]
        
        # Check for completion signals
        has_completion_signal = any(indicator in content for indicator in strong_indicators)
        
        # Stricter requirements for accuracy
        verification_level = state.get("verification_level", "thorough")  # Default to thorough
        step_count = state.get("step_count", 0)
        cross_validation_sources = len(state.get("cross_validation_sources", []))
        
        # Enhanced completion criteria
        if verification_level == "basic":
            # Require multiple verification steps and sources
            return (has_completion_signal and confidence >= 0.8 and 
                   step_count >= 4 and cross_validation_sources >= 1)
        elif verification_level == "thorough":
            # Require higher confidence and multiple sources
            return (has_completion_signal and confidence >= 0.85 and 
                   step_count >= 6 and cross_validation_sources >= 2)
        else:  # exhaustive
            # Maximum verification for critical accuracy
            return (has_completion_signal and confidence >= 0.9 and 
                   step_count >= 8 and cross_validation_sources >= 3)
        
        return False

    def _validate_tool_results(self, state: EnhancedAgentState, tool_output: dict) -> dict:
        """Cross-validate and enhance tool results."""
        # For now, return as-is but could add sophisticated validation
        # Could include: result consistency checks, source reliability assessment, etc.
        return tool_output

    def _update_tool_performance(self, state: EnhancedAgentState, tool_output: dict, success: bool):
        """Track tool performance for adaptive selection."""
        # Could implement sophisticated tool performance tracking
        pass

    def _generate_error_recovery_guidance(self, state: EnhancedAgentState, error: Exception) -> str:
        """Generate intelligent error recovery strategies."""
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            return """
TIMEOUT RECOVERY: Tool execution timed out.
- Try a simplified query approach
- Use alternative tools if available
- Break complex requests into smaller parts
- Continue with available information if error isn't critical
"""
        elif "not found" in error_str or "404" in error_str:
            return """
NOT FOUND RECOVERY: Resource not available.
- Try alternative search terms
- Use different tools for the same information
- Search for related or broader topics
- Consider if information exists in different format
"""
        else:
            return f"""
GENERAL ERROR RECOVERY: {error}
- Assess if error is critical to the task
- Try alternative approaches or tools
- Modify search strategy or parameters
- Continue with best available information if appropriate
"""

    def _generate_reflection_insight(self, state: EnhancedAgentState) -> str:
        """Generate meaningful reflection insights."""
        confidence = state.get("confidence", 0.3)
        step_count = state.get("step_count", 0)
        
        if confidence < 0.5:
            return f"Low confidence ({confidence:.0%}) suggests need for additional verification"
        elif confidence > 0.8:
            return f"High confidence ({confidence:.0%}) indicates strong evidence base"
        else:
            return f"Moderate confidence ({confidence:.0%}) suggests systematic progress"

    def _make_reflection_decision(self, state: EnhancedAgentState) -> str:
        """Make intelligent reflection-based decisions."""
        confidence = state.get("confidence", 0.3)
        error_attempts = state.get("error_recovery_attempts", 0)
        
        if error_attempts > 2:
            return "pivot"
        elif confidence < 0.4:
            return "verify"
        elif confidence > 0.8:
            return "conclude"
        else:
            return "continue"

    def _log_reasoning_step(self, state: EnhancedAgentState):
        """Log detailed reasoning step information."""
        if self.log_handler:
            log_payload = {
                "run_id": state['run_id'],
                "step_type": "ADVANCED_REASONING",
                "payload": {
                    "step_count": state.get("step_count", 0),
                    "confidence": state.get("confidence", 0.0),
                    "verification_level": state.get("verification_level", "basic"),
                    "plan_step": state.get("current_step", 0)
                }
            }
            self.log_handler.emit(logging.makeLogRecord({'msg': log_payload}))

    def _log_tool_execution(self, state: EnhancedAgentState):
        """Log tool execution details."""
        if self.log_handler:
            last_message = state['messages'][-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                log_payload = {
                    "run_id": state['run_id'],
                    "step_type": "ENHANCED_TOOL_EXECUTION",
                    "payload": {
                        "tool_calls": last_message.tool_calls,
                        "step_count": state.get("step_count", 0),
                        "confidence": state.get("confidence", 0.0)
                    }
                }
                self.log_handler.emit(logging.makeLogRecord({'msg': log_payload}))

    def _log_tool_results(self, state: EnhancedAgentState, tool_output: dict):
        """Log enhanced tool results."""
        if self.log_handler:
            log_payload = {
                "run_id": state['run_id'],
                "step_type": "VALIDATED_TOOL_RESULTS",
                "payload": {
                    "tool_outputs": tool_output.get('messages', []),
                    "step_count": state.get("step_count", 0),
                    "verification_level": state.get("verification_level", "basic")
                }
            }
            self.log_handler.emit(logging.makeLogRecord({'msg': log_payload}))

    def run(self, inputs: dict):
        """Run the advanced agent with full state initialization."""
        # Initialize comprehensive state
        enhanced_inputs = {
            **inputs,
            "master_plan": [],
            "current_step": 1,
            "plan_revisions": 0,
            "reflections": [],
            "confidence_history": [],
            "error_recovery_attempts": 0,
            "step_count": 0,
            "confidence": 0.3,
            "reasoning_complete": False,
            "verification_level": "thorough",  # DIRECTIVE: Default to thorough verification
            "tool_success_rates": {},
            "tool_results": [],
            "cross_validation_sources": [],
            # New state flags for directives
            "answer_synthesized": False,
            "fact_checked": False,
            "logic_reviewed": False,
            "plan_debugging_requested": False,
            "plan_failures": []
        }
        
        return self.graph.invoke(enhanced_inputs)

    def stream(self, inputs: dict):
        """Stream the advanced agent execution."""
        # Initialize comprehensive state
        enhanced_inputs = {
            **inputs,
            "master_plan": [],
            "current_step": 1,
            "plan_revisions": 0,
            "reflections": [],
            "confidence_history": [],
            "error_recovery_attempts": 0,
            "step_count": 0,
            "confidence": 0.3,
            "reasoning_complete": False,
            "verification_level": "thorough",  # DIRECTIVE: Default to thorough verification
            "tool_success_rates": {},
            "tool_results": [],
            "cross_validation_sources": [],
            # New state flags for directives
            "answer_synthesized": False,
            "fact_checked": False,
            "logic_reviewed": False,
            "plan_debugging_requested": False,
            "plan_failures": []
        }
        
        return self.graph.stream(enhanced_inputs) 