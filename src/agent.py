import operator
import logging
import time
import random
from typing import Annotated, List, TypedDict
from uuid import UUID

from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Configure logging
logger = logging.getLogger(__name__)

# --- Rate Limiting Configuration ---
class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_requests_per_minute=60):  # Conservative limit for faster model
        self.max_requests = max_requests_per_minute
        self.requests = []
        
    def wait_if_needed(self):
        """Wait if we're approaching rate limits."""
        now = time.time()
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        if len(self.requests) >= self.max_requests:
            sleep_time = 60 - (now - self.requests[0]) + 1  # Wait until oldest request expires + 1 second
            if sleep_time > 0:
                logger.warning(f"Rate limit approaching, sleeping for {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
        
        self.requests.append(now)

# Global rate limiter
rate_limiter = RateLimiter(max_requests_per_minute=60)  # Increased for faster model

def exponential_backoff_retry(func, max_retries=3):
    """Retry function with exponential backoff for rate limit errors."""
    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed()  # Proactive rate limiting
            return func()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit_exceeded" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                    logger.warning(f"Rate limit hit, waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Max retries reached for rate limit error: {e}")
                    raise
            elif "context_length_exceeded" in error_str:
                logger.error(f"Context length exceeded: {e}")
                raise
            else:
                # For other errors, don't retry
                raise
    
    return None

# --- Enhanced Agent State Definition ---

class AgentState(TypedDict):
    """
    Enhanced agent state for sophisticated reasoning while maintaining clean final output.

    Attributes:
        messages: The history of messages in the conversation.
        run_id: A unique identifier for the agent run.
        log_to_db: A flag to control database logging for this run.
        plan: Current strategic plan for solving the problem.
        step_count: Number of reasoning steps taken.
        confidence: Confidence level in current answer (0-1).
        reasoning_complete: Flag indicating reasoning is done, ready for final answer.
    """
    messages: Annotated[List[AnyMessage], operator.add]
    run_id: UUID
    log_to_db: bool
    plan: str
    step_count: int
    confidence: float
    reasoning_complete: bool

# --- World-Class ReAct Agent Implementation ---

class ReActAgent:
    """
    A sophisticated ReAct agent with advanced planning and reflection,
    but clean, concise final answers.
    """
    def __init__(self, tools: list, log_handler: logging.Handler = None):
        self.tools = tools
        self.log_handler = log_handler
        self.graph = self._build_graph()
        self.max_reasoning_steps = 15  # Prevent infinite loops

    def _get_llm(self):
        """Initializes and returns the Groq LLM optimized for reasoning."""
        return ChatGroq(
            temperature=0.1,  # Slight temperature for creative problem solving
            model_name="llama-3.1-8b-instant",
            max_tokens=2048,  # Increased for detailed reasoning
            max_retries=1,
            request_timeout=60  # Longer timeout for complex reasoning
        )
    
    def _get_system_prompt(self):
        """Returns a sophisticated system prompt for world-class ReAct reasoning."""
        return """You are Orion, a world-class AI research assistant with sophisticated reasoning capabilities.

🎯 MISSION: Solve complex problems through systematic reasoning, planning, and tool use. Provide ONLY the final answer when complete.

🧠 SOPHISTICATED REASONING FRAMEWORK:

**PHASE 1: UNDERSTAND & PLAN**
- Analyze the question deeply
- Identify what type of problem this is
- Break down into logical steps
- Plan which tools to use and when

**PHASE 2: SYSTEMATIC EXECUTION**  
- Execute your plan step by step
- Use tools strategically to gather information
- Validate findings from multiple sources when possible
- Handle errors gracefully with alternative approaches

**PHASE 3: REFLECTION & VERIFICATION**
- Check if you have sufficient information
- Verify your answer makes sense
- Cross-reference with multiple sources if critical
- Assess your confidence level

**PHASE 4: CONCISE FINAL ANSWER**
When your reasoning is complete and you're confident in your answer:
- Respond with ONLY the direct, concise answer
- No reasoning, no explanations, no "Based on..." 
- Just the precise answer the question asks for

🛠️ TOOL STRATEGY:
- **file_reader**: Text files, CSV, code files
- **advanced_file_reader**: Excel, PDF, Word documents  
- **audio_transcriber**: MP3, WAV audio files
- **video_analyzer**: YouTube videos, video analysis
- **image_analyzer**: Images, charts, chess positions
- **web_researcher**: Wikipedia, encyclopedic research
- **semantic_search_tool**: Knowledge base searches
- **python_interpreter**: Calculations, data processing
- **tavily_search**: Current events, real-time search

🔍 REFLECTION CHECKPOINTS:
Every few steps, ask yourself:
- Am I making progress toward the answer?
- Do I need more information?
- Should I verify this finding?
- Am I confident enough to answer?
- What might I be missing?

⚡ EXECUTION EXCELLENCE:
- Be methodical and systematic
- Use multiple sources for verification when important
- Handle file/tool errors gracefully
- Don't repeat unsuccessful approaches
- Build evidence step by step

🎯 CRITICAL FINAL ANSWER RULES:
When you have the complete answer, respond with ONLY:
- The direct answer (number, name, code, etc.)
- No prefixes like "Final Answer:" or "The answer is:"
- No explanations or reasoning steps
- No additional context or qualifiers

EXAMPLES OF PERFECT FINAL ANSWERS:
- Question: "How many albums?" → Response: "3"
- Question: "What chess move?" → Response: "Rd5"  
- Question: "Which country code?" → Response: "EGY"
- Question: "What is the surname?" → Response: "Johnson"

Remember: Think sophisticatedly, plan strategically, reason systematically, but answer concisely."""

    def _build_graph(self):
        """
        Builds a sophisticated LangGraph state machine with clean final output.
        """
        llm = self._get_llm()
        model_with_tools = llm.bind_tools(self.tools)

        # Planning and Reasoning Node
        def reasoning_node(state: AgentState):
            """
            Sophisticated reasoning node with planning and reflection.
            """
            if state.get('log_to_db', False) and self.log_handler:
                log_payload = {
                    "run_id": state['run_id'],
                    "step_type": "REASONING",
                    "payload": {
                        "step_count": state.get("step_count", 0),
                        "confidence": state.get("confidence", 0.0)
                    }
                }
                self.log_handler.emit(logging.makeLogRecord({'msg': log_payload}))

            messages = state["messages"]
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self._get_system_prompt())] + messages

            step_count = state.get("step_count", 0)
            
            # Add planning guidance for initial steps
            if step_count == 0:
                planning_prompt = """
INITIAL PLANNING: Before taking any actions, think through:
1. What type of question is this?
2. What information do I need to find?
3. Which tools should I use in what order?
4. How will I verify my answer?

Then proceed with systematic execution.
"""
                messages.append(SystemMessage(content=planning_prompt))
            
            # Add reflection prompts for longer reasoning chains
            elif step_count > 3 and step_count % 3 == 0:
                reflection_prompt = f"""
REFLECTION CHECKPOINT (Step {step_count}):
- Have I made progress toward answering the question?
- Do I have sufficient information to be confident?
- Should I verify my findings from another source?
- Am I ready to provide the final answer?

Continue systematically or provide the final answer if confident.
"""
                messages.append(SystemMessage(content=reflection_prompt))

            # Intelligent context management
            if len(messages) > 15:
                # Keep system prompt and recent messages
                system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
                recent_msgs = messages[-10:]
                messages = system_msgs[:2] + recent_msgs

            def make_llm_call():
                return model_with_tools.invoke(messages)

            response = exponential_backoff_retry(make_llm_call, max_retries=3)
            
            # Update reasoning state
            new_step_count = step_count + 1
            new_confidence = state.get("confidence", 0.3)
            
            # Increase confidence based on progress indicators
            content_lower = response.content.lower()
            if any(indicator in content_lower for indicator in ["found", "according", "based on", "result"]):
                new_confidence = min(0.9, new_confidence + 0.15)
            elif response.tool_calls:
                new_confidence = min(0.85, new_confidence + 0.1)
            
            # Check if reasoning appears complete
            reasoning_complete = False
            final_indicators = ["final answer", "the answer is", "therefore", "conclusion", "result:"]
            if any(indicator in content_lower for indicator in final_indicators) and new_confidence > 0.7:
                reasoning_complete = True
            
            return {
                "messages": [response],
                "step_count": new_step_count,
                "confidence": new_confidence,
                "reasoning_complete": reasoning_complete
            }

        # Enhanced Tool Node
        tool_node = ToolNode(self.tools)

        def enhanced_tool_node(state: AgentState):
            """Enhanced tool execution with advanced error handling and cross-validation."""
            if state.get('log_to_db', False) and self.log_handler:
                last_message = state['messages'][-1]
                if last_message.tool_calls:
                    log_payload = {
                        "run_id": state['run_id'],
                        "step_type": "ACTION",
                        "payload": {
                            "tool_calls": last_message.tool_calls,
                            "step_count": state.get("step_count", 0)
                        }
                    }
                    self.log_handler.emit(logging.makeLogRecord({'msg': log_payload}))
            
            try:
                tool_output = tool_node.invoke(state)
                
                # Enhanced result validation and cross-validation
                validated_output = self._validate_and_cross_check_results(state, tool_output)
                
                if state.get('log_to_db', False) and self.log_handler:
                    log_payload = {
                        "run_id": state['run_id'],
                        "step_type": "VALIDATED_OBSERVATION",
                        "payload": {
                            "tool_outputs": [out.dict() for out in validated_output['messages']],
                            "step_count": state.get("step_count", 0),
                            "validation_score": validated_output.get("validation_score", 1.0)
                        }
                    }
                    self.log_handler.emit(logging.makeLogRecord({'msg': log_payload}))

                return validated_output
                
            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                
                # Advanced error recovery with adaptive strategies
                recovery_strategy = self._determine_recovery_strategy(state, e)
                error_guidance = self._generate_recovery_guidance(e, recovery_strategy)
                
                # Track error for adaptive learning
                error_count = state.get("error_recovery_attempts", 0) + 1
                
                return {
                    "messages": [SystemMessage(content=error_guidance)],
                    "error_recovery_attempts": error_count
                }

        def should_continue(state: AgentState):
            """Sophisticated decision logic for continuing or finishing."""
            last_message = state["messages"][-1]
            step_count = state.get("step_count", 0)
            confidence = state.get("confidence", 0.0)
            reasoning_complete = state.get("reasoning_complete", False)
            
            # Continue if there are tool calls to execute
            if last_message.tool_calls:
                return "tools"
            
            # Stop if max steps reached
            if step_count >= self.max_reasoning_steps:
                logger.warning(f"Max reasoning steps ({self.max_reasoning_steps}) reached")
                return END
            
            # Stop if reasoning appears complete and confidence is high
            if reasoning_complete and confidence > 0.7:
                if state.get('log_to_db', False) and self.log_handler:
                    log_payload = {
                        "run_id": state['run_id'],
                        "step_type": "FINAL_ANSWER",
                        "payload": {
                            "final_answer": last_message.content,
                            "total_steps": step_count,
                            "final_confidence": confidence
                        }
                    }
                    self.log_handler.emit(logging.makeLogRecord({'msg': log_payload}))
                return END
            
            # Stop if very high confidence regardless
            if confidence > 0.9:
                return END
            
            # Continue reasoning
            return "reasoning"

        # Build the sophisticated graph
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("reasoning", reasoning_node)
        graph_builder.add_node("tools", enhanced_tool_node)

        graph_builder.set_entry_point("reasoning")
        graph_builder.add_conditional_edges(
            "reasoning",
            should_continue,
            {
                "tools": "tools",
                "reasoning": "reasoning",
                "end": END
            }
        )
        graph_builder.add_edge("tools", "reasoning")

        return graph_builder.compile()

    def run(self, inputs: dict):
        """
        Invokes the enhanced agent graph with sophisticated reasoning.
        """
        # Initialize enhanced state
        if "plan" not in inputs:
            inputs["plan"] = ""
        if "step_count" not in inputs:
            inputs["step_count"] = 0
        if "confidence" not in inputs:
            inputs["confidence"] = 0.3
        if "reasoning_complete" not in inputs:
            inputs["reasoning_complete"] = False
            
        return self.graph.invoke(inputs)

    def stream(self, inputs: dict):
        """
        Streams the agent's sophisticated execution steps.
        """
        # Initialize enhanced state
        if "plan" not in inputs:
            inputs["plan"] = ""
        if "step_count" not in inputs:
            inputs["step_count"] = 0
        if "confidence" not in inputs:
            inputs["confidence"] = 0.3
        if "reasoning_complete" not in inputs:
            inputs["reasoning_complete"] = False
            
        return self.graph.stream(inputs)

    def _validate_and_cross_check_results(self, state: AgentState, tool_output: dict) -> dict:
        """Validate and cross-check tool results (simplified implementation)."""
        # For now, return the original output with validation metadata
        # In a full implementation, this would do sophisticated cross-validation
        validation_score = 0.8  # Default good score
        
        return {
            **tool_output,
            "validation_score": validation_score
        }
    
    def _determine_recovery_strategy(self, state: AgentState, error: Exception) -> str:
        """Determine the best recovery strategy for an error."""
        error_str = str(error).lower()
        
        if "429" in error_str or "rate limit" in error_str:
            return "rate_limit_backoff"
        elif "400" in error_str or "tool_use_failed" in error_str:
            return "simplify_approach"
        elif "recursion limit" in error_str:
            return "force_conclusion"
        else:
            return "alternative_tool"
    
    def _generate_recovery_guidance(self, error: Exception, strategy: str) -> str:
        """Generate recovery guidance based on error and strategy."""
        if strategy == "rate_limit_backoff":
            return """
🔄 RATE LIMIT RECOVERY: Implementing simplified approach.
- Reduce token usage with more direct queries
- Use fewer tools per step
- Conclude more quickly to avoid hitting limits
- Focus on essential information only
"""
        
        elif strategy == "simplify_approach":
            return """
🎯 TOOL ERROR RECOVERY: Simplifying approach.
- Use simpler, more direct tool calls
- Avoid complex multi-function calls
- Try alternative tools if available
- Focus on basic information gathering
"""
        
        elif strategy == "force_conclusion":
            return """
⚡ RECURSION LIMIT REACHED: Forcing conclusion.
- Stop complex reasoning and provide best answer with current information
- Avoid further tool calls unless absolutely necessary
- Conclude based on evidence gathered so far
- Prioritize direct answers over extensive analysis
"""
        
        else:
            return """
🔧 GENERAL ERROR RECOVERY: Trying alternative approach.
- Switch to different tools if available
- Simplify query strategy
- Continue with available information
- Focus on core question requirements
""" 