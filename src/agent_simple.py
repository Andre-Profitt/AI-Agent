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

# --- Conservative Rate Limiting ---
class ConservativeRateLimiter:
    """Ultra-conservative rate limiter to prevent API overuse."""
    
    def __init__(self, max_requests_per_minute=30):  # Much more conservative
        self.max_requests = max_requests_per_minute
        self.requests = []
        
    def wait_if_needed(self):
        """Wait if approaching rate limits with conservative buffer."""
        now = time.time()
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        
        if len(self.requests) >= self.max_requests:
            sleep_time = 65 - (now - self.requests[0])  # Extra buffer
            if sleep_time > 0:
                logger.warning(f"Conservative rate limiting: waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        self.requests.append(now)

# Global conservative rate limiter
rate_limiter = ConservativeRateLimiter(max_requests_per_minute=25)

def safe_retry(func, max_retries=2):
    """Safe retry with conservative backoff."""
    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed()
            return func()
        except Exception as e:
            error_str = str(e)
            if any(err in error_str for err in ["429", "rate_limit", "token"]):
                wait_time = 10 + (attempt * 5)  # Conservative backoff
                logger.warning(f"Rate limit protection: waiting {wait_time}s")
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    raise
            else:
                raise
    return None

# --- Simplified Agent State ---
class SimpleAgentState(TypedDict):
    """Simplified state to reduce complexity."""
    messages: Annotated[List[AnyMessage], operator.add]
    run_id: UUID
    log_to_db: bool
    step_count: int
    confidence: float
    reasoning_complete: bool

# --- Simplified ReAct Agent ---
class SimpleReActAgent:
    """
    Simplified ReAct agent optimized for reliability and efficiency.
    Maintains sophisticated reasoning but reduces token usage and complexity.
    """
    
    def __init__(self, tools: list, log_handler: logging.Handler = None):
        self.tools = tools
        self.log_handler = log_handler
        self.graph = self._build_simple_graph()
        self.max_steps = 8  # Much more conservative limit

    def _get_llm(self):
        """Conservative LLM configuration."""
        return ChatGroq(
            temperature=0.0,  # Zero temperature for consistency
            model_name="llama-3.1-8b-instant",
            max_tokens=1024,  # Reduced token limit
            max_retries=1,
            request_timeout=30
        )
    
    def _get_simple_system_prompt(self):
        """Concise but effective system prompt."""
        return """You are Orion, an expert AI assistant. Solve problems efficiently and provide only the final answer.

APPROACH:
1. Analyze the question
2. Use appropriate tools to gather information  
3. Provide the direct answer only

TOOLS AVAILABLE:
- web_researcher: Search Wikipedia/encyclopedic sources
- semantic_search_tool: Search knowledge base
- python_interpreter: Calculations and data processing
- tavily_search: Current information search
- file_reader: Read text files
- advanced_file_reader: Read Excel/PDF/Word files
- audio_transcriber: Transcribe audio files
- video_analyzer: Analyze YouTube videos
- image_analyzer: Analyze images

CRITICAL RULES:
- When you have the answer, respond with ONLY the direct answer
- No "Based on my research" or "The answer is" prefixes
- No explanations or reasoning in final response
- Just the precise answer the question asks for

EXAMPLES:
Q: "How many albums?" → A: "3"
Q: "What country code?" → A: "EGY" 
Q: "Calculate 1729/7" → A: "247"

Be efficient, be accurate, be concise."""

    def _build_simple_graph(self):
        """Build simplified but effective graph."""
        llm = self._get_llm()
        model_with_tools = llm.bind_tools(self.tools)

        def reasoning_node(state: SimpleAgentState):
            """Streamlined reasoning with early stopping."""
            step_count = state.get("step_count", 0)
            
            # Force early conclusion if approaching limits
            if step_count >= self.max_steps - 2:
                conclusion_prompt = """
CONCLUSION REQUIRED: You've reached the step limit. 
Provide your best answer based on the information gathered so far.
Respond with ONLY the direct answer - no explanations.
"""
                messages = state["messages"] + [SystemMessage(content=conclusion_prompt)]
            else:
                messages = state["messages"]
                
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self._get_simple_system_prompt())] + messages

            # Aggressive context management to save tokens
            if len(messages) > 8:
                system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
                recent_msgs = messages[-5:]  # Very aggressive pruning
                messages = system_msgs[:1] + recent_msgs

            def make_call():
                return model_with_tools.invoke(messages)

            try:
                response = safe_retry(make_call, max_retries=2)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                # Return error guidance
                return {
                    "messages": [SystemMessage(content="Provide your best answer with available information.")],
                    "reasoning_complete": True
                }
            
            # Update state conservatively
            new_step_count = step_count + 1
            new_confidence = min(0.8, state.get("confidence", 0.3) + 0.2)
            
            # Early completion detection
            content = response.content.lower() if response.content else ""
            reasoning_complete = (
                new_step_count >= self.max_steps or
                any(indicator in content for indicator in ["answer", "result", "conclusion"]) or
                not response.tool_calls
            )
            
            return {
                "messages": [response],
                "step_count": new_step_count,
                "confidence": new_confidence,
                "reasoning_complete": reasoning_complete
            }

        def tool_node_safe(state: SimpleAgentState):
            """Safe tool execution with robust error handling."""
            try:
                tool_node = ToolNode(self.tools)
                result = tool_node.invoke(state)
                return result
            except Exception as e:
                logger.error(f"Tool error: {e}")
                error_type = str(e).lower()
                
                if "rate limit" in error_type or "429" in error_type:
                    guidance = "Rate limit hit. Conclude with available information."
                elif "tool_use_failed" in error_type or "400" in error_type:
                    guidance = "Tool call failed. Try simpler approach or conclude."
                else:
                    guidance = "Error occurred. Proceed with available information."
                
                return {
                    "messages": [SystemMessage(content=guidance)],
                    "reasoning_complete": True  # Force completion on errors
                }

        def should_continue_safe(state: SimpleAgentState):
            """Conservative continuation logic."""
            last_message = state["messages"][-1]
            step_count = state.get("step_count", 0)
            reasoning_complete = state.get("reasoning_complete", False)
            
            # Stop immediately if any stopping condition is met
            if (step_count >= self.max_steps or 
                reasoning_complete or 
                not hasattr(last_message, 'tool_calls') or 
                not last_message.tool_calls):
                return END
            
            return "tools"

        # Build minimal graph
        graph_builder = StateGraph(SimpleAgentState)
        graph_builder.add_node("reasoning", reasoning_node)
        graph_builder.add_node("tools", tool_node_safe)

        graph_builder.set_entry_point("reasoning")
        graph_builder.add_conditional_edges(
            "reasoning",
            should_continue_safe,
            {"tools": "tools", "end": END}
        )
        graph_builder.add_edge("tools", "reasoning")

        return graph_builder.compile()

    def run(self, inputs: dict):
        """Run with safe initialization."""
        safe_inputs = {
            **inputs,
            "step_count": 0,
            "confidence": 0.3,
            "reasoning_complete": False
        }
        
        try:
            return self.graph.invoke(safe_inputs)
        except Exception as e:
            logger.error(f"Graph execution failed: {e}")
            # Return minimal safe response
            return {
                "messages": [SystemMessage(content="Unable to process request due to system constraints.")],
                "step_count": 1,
                "confidence": 0.1,
                "reasoning_complete": True
            }

    def stream(self, inputs: dict):
        """Stream with safe initialization."""
        safe_inputs = {
            **inputs,
            "step_count": 0,
            "confidence": 0.3,
            "reasoning_complete": False
        }
        
        try:
            return self.graph.stream(safe_inputs)
        except Exception as e:
            logger.error(f"Graph streaming failed: {e}")
            # Yield safe fallback
            yield {
                "messages": [SystemMessage(content="System error occurred.")],
                "step_count": 1,
                "confidence": 0.1,
                "reasoning_complete": True
            }

# Legacy compatibility
ReActAgent = SimpleReActAgent 