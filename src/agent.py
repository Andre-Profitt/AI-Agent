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

# --- Agent State Definition ---

class AgentState(TypedDict):
    """
    Represents the state of our agent.

    Attributes:
        messages: The history of messages in the conversation.
        run_id: A unique identifier for the agent run.
        log_to_db: A flag to control database logging for this run.
    """
    messages: Annotated[List[AnyMessage], operator.add]
    run_id: UUID
    log_to_db: bool

# --- Agent Graph Implementation ---

class ReActAgent:
    """
    A ReAct (Reasoning and Acting) agent implemented with LangGraph.
    """
    def __init__(self, tools: list, log_handler: logging.Handler = None):
        self.tools = tools
        self.log_handler = log_handler
        self.graph = self._build_graph()

    def _get_llm(self):
        """Initializes and returns the Groq LLM with rate-limit friendly settings."""
        # Use the faster, cheaper Llama 3.1 8B Instant model
        return ChatGroq(
            temperature=0, 
            model_name="llama-3.1-8b-instant",  # Upgraded: 750 tok/s, $0.05-0.08, 128k context
            max_tokens=1024,  # Limit response length to avoid context issues
            max_retries=1,    # Let our custom retry logic handle this
            request_timeout=30  # Shorter timeout for faster failure detection
        )
    
    def _get_system_prompt(self):
        """Returns a system prompt optimized for final-answer-only responses."""
        return """You are Orion, an AI assistant. Use tools to gather information, then provide ONLY the final answer.

CRITICAL: Your response must be ONLY the direct answer. No explanations, no reasoning, no "Final Answer:" prefix.

EXAMPLES:
- Question: "What is 2+2?" → Response: "4"
- Question: "Who invented the telephone?" → Response: "Alexander Graham Bell"  
- Question: "How many albums?" → Response: "3"

PROCESS:
1. Use tools to gather needed information
2. Once you have the answer, respond with ONLY that answer
3. No additional text, explanations, or prefixes

AVAILABLE TOOLS: file_reader, semantic_search_tool, web_researcher, python_interpreter, tavily_search, image_analyzer, video_analyzer, audio_transcriber, and others.

Remember: Final response = direct answer only."""

    def _build_graph(self):
        """
        Builds the LangGraph state machine for the agent.
        """
        llm = self._get_llm()
        model_with_tools = llm.bind_tools(self.tools)

        # Define the graph nodes
        def reason_node(state: AgentState):
            """
            The "think" node: calls the LLM to decide the next action.
            """
            if state.get('log_to_db', False) and self.log_handler:
                log_payload = {
                    "run_id": state['run_id'],
                    "step_type": "REASON",
                    "payload": {"messages": [msg.dict() for msg in state['messages']]}
                }
                self.log_handler.emit(logging.makeLogRecord({'msg': log_payload}))

            # Inject system prompt as the first message if not already present
            messages = state["messages"]
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=self._get_system_prompt())] + messages

            # Trim messages if too long to avoid context length errors
            if len(messages) > 10:  # Keep only recent messages
                messages = messages[:1] + messages[-9:]  # Keep system prompt + last 9 messages

            def make_llm_call():
                return model_with_tools.invoke(messages)

            response = exponential_backoff_retry(make_llm_call, max_retries=3)
            return {"messages": [response]}

        # Use the pre-built ToolNode for executing tools
        tool_node = ToolNode(self.tools)

        def log_tool_call(state: AgentState):
            """A wrapper around the tool_node to log the action and observation."""
            if state.get('log_to_db', False) and self.log_handler:
                last_message = state['messages'][-1]
                if last_message.tool_calls:
                    log_payload = {
                        "run_id": state['run_id'],
                        "step_type": "ACTION",
                        "payload": {"tool_calls": last_message.tool_calls}
                    }
                    self.log_handler.emit(logging.makeLogRecord({'msg': log_payload}))
            
            # Execute the tool
            tool_output = tool_node.invoke(state)

            if state.get('log_to_db', False) and self.log_handler:
                log_payload = {
                    "run_id": state['run_id'],
                    "step_type": "OBSERVATION",
                    "payload": {"tool_outputs": [out.dict() for out in tool_output['messages']]}
                }
                self.log_handler.emit(logging.makeLogRecord({'msg': log_payload}))

            return tool_output

        # Define the conditional edge
        def should_continue(state: AgentState):
            """
            Determines the next step: call a tool or finish.
            """
            last_message = state["messages"][-1]
            if last_message.tool_calls:
                return "tools"
            
            if state.get('log_to_db', False) and self.log_handler:
                log_payload = {
                    "run_id": state['run_id'],
                    "step_type": "FINAL_ANSWER",
                    "payload": {"final_answer": last_message.content}
                }
                self.log_handler.emit(logging.makeLogRecord({'msg': log_payload}))

            return END

        # Build the graph
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("reason", reason_node)
        graph_builder.add_node("tools", log_tool_call)

        graph_builder.set_entry_point("reason")
        graph_builder.add_conditional_edges(
            "reason",
            should_continue,
        )
        graph_builder.add_edge("tools", "reason")

        return graph_builder.compile()

    def run(self, inputs: dict):
        """
        Invokes the agent graph with the given inputs.
        """
        return self.graph.invoke(inputs)

    def stream(self, inputs: dict):
        """
        Streams the agent's execution steps.
        """
        return self.graph.stream(inputs) 