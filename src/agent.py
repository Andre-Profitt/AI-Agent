import operator
import logging
from typing import Annotated, List, TypedDict
from uuid import UUID

from langchain_core.messages import AnyMessage, BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Configure logging
logger = logging.getLogger(__name__)

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
        """Initializes and returns the Groq LLM."""
        return ChatGroq(temperature=0, model_name="llama3-70b-8192")

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

            response = model_with_tools.invoke(state["messages"])
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