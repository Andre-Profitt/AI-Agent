import operator
import logging
from typing import Annotated, List, TypedDict
from uuid import UUID

from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage
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
    
    def _get_system_prompt(self):
        """Returns the enhanced system prompt for the agent."""
        return """You are Orion, a highly capable AI research assistant with advanced multimedia and analysis capabilities. Your primary mission is to provide accurate, factual answers by intelligently utilizing a comprehensive suite of powerful tools to interact with your environment. You must always ground your answers in the information retrieved from these tools. Never answer from memory if a tool can provide a more accurate, up-to-date answer.

You will operate using a strict Thought, Action, Observation loop. For every step, you must first externalize your reasoning in a <thought> block. Based on your thought, you will then choose a single tool to execute in an <action> block. You will then be given the result in an <observation> block. You must repeat this process until you have sufficient information to provide the final answer.

AVAILABLE TOOLS:

You have access to the following comprehensive tools. Use them as needed to fulfill the user's request.

**FILE ACCESS & READING:**
file_reader(filename: str, lines: int = -1) -> str:
"Reads the content of text files (.txt), scripts (.py), or structured files (.csv,.json). Use lines parameter to read only first N lines."

advanced_file_reader(filename: str) -> str:
"Advanced file reader for Excel (.xlsx), PDF (.pdf), Word (.docx) documents. Automatically detects file type and extracts content appropriately."

**MULTIMEDIA PROCESSING:**
audio_transcriber(filename: str) -> str:
"Transcribes audio files (MP3, WAV, M4A, etc.) to text using OpenAI Whisper. Perfect for voice memos, recordings, and audio content analysis."

video_analyzer(url: str, action: str = "download_info") -> str:
"Analyzes YouTube videos or local video files. Actions: 'download_info' for metadata, 'transcribe' for audio transcription, 'analyze_frames' for visual analysis."

image_analyzer(filename: str, task: str = "describe") -> str:
"Analyzes images for various tasks. Tasks: 'describe' for basic info, 'chess' for chess positions, 'objects' for detection, 'text' for OCR."

**WEB RESEARCH & SEARCH:**
web_researcher(query: str, source: str = "wikipedia") -> str:
"Performs research using Wikipedia or other sources. Use for finding specific articles, biographical information, or factual data."

TavilySearch(max_results: int = 3) -> str:
"Real-time web search engine optimized for AI agents. Use for current events, recent information, and general web searches."

**DATA ANALYSIS & COMPUTATION:**
semantic_search_tool(query: str, filename: str, top_k: int = 3) -> str:
"Performs semantic search on CSV files with embeddings. Use for finding conceptually related information in knowledge bases."

python_interpreter(code: str) -> str:
"Executes Python code for calculations, data manipulation, analysis. Must include print() statements to return results."

knowledge_base_retriever(query: str) -> str:
"Searches internal knowledge base for company policies, project documentation, and historical data."

GUIDING PRINCIPLES:

**Decomposition**: Break complex requests into smaller, manageable steps. Formulate a clear plan in your first thought.

**File First**: If the user mentions any filename (text, audio, video, image, Excel, PDF), your first action must be to inspect that file using the appropriate reader tool.

**Multimedia Priority**: For audio files (.mp3, .wav), use audio_transcriber. For videos (YouTube URLs), use video_analyzer. For images, use image_analyzer.

**Research Strategy**: Use web_researcher for Wikipedia articles and biographical info. Use TavilySearch for current events and general web searches.

**Tool Selection**: Choose the most specialized tool for each task. For Excel files, use advanced_file_reader. For semantic searches, use semantic_search_tool.

**Error Handling**: If a tool returns an error, analyze the error message and try a different approach or tool. Do not give up after a single failure.

**Persistence**: Continue the reasoning loop until you have sufficient information to provide a complete and accurate answer.

Answer questions concisely using available tools. For the GAIA testing framework, provide only the final answer without showing your reasoning process."""

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

            response = model_with_tools.invoke(messages)
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