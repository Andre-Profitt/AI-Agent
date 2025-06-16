import os
import uuid
import logging
from typing import List, Tuple

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.agent import ReActAgent
from src.database import get_supabase_client, SupabaseLogHandler
from src.tools import get_tools

# --- Initialization ---
# Load environment variables
load_dotenv()

# Configure root logger to be very permissive
logging.basicConfig(level=logging.DEBUG)
# Get a specific logger for our application
logger = logging.getLogger(__name__)
# Prevent passing logs to the root logger's handlers
logger.propagate = False

# Initialize Supabase client and custom log handler
try:
    supabase_client = get_supabase_client()
    supabase_handler = SupabaseLogHandler(supabase_client)
    # Add the custom handler to our app's logger
    logger.addHandler(supabase_handler)
    logger.info("Supabase logger initialized successfully.")
    LOGGING_ENABLED = True
except Exception as e:
    logger.error(f"Failed to initialize Supabase logger: {e}. Trajectory logging will be disabled.")
    LOGGING_ENABLED = False

# Initialize tools and agent
try:
    tools = get_tools()
    agent_graph = ReActAgent(tools=tools, log_handler=supabase_handler if LOGGING_ENABLED else None).graph
    logger.info("ReAct Agent initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize agent: {e}", exc_info=True)
    # Exit if agent cannot be created, as the app is non-functional
    exit("Critical error: Agent could not be initialized. Check logs.")

# --- Gradio Interface Logic ---

def format_chat_history(chat_history: List[List[str]]) -> List:
    """Formats Gradio's chat history into a list of LangChain messages."""
    messages = []
    for turn in chat_history:
        user_message, ai_message = turn
        if user_message:
            messages.append(HumanMessage(content=user_message))
        if ai_message:
            # This part is tricky as AI message might contain tool calls.
            # For simplicity, we'll treat it as a plain AIMessage here.
            # A more robust implementation would parse this for tool calls.
            messages.append(AIMessage(content=ai_message))
    return messages

def chat_interface_logic(message: str, history: List[List[str]], log_to_db: bool):
    """
    The core logic for the Gradio chat interface.
    It streams the agent's response.
    """
    logger.info(f"Received new message: '{message}'")
    
    # Format history and add new message
    formatted_history = format_chat_history(history)
    current_messages = formatted_history + [HumanMessage(content=message)]
    
    # Prepare agent input
    run_id = uuid.uuid4()
    agent_input = {
        "messages": current_messages,
        "run_id": run_id,
        "log_to_db": log_to_db and LOGGING_ENABLED
    }

    # Stream the agent's response
    full_response = ""
    intermediate_steps = ""
    
    try:
        for chunk in agent_graph.stream(agent_input, stream_mode="values"):
            last_message = chunk["messages"][-1]

            # Handle intermediate tool calls and thoughts
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                thought = last_message.additional_kwargs.get("thought", "")
                tool_call_str = "\n".join(
                    f"  - Tool: `{tc['name']}`, Args: `{tc['args']}`" for tc in last_message.tool_calls
                )
                intermediate_steps += f"🤔 **Thinking:**\n{thought}\n\n" if thought else ""
                intermediate_steps += f"🛠️ **Action:**\n{tool_call_str}\n\n"
                yield intermediate_steps, full_response

            # Handle tool outputs (observations)
            elif isinstance(last_message, ToolMessage):
                intermediate_steps += f"👀 **Observation:**\n  - `{last_message.content}`\n\n"
                yield intermediate_steps, full_response

            # Handle the final answer
            elif isinstance(last_message, AIMessage):
                full_response = last_message.content
                yield intermediate_steps, full_response

    except Exception as e:
        logger.error(f"An error occurred during agent execution for run_id {run_id}: {e}", exc_info=True)
        yield intermediate_steps, f"An error occurred: {e}"


def build_gradio_interface():
    """Builds and returns the Gradio chat interface."""
    with gr.Blocks(theme=gr.themes.Default(primary_hue="blue"), title="Multi-Tool ReAct Agent") as demo:
        gr.Markdown("# 🤖 Multi-Tool ReAct Agent")
        gr.Markdown(
            "Powered by Groq, LangGraph, LlamaIndex, Tavily, and Supabase. "
            "Ask me a question that requires web search, knowledge base retrieval, or code execution."
        )

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    [],
                    label="Conversation",
                    bubble_fill=False,
                    height=600,
                )
                message_box = gr.Textbox(
                    placeholder="e.g., What is the result of 123 * 456?",
                    label="Your Message",
                    show_label=False,
                    lines=2,
                )
            with gr.Column(scale=1):
                gr.Markdown("### Agent Trajectory")
                gr.Markdown("Intermediate thoughts, actions, and observations from the agent will appear here.")
                intermediate_steps_display = gr.Markdown(label="Intermediate Steps")
                log_to_db_checkbox = gr.Checkbox(
                    label="Log Trajectory to Supabase",
                    value=True,
                    visible=LOGGING_ENABLED
                )

        def on_submit(message, chat_history, log_to_db):
            chat_history.append([message, None])
            
            # Use a generator to stream responses
            response_generator = chat_interface_logic(message, chat_history, log_to_db)
            
            for steps, response in response_generator:
                chat_history[-1] = [message, response]
                yield chat_history, steps

        message_box.submit(
            on_submit,
            [message_box, chatbot, log_to_db_checkbox],
            [chatbot, intermediate_steps_display]
        ).then(
            lambda: gr.update(value=""), None, [message_box], queue=False
        )

    return demo

if __name__ == "__main__":
    app = build_gradio_interface()
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true"
    ) 