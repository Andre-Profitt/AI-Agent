import os
import uuid
import logging
import re
from typing import List, Tuple

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.agent_simple import SimpleReActAgent  # Use the simplified agent
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

# Initialize tools and simplified agent
try:
    tools = get_tools()
    agent_graph = SimpleReActAgent(tools=tools, log_handler=supabase_handler if LOGGING_ENABLED else None).graph
    logger.info("Simplified ReAct Agent initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize agent: {e}", exc_info=True)
    # Exit if agent cannot be created, as the app is non-functional
    exit("Critical error: Agent could not be initialized. Check logs.")

# --- Enhanced Answer Extraction ---

def extract_final_answer(response: str) -> str:
    """
    Extract clean final answer from agent response with aggressive cleaning.
    """
    if not response or not isinstance(response, str):
        return response
    
    # Remove common prefixes aggressively
    prefixes_to_remove = [
        "final answer:", "the answer is:", "answer:", "result:", "conclusion:",
        "therefore:", "so the answer is:", "based on my analysis,", "after analyzing,",
        "in conclusion,", "to summarize,", "the final result is:", "my final answer is:",
        "the correct answer is:", "ultimately,", "in summary,", "to conclude,",
        "based on my research,", "according to my findings,"
    ]
    
    response_lower = response.lower().strip()
    
    # Check for prefixes and extract what comes after
    for prefix in prefixes_to_remove:
        if prefix in response_lower:
            idx = response_lower.rfind(prefix)
            if idx != -1:
                extracted = response[idx + len(prefix):].strip()
                if extracted:
                    response = extracted
                    break
    
    # Remove explanation patterns
    explanation_patterns = [
        r"based on.*?[,.]", r"according to.*?[,.]", r"after.*?analysis.*?[,.]",
        r"following.*?research.*?[,.]", r"from.*?information.*?[,.]",
        r"using.*?tool.*?[,.]", r"through.*?search.*?[,.]"
    ]
    
    for pattern in explanation_patterns:
        response = re.sub(pattern, "", response, flags=re.IGNORECASE | re.DOTALL)
    
    # Clean up formatting aggressively
    response = response.strip()
    response = re.sub(r'^[*\-•]\s*', '', response)  # Remove bullet points
    response = re.sub(r'\s+', ' ', response)  # Normalize whitespace
    response = response.strip('.,!?')
    
    # If multiline, prefer the shortest factual line
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    if lines and len(lines) > 1:
        # Look for the shortest line that looks like an answer
        shortest_answer = min(lines, key=len)
        if len(shortest_answer) < 50:  # Reasonable answer length
            response = shortest_answer
    
    return response.strip()

# --- Gradio Interface Logic ---

def format_chat_history(chat_history: List[List[str]]) -> List:
    """Formats Gradio's chat history into a list of LangChain messages."""
    messages = []
    for turn in chat_history:
        user_message, ai_message = turn
        if user_message:
            messages.append(HumanMessage(content=user_message))
        if ai_message:
            messages.append(AIMessage(content=ai_message))
    return messages

def chat_interface_logic(message: str, history: List[List[str]], log_to_db: bool):
    """
    Simplified but effective chat interface logic with robust error handling.
    """
    logger.info(f"Received message: '{message}'")
    
    # Format history and add new message
    formatted_history = format_chat_history(history)
    current_messages = formatted_history + [HumanMessage(content=message)]
    
    # Prepare agent input with simplified state
    run_id = uuid.uuid4()
    agent_input = {
        "messages": current_messages,
        "run_id": run_id,
        "log_to_db": log_to_db and LOGGING_ENABLED,
        "step_count": 0,
        "confidence": 0.3,
        "reasoning_complete": False
    }

    # Stream the agent's response with robust error handling
    full_response = ""
    intermediate_steps = ""
    step_count = 0
    
    try:
        for chunk in agent_graph.stream(agent_input, stream_mode="values"):
            last_message = chunk.get("messages", [])[-1] if chunk.get("messages") else None
            current_confidence = chunk.get("confidence", 0.0)
            current_step = chunk.get("step_count", 0)

            # Handle tool calls
            if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                step_count = current_step
                thought = last_message.content if last_message.content else "Using tools to gather information..."
                tool_call_str = "\n".join(
                    f"  - **{tc.get('name', 'unknown')}**" for tc in last_message.tool_calls
                )
                
                intermediate_steps += f"**Step {step_count}** (Confidence: {current_confidence:.0%})\n"
                intermediate_steps += f"🤔 **Reasoning:** {thought[:100]}...\n\n"
                intermediate_steps += f"🛠️ **Tools:** {tool_call_str}\n\n"
                
                yield intermediate_steps, full_response

            # Handle tool outputs
            elif isinstance(last_message, ToolMessage):
                observation = last_message.content[:150] + "..." if len(last_message.content) > 150 else last_message.content
                intermediate_steps += f"👀 **Result:** {observation}\n\n"
                yield intermediate_steps, full_response

            # Handle final answers
            elif isinstance(last_message, AIMessage):
                step_count = current_step
                content = last_message.content
                
                # Check if this is likely a final answer
                is_final = (
                    current_confidence > 0.6 or 
                    chunk.get("reasoning_complete", False) or
                    not hasattr(last_message, 'tool_calls') or
                    not last_message.tool_calls
                )
                
                if is_final:
                    # Extract clean final answer
                    clean_answer = extract_final_answer(content)
                    full_response = clean_answer
                    
                    intermediate_steps += f"**Final Step {step_count}** (Confidence: {current_confidence:.0%})\n"
                    intermediate_steps += f"✅ **Answer:** {clean_answer}\n\n"
                    
                    logger.info(f"Final answer: '{clean_answer}'")
                    break
                else:
                    # Intermediate reasoning
                    intermediate_steps += f"**Step {step_count}** (Confidence: {current_confidence:.0%})\n"
                    intermediate_steps += f"🧠 **Thinking:** {content[:100]}...\n\n"
                
                yield intermediate_steps, full_response
            
            # Safety: Break if too many steps
            if step_count > 10:
                logger.warning("Step limit reached, forcing conclusion")
                break

    except Exception as e:
        logger.error(f"Agent execution error for run_id {run_id}: {e}", exc_info=True)
        error_msg = f"An error occurred: {str(e)[:100]}..."
        yield intermediate_steps + f"\n💥 **Error:** {error_msg}", error_msg

def build_gradio_interface():
    """Builds the Gradio chat interface with simplified agent."""
    with gr.Blocks(theme=gr.themes.Default(primary_hue="blue"), title="Simplified Multi-Tool ReAct Agent") as demo:
        gr.Markdown("# 🤖 Reliable Multi-Tool ReAct Agent")
        gr.Markdown(
            "**Optimized for stability and efficiency** | "
            "Groq + LangGraph + LlamaIndex + Tavily + Supabase | "
            "Smart reasoning with clean, concise answers."
        )

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    [],
                    label="Chat - Direct Answers Only",
                    height=600,
                )
                message_box = gr.Textbox(
                    placeholder="Ask any question requiring research, analysis, or computation...",
                    label="Your Message",
                    show_label=False,
                    lines=2,
                )
            with gr.Column(scale=1):
                gr.Markdown("### 🔍 Agent Process")
                gr.Markdown("Watch the agent's reasoning and tool usage here.")
                intermediate_steps_display = gr.Markdown(label="Agent Reasoning", value="Ready for input...")
                log_to_db_checkbox = gr.Checkbox(
                    label="Log to Database",
                    value=True,
                    visible=LOGGING_ENABLED
                )

        def on_submit(message, chat_history, log_to_db):
            chat_history.append([message, None])
            
            # Use generator to stream responses
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