import os
import uuid
import logging
import re
from typing import List, Tuple

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from src.advanced_agent import AdvancedReActAgent
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
    # Use the more sophisticated AdvancedReActAgent which includes strategic planning,
    # reflection nodes, and robust error handling improvements.
    agent_graph = AdvancedReActAgent(
        tools=tools,
        log_handler=supabase_handler if LOGGING_ENABLED else None
    ).graph
    logger.info("AdvancedReAct Agent initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize agent: {e}", exc_info=True)
    # Instead of exiting, try to provide more specific error details
    print(f"AGENT INITIALIZATION ERROR: {e}")
    print(f"Error Type: {type(e).__name__}")
    
    # Check if tools were initialized successfully
    try:
        tools = get_tools()
        print(f"✅ Tools initialized successfully: {len(tools)} tools available")
        print(f"Tool names: {[tool.name for tool in tools]}")
    except Exception as tool_error:
        print(f"❌ Tools initialization failed: {tool_error}")
    
    # Exit only if it's a critical issue we can't recover from
    import traceback
    print("Full traceback:")
    traceback.print_exc()
    exit("Critical error: Agent could not be initialized. Check logs above for details.")

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

def extract_final_answer(response: str) -> str:
    """
    Extracts only the final answer from sophisticated reasoning chains.
    Handles complex responses that include planning, reasoning, and conclusions.
    """
    if not response or not isinstance(response, str):
        return response
    
    # Remove common prefixes
    prefixes_to_remove = [
        "final answer:",
        "the answer is:",
        "answer:",
        "result:",
        "conclusion:",
        "therefore:",
        "so the answer is:",
        "based on my analysis,",
        "after analyzing,",
        "in conclusion,",
        "to summarize,",
        "the final result is:",
        "my final answer is:",
        "the correct answer is:",
        "ultimately,",
        "in summary,",
        "to conclude,"
    ]
    
    response_lower = response.lower().strip()
    
    # Check for prefixes and extract what comes after
    for prefix in prefixes_to_remove:
        if prefix in response_lower:
            # Find the last occurrence of the prefix
            idx = response_lower.rfind(prefix)
            if idx != -1:
                extracted = response[idx + len(prefix):].strip()
                if extracted:
                    response = extracted
                    break
    
    # Remove explanation patterns
    explanation_patterns = [
        r"based on.*?[,.]",
        r"according to.*?[,.]",
        r"after.*?analysis.*?[,.]",
        r"following.*?research.*?[,.]",
        r"from.*?information.*?[,.]",
        r"using.*?tool.*?[,.]",
        r"through.*?search.*?[,.]"
    ]
    
    for pattern in explanation_patterns:
        response = re.sub(pattern, "", response, flags=re.IGNORECASE | re.DOTALL)
    
    # Handle sentences that start with reasoning
    reasoning_starters = [
        "this is because",
        "the reason is",
        "this indicates",
        "this suggests",
        "this shows that",
        "this means",
        "this tells us",
        "we can see that",
        "it appears that",
        "the data shows",
        "evidence suggests"
    ]
    
    lines = response.split('\n')
    final_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        line_lower = line.lower()
        is_reasoning = any(starter in line_lower for starter in reasoning_starters)
        
        if not is_reasoning:
            final_lines.append(line)
    
    if final_lines:
        response = '\n'.join(final_lines)
    
    # Clean up common artifacts
    response = response.strip()
    response = re.sub(r'^[*\-•]\s*', '', response)  # Remove bullet points
    response = re.sub(r'\s+', ' ', response)  # Normalize whitespace
    
    # Extract just numbers/codes if that's what's needed
    response = response.strip('.,!?')
    
    # If response is very long (>100 chars) and contains multiple sentences,
    # try to extract the most important part
    if len(response) > 100 and '. ' in response:
        sentences = response.split('. ')
        # Look for the shortest sentence that might be the answer
        potential_answers = [s.strip() for s in sentences if len(s.strip()) < 50]
        if potential_answers:
            # Prefer sentences with numbers, dates, or short factual statements
            for ans in potential_answers:
                if any(char.isdigit() for char in ans) or len(ans.split()) <= 5:
                    response = ans
                    break
            else:
                response = potential_answers[0]
    
    return response.strip()


def chat_interface_logic(message: str, history: List[List[str]], log_to_db: bool):
    """
    The core logic for the Gradio chat interface with sophisticated reasoning 
    and clean final answer extraction.
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
        "log_to_db": log_to_db and LOGGING_ENABLED,
        "plan": "",
        "step_count": 0,
        "confidence": 0.3,
        "reasoning_complete": False
    }

    # Stream the agent's response
    full_response = ""
    intermediate_steps = ""
    reasoning_chain = ""
    step_count = 0
    
    try:
        for chunk in agent_graph.stream(agent_input, stream_mode="values"):
            last_message = chunk["messages"][-1]
            current_confidence = chunk.get("confidence", 0.0)
            current_step = chunk.get("step_count", 0)

            # Handle intermediate tool calls and thoughts
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                step_count = current_step
                thought = last_message.content if last_message.content else "Planning next action..."
                tool_call_str = "\n".join(
                    f"  - **{tc['name']}**: {tc['args']}" for tc in last_message.tool_calls
                )
                
                intermediate_steps += f"**Step {step_count}** (Confidence: {current_confidence:.0%})\n"
                intermediate_steps += f"🤔 **Reasoning:** {thought}\n\n"
                intermediate_steps += f"🛠️ **Action:**\n{tool_call_str}\n\n"
                
                reasoning_chain += f"Step {step_count}: {thought}\n"
                yield intermediate_steps, full_response

            # Handle tool outputs (observations)
            elif isinstance(last_message, ToolMessage):
                observation = last_message.content[:200] + "..." if len(last_message.content) > 200 else last_message.content
                intermediate_steps += f"👀 **Observation:** {observation}\n\n"
                reasoning_chain += f"Observation: {last_message.content}\n"
                yield intermediate_steps, full_response

            # Handle reasoning and final answers
            elif isinstance(last_message, AIMessage):
                step_count = current_step
                content = last_message.content
                
                # Check if this is likely a final answer
                is_final = (
                    current_confidence > 0.7 or 
                    chunk.get("reasoning_complete", False) or
                    any(indicator in content.lower() for indicator in [
                        "final answer", "the answer is", "therefore", "conclusion", "result:"
                    ])
                )
                
                if is_final:
                    # Extract clean final answer
                    clean_answer = extract_final_answer(content)
                    full_response = clean_answer
                    
                    intermediate_steps += f"**Final Step {step_count}** (Confidence: {current_confidence:.0%})\n"
                    intermediate_steps += f"✅ **Final Answer:** {clean_answer}\n\n"
                    
                    logger.info(f"Final answer extracted: '{clean_answer}' from raw response: '{content[:100]}...'")
                else:
                    # This is intermediate reasoning
                    intermediate_steps += f"**Step {step_count}** (Confidence: {current_confidence:.0%})\n"
                    intermediate_steps += f"🧠 **Reasoning:** {content}\n\n"
                    reasoning_chain += f"Step {step_count}: {content}\n"
                
                yield intermediate_steps, full_response

    except Exception as e:
        logger.error(f"An error occurred during agent execution for run_id {run_id}: {e}", exc_info=True)
        yield intermediate_steps, f"An error occurred: {e}"


def build_gradio_interface():
    """Builds and returns the Gradio chat interface."""
    with gr.Blocks(theme=gr.themes.Default(primary_hue="blue"), title="Multi-Model AI Agent") as demo:
        gr.Markdown("# 🤖 Multi-Model AI Agent")
        gr.Markdown(
            "Powered by multiple Groq models, LangGraph, LlamaIndex, Tavily, and Supabase. "
            "Features adaptive model selection for optimal performance across different types of questions. "
            "Ask me anything that requires web search, knowledge base retrieval, code execution, or complex reasoning."
        )

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    [],
                    label="Conversation",
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