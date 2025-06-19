"""
Main application with enhanced FSM-based agent.
"""

import logging
import gradio as gr
from typing import Dict, Any, List, Optional
import os

from .advanced_agent_fsm import FSMReActAgent
from .data_quality import DataQualityLevel
from .advanced_reasoning import ReasoningType
from .tools import get_tools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize agent
tools = get_tools()
agent = FSMReActAgent(
    tools=tools,
    model_name="gpt-4",
    quality_level=DataQualityLevel.THOROUGH,
    reasoning_type=ReasoningType.LAYERED
)

def chat_interface_logic_sync(
    message: str,
    history: List[List[str]]
) -> Dict[str, Any]:
    """Enhanced chat interface with improved error handling and data quality."""
    try:
        # Run agent
        result = agent.run(message)
        
        # Handle different statuses
        if result["status"] == "success":
            return {
                "response": result["answer"],
                "reasoning": result["reasoning_path"],
                "error": None
            }
        elif result["status"] == "error":
            return {
                "response": None,
                "reasoning": None,
                "error": f"Error: {result['error']}"
            }
        else:
            return {
                "response": "Incomplete response",
                "reasoning": result["tool_results"],
                "error": None
            }
            
    except Exception as e:
        logger.error(f"Error in chat interface: {str(e)}")
        return {
            "response": None,
            "reasoning": None,
            "error": f"Unexpected error: {str(e)}"
        }

def create_interface():
    """Create the Gradio interface."""
    with gr.Blocks() as interface:
        gr.Markdown("# Enhanced AI Agent")
        
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot()
                msg = gr.Textbox(
                    label="Input",
                    placeholder="Enter your message here..."
                )
                clear = gr.Button("Clear")
            
            with gr.Column():
                reasoning = gr.JSON(label="Reasoning Path")
                error = gr.Textbox(label="Error", visible=False)
        
        def user(user_message, history):
            return "", history + [[user_message, None]]
        
        def bot(history):
            result = chat_interface_logic_sync(
                history[-1][0],
                history[:-1]
            )
            
            if result["error"]:
                error.update(value=result["error"], visible=True)
                return history + [[None, "An error occurred. Please try again."]]
            
            error.update(value="", visible=False)
            reasoning.update(value=result["reasoning"])
            return history + [[None, result["response"]]]
        
        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch() 