#!/usr/bin/env python3
"""
AI Agent Application - Fixed Import Structure
This file includes the proper import configuration to resolve the
"attempted relative import with no known parent package" error.
"""

# Fix Python path to recognize the package structure
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Standard library imports
import os
import uuid
import logging
import re
import json
import time
import datetime
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import threading

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Import modular components
from config import config, Environment
from session import SessionManager, ParallelAgentPool
from ui import (
    create_main_chat_interface, 
    create_gaia_evaluation_tab, 
    create_analytics_tab,
    create_documentation_tab,
    create_custom_css,
    format_message_with_steps,
    export_conversation_to_file
)
from gaia_logic import GAIAEvaluator, GAIA_AVAILABLE

# Only load .env file if not in a Hugging Face Space
if config.environment != Environment.HUGGINGFACE_SPACE:
    load_dotenv()

# Configure logging based on config - MUST BE BEFORE ANY LOGGER USAGE
logging.basicConfig(
    level=getattr(logging, config.logging.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.propagate = False

# Import the new FSM agent and integration hub
try:
    from src.tools.base_tool import BaseTool
    from src.reasoning.reasoning_path import ReasoningPath, ReasoningType
    from src.errors.error_category import ErrorCategory
    from src.advanced_agent_fsm import FSMReActAgent, validate_user_prompt, ValidationResult
    from src.database import get_supabase_client, SupabaseLogHandler
    # Import integration hub for centralized component management
    from src.integration_hub import initialize_integrations, cleanup_integrations, get_integration_hub, get_tools
    # Import enhanced tools for GAIA
    from src.tools_enhanced import get_enhanced_tools
    from src.knowledge_ingestion import KnowledgeIngestionService
    # Add GAIA optimizations imports
    from src.gaia.caching import response_cache, question_cache, error_cache
    from src.gaia.metrics import gaia_metrics
    from src.gaia.tools import gaia_chess_analyzer, gaia_music_search, gaia_country_code_lookup, gaia_mathematical_calculator
    from src.advanced_agent_fsm import analyze_question_type, create_gaia_optimized_plan, GAIAAgentState
    from src.gaia.testing import GAIATestPatterns
except ImportError as e:
    import sys
    import os
    print(f"Import Error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Contents of src directory: {os.listdir('src') if os.path.exists('src') else 'src directory not found'}")
    raise

# Helper functions for chat processing
def format_chat_history(history: List[List[str]]) -> List[Dict[str, str]]:
    """Format chat history for agent processing."""
    formatted = []
    for user_msg, assistant_msg in history:
        formatted.append({"role": "user", "content": user_msg})
        formatted.append({"role": "assistant", "content": assistant_msg})
    return formatted

def extract_final_answer(response_output: str) -> str:
    """Extract clean final answer from agent response."""
    if not response_output:
        return "I couldn't generate a response."
    
    # Remove common formatting artifacts
    cleaned = response_output.strip()
    
    # Remove LaTeX boxing notation
    cleaned = re.sub(r'\\boxed\{([^}]+)\}', r'\1', cleaned)
    
    # Remove "final answer" prefixes
    cleaned = re.sub(r'^[Tt]he\s+final\s+answer\s+is\s*:?\s*', '', cleaned)
    cleaned = re.sub(r'^[Ff]inal\s+answer\s*:?\s*', '', cleaned)
    
    # Remove tool call artifacts
    cleaned = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'<invoke>.*?</invoke>', '', cleaned, flags=re.DOTALL)
    
    # Remove mathematical formatting
    cleaned = re.sub(r'\$([^$]+)\$', r'\1', cleaned)
    
    # Clean up extra whitespace and punctuation
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip()
    
    # Remove trailing punctuation if it's just formatting
    if cleaned.endswith('.') and len(cleaned) < 50:
        cleaned = cleaned[:-1]
    
    return cleaned if cleaned else "I couldn't generate a response."

class AIAgentApp:
    """Main application class for the AI Agent."""
    
    def __init__(self):
        """Initialize with proper error handling and fallbacks."""
        self.agent = None
        self.tools = []
        self.session_manager = None
        self.log_handler = None
        self.model_name = None
        self.integration_hub = None
        self.setup_environment()
        self.initialize_components()
        
    def setup_environment(self):
        # Enable tracing if configured
        if config.is_tracing_enabled:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = config.langsmith_project
            os.environ["LANGCHAIN_ENDPOINT"] = config.langsmith_endpoint
            os.environ["LANGCHAIN_API_KEY"] = config.langsmith_api_key
            logger.info("LangSmith tracing enabled")
        # Initialize Supabase logging if available
        if config.has_database:
            try:
                from src.database import get_supabase_client, SupabaseLogHandler
                supabase_client = get_supabase_client()
                self.log_handler = SupabaseLogHandler(supabase_client)
                logger.addHandler(self.log_handler)
                logger.info("Supabase logging enabled")
            except Exception as e:
                logger.warning(f"Supabase logging disabled: {e}")
        # Set model name
        self.model_name = config.primary_model
        logger.info(f"Using model: {self.model_name}")
        
    def initialize_components(self):
        """Initialize with comprehensive error handling."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Initialize integration hub first
                self._initialize_integration_hub()
                
                # Initialize tools from integration hub
                self.tools = self._initialize_tools_with_fallback()
                
                # Initialize agent with validated configuration
                self.agent = self._initialize_agent()
                
                # Initialize session manager
                self.session_manager = SessionManager(max_sessions=10)
                logger.info("All components initialized successfully")
                break
            except Exception as e:
                logger.error(f"Initialization attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    self._initialize_minimal_setup()
                else:
                    time.sleep(2 ** attempt)
    
    async def _initialize_integration_hub(self):
        """Initialize the integration hub"""
        try:
            self.integration_hub = get_integration_hub()
            await initialize_integrations()
            logger.info("Integration hub initialized successfully")
        except Exception as e:
            logger.warning(f"Integration hub initialization failed: {e}")
            self.integration_hub = None
    
    def _initialize_tools_with_fallback(self):
        try:
            # Use integration hub if available
            if self.integration_hub and self.integration_hub.is_ready():
                return self.integration_hub.get_tools()
            else:
                # Fallback to direct tool import
                from src.tools_enhanced import get_enhanced_tools
                return get_enhanced_tools()
        except ImportError:
            logger.warning("Enhanced tools unavailable, using basic tools")
            return self._get_basic_tools()
    
    def _get_basic_tools(self):
        # Minimal tool fallback
        from src.tools import file_reader
        return [file_reader]
    
    def _initialize_agent(self):
        try:
            from src.advanced_agent_fsm import FSMReActAgent
            return FSMReActAgent(
                tools=self.tools,
                model_name=self.model_name,
                log_handler=self.log_handler,
                model_preference="balanced",
                use_crew=True
            )
        except Exception as e:
            logger.error(f"Failed to initialize FSM agent: {e}")
            raise
    
    def _initialize_minimal_setup(self):
        """Initialize minimal setup when all else fails"""
        logger.warning("Initializing minimal setup")
        self.tools = self._get_basic_tools()
        self.agent = None
        self.session_manager = SessionManager(max_sessions=5)
        
    def process_gaia_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process GAIA benchmark questions."""
        return GAIAEvaluator.process_questions(questions, self.agent)
        
    def process_chat_message(self, message: str, history: list, log_to_db: bool = True, session_id: str = None) -> tuple:
        """Process chat message with timeout and error handling."""
        if not session_id:
            session_id = str(uuid.uuid4())
            
        def timeout_handler(signum, frame):
            raise TimeoutError("Agent response timeout")
            
        try:
            # Validate input
            validation_result = validate_user_prompt(message)
            if not validation_result.is_valid:
                return history + [[message, f"Invalid input: {validation_result.error_message}"]], session_id
            
            # Process with agent
            response = self.agent.run(message)
            final_answer = extract_final_answer(response)
            
            # Log to database if enabled
            if log_to_db and self.log_handler:
                self.log_handler.log_interaction(session_id, message, final_answer)
                
            return history + [[message, final_answer]], session_id
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_message = f"I encountered an error: {str(e)}"
            return history + [[message, error_message]], session_id
        
    def build_interface(self):
        """Build the Gradio interface."""
        return build_gradio_interface(
            self.process_chat_message,
            self.process_gaia_questions,
            self.session_manager
        )

def build_gradio_interface(process_chat_message, process_gaia_questions, session_manager):
    """Build the Gradio interface with all tabs and features."""
    
    # Create custom CSS
    custom_css = create_custom_css()
    
    with gr.Blocks(css=custom_css, title="AI Agent System") as interface:
        gr.Markdown("# 🤖 Advanced AI Agent System")
        
        with gr.Tabs():
            # Main Chat Tab
            with gr.Tab("💬 Advanced Chat"):
                create_main_chat_interface(process_chat_message, session_manager)
            
            # GAIA Evaluation Tab
            if GAIA_AVAILABLE:
                with gr.Tab("🏆 GAIA Evaluation"):
                    create_gaia_evaluation_tab(process_gaia_questions)
            
            # Analytics Tab
            with gr.Tab("📊 Analytics"):
                create_analytics_tab()
            
            # Documentation Tab
            with gr.Tab("📚 Documentation"):
                create_documentation_tab()
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("Built with ❤️ using Gradio and LangChain")
    
    return interface

def main():
    """Main entry point."""
    try:
        # Initialize the application
        app = AIAgentApp()
        
        # Build the interface
        interface = app.build_interface()
        
        # Launch the interface
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True
        )
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        print(f"Failed to start application: {e}")
        sys.exit(1)
    finally:
        # Cleanup integration hub
        try:
            if app.integration_hub:
                asyncio.run(cleanup_integrations())
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

if __name__ == "__main__":
    main() 