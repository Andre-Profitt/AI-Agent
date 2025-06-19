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
from src.infrastructure.di.container import get_container, setup_container

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

# Import the new FSM agent instead of the old one
try:
    from src.tools.base_tool import BaseTool
    from src.reasoning.reasoning_path import ReasoningPath, ReasoningType
    from src.errors.error_category import ErrorCategory
    from src.advanced_agent_fsm import FSMReActAgent, validate_user_prompt
    from src.database import get_supabase_client, SupabaseLogHandler
    # Import enhanced tools for GAIA
    from src.tools_enhanced import get_enhanced_tools
    from src.knowledge_ingestion import KnowledgeIngestionService
except ImportError as e:
    import sys
    import os
    print(f"Import Error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Contents of src directory: {os.listdir('src') if os.path.exists('src') else 'src directory not found'}")
    raise

class AIAgentApp:
    """Main application class for the AI Agent."""
    
    def __init__(self):
        """Initialize the AI Agent application."""
        self.agent = None
        self.tools = []
        self.setup_environment()
        self.initialize_components()
        
    def setup_environment(self):
        """Set up environment variables and configuration."""
        # Initialize Supabase client and custom log handler
        try:
            self.supabase_client = get_supabase_client()
            self.supabase_handler = SupabaseLogHandler(self.supabase_client)
            # Add the custom handler to our app's logger
            logger.addHandler(self.supabase_handler)
            logger.info("Supabase logger initialized successfully.")
            self.logging_enabled = True
        except Exception as e:
            logger.error(f"Failed to initialize Supabase logger: {e}. Trajectory logging will be disabled.")
            self.logging_enabled = False
        
    def initialize_components(self):
        """Initialize all application components."""
        try:
            # Initialize tools
            self.tools = get_enhanced_tools()
            logger.info(f"Enhanced tools loaded successfully: {len(self.tools)} tools available")
            
            # Initialize agent
            try:
                from src.next_gen_integration import create_next_gen_agent
                self.agent = create_next_gen_agent(
                    enable_all_features=True,
                    log_handler=self.supabase_handler if self.logging_enabled else None,
                    model_preference="balanced",
                    use_crew=True
                )
                logger.info("Next-generation FSM Agent initialized successfully with all features.")
            except ImportError:
                self.agent = FSMReActAgent(
                    tools=self.tools,
                    log_handler=self.supabase_handler if self.logging_enabled else None,
                    model_preference="balanced",
                    use_crew=True
                )
                logger.info("Standard FSM-based ReAct Agent initialized successfully.")
            
            # Initialize other components
            self.session_manager = SessionManager()
            self.parallel_pool = ParallelAgentPool()
            self.gaia_evaluator = GAIAEvaluator()
            
        except Exception as e:
            logger.critical(f"Failed to initialize components: {e}", exc_info=True)
            raise
    
    def process_gaia_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process GAIA evaluation questions."""
        return self.gaia_evaluator.run_evaluation(questions)
    
    def process_chat_message(self, message: str, history: List[List[str]], log_to_db: bool = True, session_id: str = None) -> Tuple[str, List[List[str]]]:
        """Process a chat message and return the response."""
        try:
            # Format chat history
            messages = format_chat_history(history)
            
            # Process with agent
            response = self.agent.run({"input": message})
            
            # Extract clean answer
            answer = extract_final_answer(response["output"])
            
            # Update history
            history.append([message, answer])
            
            # Log if enabled
            if log_to_db and self.logging_enabled:
                self._log_interaction(message, answer, session_id)
            
            return answer, history
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            return error_msg, history
    
    def _log_interaction(self, message: str, response: str, session_id: str = None):
        """Log interaction to database."""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        try:
            # Log to Supabase
            self.supabase_client.table("interactions").insert({
                "session_id": session_id,
                "message": message,
                "response": response,
                "timestamp": datetime.datetime.now().isoformat()
            }).execute()
        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")
    
    def build_interface(self):
        """Build and return the Gradio interface."""
        return build_gradio_interface(
            self.process_chat_message,
            self.process_gaia_questions,
            self.session_manager
        )

def main():
    """Main entry point for the application."""
    # Setup DI container
    setup_container()
    container = get_container()
    # Resolve use case and other dependencies
    process_message_use_case = container.resolve("process_message_use_case")
    # TODO: Refactor Gradio/chat logic to use process_message_use_case and other services from the container
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='AI Agent Application')
    parser.add_argument('--mode', choices=['cli', 'gradio'], default='gradio', help='Run mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and initialize the application
    app = AIAgentApp()
    
    try:
        if args.mode == 'cli':
            # Run CLI mode
            print("AI Agent CLI - Type 'quit' to exit")
            history = []
            
            while True:
                try:
                    user_input = input("\nEnter your message: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("Goodbye!")
                        break
                        
                    if not user_input:
                        continue
                    
                    response, history = app.process_chat_message(user_input, history)
                    print(f"\nResponse: {response}")
                    
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    logger.error(f"Error in CLI: {e}")
                    print(f"An error occurred: {e}")
        else:
            # Run Gradio interface
            interface = app.build_interface()
            interface.launch()
            
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 