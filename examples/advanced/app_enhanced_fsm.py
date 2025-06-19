#!/usr/bin/env python3
"""
AI Agent Application - Enhanced FSM Integration
This file includes the Enhanced FSM integration with backward compatibility.
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
from src.config import config, Environment
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

# Import the Enhanced FSM agent and other components
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
    
    # Import the Enhanced FSM Agent
    from src.migrated_enhanced_fsm_agent import MigratedEnhancedFSMAgent
    
except ImportError as e:
    import sys
    import os
    logger.info("Import Error: {}", extra={"e": e})
    logger.info("Current working directory: {}", extra={"os_getcwd__": os.getcwd()})
    logger.info("Python path: {}", extra={"sys_path": sys.path})
    logger.info("Contents of src directory: {}", extra={"os_listdir__src___if_os_path_exists__src___else__src_directory_not_found_": os.listdir('src') if os.path.exists('src') else 'src directory not found'})
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

class EnhancedAIAgentApp:
    """Enhanced AI Agent application with Enhanced FSM integration."""
    
    def __init__(self, use_enhanced_fsm: bool = True):
        """Initialize with Enhanced FSM support."""
        self.agent = None
        self.enhanced_agent = None
        self.tools = []
        self.session_manager = None
        self.log_handler = None
        self.model_name = None
        self.integration_hub = None
        self.use_enhanced_fsm = use_enhanced_fsm
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
                
                # Initialize agents with validated configuration
                if self.use_enhanced_fsm:
                    self.enhanced_agent = self._initialize_enhanced_agent()
                    logger.info("Enhanced FSM Agent initialized successfully")
                
                # Always initialize the original agent for fallback
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
    
    def _initialize_enhanced_agent(self):
        """Initialize the Enhanced FSM Agent"""
        try:
            return MigratedEnhancedFSMAgent(
                tools=self.tools,
                enable_hierarchical=True,
                enable_probabilistic=True,
                enable_discovery=True,
                enable_metrics=True,
                fsm_name="EnhancedAIAgent"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced FSM agent: {e}")
            raise
    
    def _initialize_agent(self):
        """Initialize the original FSM agent for fallback"""
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
        self.enhanced_agent = None
        self.session_manager = SessionManager(max_sessions=5)
        
    def process_gaia_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process GAIA benchmark questions."""
        return GAIAEvaluator.process_questions(self.agent, questions)
        
    def process_chat_message(self, message: str, history: list, log_to_db: bool = True, session_id: str = None, use_enhanced: bool = None) -> tuple:
        """Process chat message with Enhanced FSM support."""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Determine which agent to use
        if use_enhanced is None:
            use_enhanced = self.use_enhanced_fsm and self.enhanced_agent is not None
        
        try:
            # Validate input
            validation_result = validate_user_prompt(message)
            if not validation_result.is_valid:
                return history + [[message, f"Invalid input: {validation_result.error_message}"]], session_id
            
            # Process with appropriate agent
            if use_enhanced and self.enhanced_agent:
                logger.info("Using Enhanced FSM Agent")
                response = self.enhanced_agent.run(message)
                final_answer = response.get('result', 'No result generated')
                
                # Log enhanced metrics if available
                if response.get('metrics'):
                    logger.info(f"Enhanced FSM Metrics: {response['metrics']['fsm_name']}")
                    logger.info(f"Final State: {response['final_state']}")
                    logger.info(f"Execution Time: {response['execution_time']:.3f}s")
                
            else:
                logger.info("Using Original FSM Agent")
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
    
    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get metrics from the Enhanced FSM Agent"""
        if self.enhanced_agent:
            return self.enhanced_agent.get_metrics()
        return {"enhanced_agent_not_available": True}
    
    def get_discovery_statistics(self) -> Dict[str, Any]:
        """Get discovery statistics from the Enhanced FSM Agent"""
        if self.enhanced_agent:
            return self.enhanced_agent.get_discovery_statistics()
        return {"enhanced_agent_not_available": True}
    
    def visualize_enhanced_fsm(self) -> str:
        """Get visualization of the Enhanced FSM"""
        if self.enhanced_agent:
            return self.enhanced_agent.visualize_current_state()
        return "Enhanced FSM Agent not available"
    
    def save_enhanced_visualization(self, filename: str):
        """Save visualization of the Enhanced FSM"""
        if self.enhanced_agent:
            self.enhanced_agent.save_visualization(filename)
        else:
            raise ValueError("Enhanced FSM Agent not available")
        
    def build_interface(self):
        """Build the Gradio interface with Enhanced FSM features."""
        return build_enhanced_gradio_interface(
            self.process_chat_message,
            self.process_gaia_questions,
            self.session_manager,
            self.get_enhanced_metrics,
            self.get_discovery_statistics,
            self.visualize_enhanced_fsm,
            self.save_enhanced_visualization
        )

def build_enhanced_gradio_interface(
    process_chat_message, 
    process_gaia_questions, 
    session_manager,
    get_enhanced_metrics,
    get_discovery_statistics,
    visualize_enhanced_fsm,
    save_enhanced_visualization
):
    """Build the enhanced Gradio interface with FSM features."""
    
    # Create custom CSS
    custom_css = create_custom_css()
    
    with gr.Blocks(css=custom_css, title="Enhanced AI Agent System") as interface:
        gr.Markdown("# 🤖 Enhanced AI Agent System with FSM")
        
        with gr.Tabs():
            # Main Chat Tab
            with gr.Tab("💬 Enhanced Chat"):
                create_enhanced_chat_interface(process_chat_message, session_manager)
            
            # FSM Analytics Tab
            with gr.Tab("🔄 FSM Analytics"):
                create_fsm_analytics_tab(
                    get_enhanced_metrics,
                    get_discovery_statistics,
                    visualize_enhanced_fsm,
                    save_enhanced_visualization
                )
            
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
        gr.Markdown("Built with ❤️ using Gradio, LangChain, and Enhanced FSM")
    
    return interface

def create_enhanced_chat_interface(process_chat_message, session_manager):
    """Create enhanced chat interface with FSM options."""
    
    with gr.Row():
        with gr.Column(scale=3):
            # Chat interface
            chatbot = gr.Chatbot(height=600, show_label=False)
            msg = gr.Textbox(
                placeholder="Ask me anything...",
                show_label=False,
                lines=2
            )
            
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")
                
            # FSM options
            with gr.Row():
                use_enhanced_fsm = gr.Checkbox(
                    label="Use Enhanced FSM", 
                    value=True,
                    info="Enable Enhanced FSM with hierarchical states and probabilistic transitions"
                )
        
        with gr.Column(scale=1):
            # FSM status
            gr.Markdown("### 🔄 FSM Status")
            fsm_status = gr.Textbox(
                label="Current FSM State",
                value="Ready",
                interactive=False
            )
            
            # Quick metrics
            gr.Markdown("### 📊 Quick Metrics")
            metrics_display = gr.JSON(
                label="FSM Metrics",
                value={},
                interactive=False
            )
    
    # Event handlers
    def process_message_with_fsm(message, history, use_enhanced):
        return process_chat_message(message, history, use_enhanced=use_enhanced)
    
    submit_btn.click(
        process_message_with_fsm,
        inputs=[msg, chatbot, use_enhanced_fsm],
        outputs=[chatbot, msg]
    )
    
    msg.submit(
        process_message_with_fsm,
        inputs=[msg, chatbot, use_enhanced_fsm],
        outputs=[chatbot, msg]
    )
    
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

def create_fsm_analytics_tab(
    get_enhanced_metrics,
    get_discovery_statistics,
    visualize_enhanced_fsm,
    save_enhanced_visualization
):
    """Create FSM analytics tab."""
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📊 FSM Metrics")
            metrics_json = gr.JSON(label="Comprehensive Metrics")
            refresh_metrics_btn = gr.Button("Refresh Metrics")
            
            gr.Markdown("### 🔍 Discovery Statistics")
            discovery_json = gr.JSON(label="Pattern Discovery")
            refresh_discovery_btn = gr.Button("Refresh Discovery")
        
        with gr.Column(scale=2):
            gr.Markdown("### 🎨 FSM Visualization")
            fsm_visualization = gr.Textbox(
                label="Current FSM State",
                lines=10,
                interactive=False
            )
            refresh_viz_btn = gr.Button("Refresh Visualization")
            
            gr.Markdown("### 💾 Save Visualization")
            save_filename = gr.Textbox(
                label="Filename",
                value="fsm_visualization.png",
                placeholder="Enter filename for visualization"
            )
            save_viz_btn = gr.Button("Save Visualization")
    
    # Event handlers
    def refresh_metrics():
        return get_enhanced_metrics()
    
    def refresh_discovery():
        return get_discovery_statistics()
    
    def refresh_visualization():
        return visualize_enhanced_fsm()
    
    def save_visualization(filename):
        try:
            save_enhanced_visualization(filename)
            return f"Visualization saved to {filename}"
        except Exception as e:
            return f"Error saving visualization: {e}"
    
    refresh_metrics_btn.click(refresh_metrics, outputs=[metrics_json])
    refresh_discovery_btn.click(refresh_discovery, outputs=[discovery_json])
    refresh_viz_btn.click(refresh_visualization, outputs=[fsm_visualization])
    save_viz_btn.click(save_visualization, inputs=[save_filename], outputs=[gr.Textbox()])

def main():
    """Main entry point."""
    try:
        # Initialize the enhanced application
        app = EnhancedAIAgentApp(use_enhanced_fsm=True)
        
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
        logger.info("Failed to start application: {}", extra={"e": e})
        sys.exit(1)
    finally:
        # Cleanup integration hub
        try:
            if app.integration_hub:
                import asyncio
                asyncio.run(cleanup_integrations())
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

if __name__ == "__main__":
    main() 