#!/usr/bin/env python3
"""
AI Agent Application - Production Ready for HuggingFace Spaces
Fixed import structure and enhanced error handling
"""

import sys
import os
from pathlib import Path

# Fix Python path BEFORE any imports
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Standard library imports
import asyncio
import uuid
import logging
from typing import List, Dict, Any, Optional
import signal
from contextlib import asynccontextmanager

# Third-party imports
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
if os.path.exists('.env'):
    load_dotenv()

# Local imports - using absolute imports
from src.config.settings import Settings
from src.config.integrations import IntegrationConfig
from src.core.monitoring import MetricsCollector
from src.core.health_check import HealthChecker
from src.services.integration_hub import IntegrationHub
from src.database.supabase_manager import SupabaseManager
from src.utils.logging import setup_logging, get_logger
from src.tools.registry import ToolRegistry

# Initialize logging
setup_logging()
logger = get_logger(__name__)

class AIAgentApp:
    """Main application class with all fixes implemented"""
    
    def __init__(self):
        self.settings = Settings()
        self.metrics = MetricsCollector()
        self.health_checker = HealthChecker()
        self.integration_hub = None
        self.db_manager = None
        self.tool_registry = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize all components with proper error handling"""
        try:
            logger.info("Starting application initialization...")
            
            # 1. Setup environment
            await self._setup_environment()
            
            # 2. Initialize database
            await self._initialize_database()
            
            # 3. Initialize tools
            await self._initialize_tools()
            
            # 4. Initialize integration hub
            await self._initialize_integration_hub()
            
            # 5. Start health monitoring
            await self._start_monitoring()
            
            self.initialized = True
            logger.info("Application initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}", exc_info=True)
            await self.shutdown()
            raise
    
    async def _setup_environment(self):
        """Setup environment with validation"""
        logger.info("Setting up environment...")
        
        # Validate required environment variables
        required_vars = ['SUPABASE_URL', 'SUPABASE_KEY', 'GROQ_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning(f"Missing environment variables: {missing_vars}")
            # Don't fail completely, allow graceful degradation
        
        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.shutdown())
    
    async def _initialize_database(self):
        """Initialize database with connection pooling"""
        logger.info("Initializing database connection...")
        
        try:
            self.db_manager = SupabaseManager(
                url=os.getenv('SUPABASE_URL', ''),
                key=os.getenv('SUPABASE_KEY', ''),
                pool_size=10,
                max_retries=3
            )
            
            await self.db_manager.initialize()
            await self.db_manager.test_connection()
            
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            # Continue without database
            self.db_manager = None
    
    async def _initialize_tools(self):
        """Initialize tool registry with all tools"""
        logger.info("Initializing tools...")
        
        self.tool_registry = ToolRegistry()
        
        # Register all tools
        from src.tools.implementations import (
            file_reader, web_researcher, python_interpreter,
            audio_transcriber, video_analyzer, image_analyzer
        )
        
        tools = [
            file_reader, web_researcher, python_interpreter,
            audio_transcriber, video_analyzer, image_analyzer
        ]
        
        for tool in tools:
            self.tool_registry.register(tool)
        
        logger.info(f"Registered {len(tools)} tools")
    
    async def _initialize_integration_hub(self):
        """Initialize integration hub with circuit breakers"""
        logger.info("Initializing integration hub...")
        
        self.integration_hub = IntegrationHub(
            db_manager=self.db_manager,
            tool_registry=self.tool_registry,
            metrics_collector=self.metrics
        )
        
        await self.integration_hub.initialize()
        
        logger.info("Integration hub initialized")
    
    async def _start_monitoring(self):
        """Start monitoring and health checks"""
        logger.info("Starting monitoring services...")
        
        # Start metrics collection
        await self.metrics.start()
        
        # Start health check endpoint
        await self.health_checker.start(
            db_manager=self.db_manager,
            integration_hub=self.integration_hub
        )
        
        logger.info("Monitoring services started")
    
    async def shutdown(self):
        """Graceful shutdown with proper cleanup"""
        logger.info("Starting graceful shutdown...")
        
        try:
            # Stop monitoring
            if self.health_checker:
                await self.health_checker.stop()
            
            if self.metrics:
                await self.metrics.stop()
            
            # Close integration hub
            if self.integration_hub:
                await self.integration_hub.shutdown()
            
            # Close database connections
            if self.db_manager:
                await self.db_manager.close()
            
            logger.info("Shutdown completed successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
    
    def create_interface(self):
        """Create Gradio interface with error handling"""
        if not self.initialized:
            raise RuntimeError("Application not initialized")
        
        with gr.Blocks(title="AI Agent - Production Ready") as interface:
            gr.Markdown("# 🤖 AI Agent System")
            
            with gr.Tabs():
                # Chat Interface
                with gr.TabItem("💬 Chat"):
                    chatbot = gr.Chatbot(height=600)
                    msg = gr.Textbox(label="Message", placeholder="Ask me anything...")
                    clear = gr.Button("Clear")
                    
                    async def process_message(message, history):
                        try:
                            # Track metrics
                            self.metrics.track_request("chat")
                            
                            # Process through integration hub
                            response = await self.integration_hub.process_message(
                                message=message,
                                history=history
                            )
                            
                            # Update history
                            history = history or []
                            history.append([message, response])
                            
                            return "", history
                            
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            self.metrics.track_error("chat", str(e))
                            error_msg = "I encountered an error. Please try again."
                            history.append([message, error_msg])
                            return "", history
                    
                    msg.submit(process_message, [msg, chatbot], [msg, chatbot])
                    clear.click(lambda: None, None, chatbot)
                
                # Health Status
                with gr.TabItem("🏥 Health"):
                    health_status = gr.JSON(label="System Health")
                    refresh_btn = gr.Button("Refresh")
                    
                    async def get_health():
                        return await self.health_checker.get_status()
                    
                    refresh_btn.click(get_health, outputs=health_status)
                
                # Metrics Dashboard
                with gr.TabItem("📊 Metrics"):
                    metrics_display = gr.JSON(label="System Metrics")
                    metrics_refresh = gr.Button("Refresh")
                    
                    async def get_metrics():
                        return self.metrics.get_all_metrics()
                    
                    metrics_refresh.click(get_metrics, outputs=metrics_display)
        
        return interface

# Application entry point
async def main():
    """Main entry point with proper lifecycle management"""
    app = AIAgentApp()
    
    try:
        # Initialize application
        await app.initialize()
        
        # Create and launch interface
        interface = app.create_interface()
        
        # Launch Gradio (blocking)
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )
        
    except Exception as e:
        logger.error(f"Application failed: {e}", exc_info=True)
        raise
    finally:
        await app.shutdown()

if __name__ == "__main__":
    # Run the application
    asyncio.run(main()) 