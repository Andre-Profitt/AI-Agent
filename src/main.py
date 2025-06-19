"""
Main entry point for the AI Agent application with clean architecture.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.infrastructure.di.container import get_container, setup_container
from src.infrastructure.config.configuration_service import ConfigurationService
from src.infrastructure.logging.logging_service import LoggingService
from src.presentation.web.gradio_interface import GradioInterface
from src.presentation.cli.cli_interface import CLIInterface
from src.application.agents.agent_factory import AgentFactory
from src.application.tools.tool_factory import ToolFactory
from src.core.use_cases.process_message import ProcessMessageUseCase
from src.core.use_cases.manage_agent import ManageAgentUseCase
from src.core.use_cases.execute_tool import ExecuteToolUseCase
from src.core.use_cases.manage_session import ManageSessionUseCase
from src.shared.types import SystemConfig, Environment


class AIAgentApplication:
    """
    Main application class with clean architecture.
    
    This class orchestrates the entire application lifecycle,
    including dependency injection, service initialization,
    and interface management.
    """
    
    def __init__(self):
        self.container = None
        self.config: Optional[SystemConfig] = None
        self.logger: Optional[logging.Logger] = None
        self.web_interface: Optional[GradioInterface] = None
        self.cli_interface: Optional[CLIInterface] = None
        
    async def initialize(self) -> None:
        """Initialize the application and all its components."""
        try:
            # 0. Setup DI container
            setup_container()
            self.container = get_container()
            # 1. Initialize configuration
            await self._initialize_configuration()
            
            # 2. Initialize logging
            await self._initialize_logging()
            
            # 3. Register dependencies (handled by setup_container)
            
            # 4. Initialize services
            await self._initialize_services()
            
            # 5. Initialize interfaces
            await self._initialize_interfaces()
            
            self.logger.info("AI Agent application initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize application: {str(e)}")
            sys.exit(1)
    
    async def _initialize_configuration(self) -> None:
        """Initialize configuration service."""
        config_service = ConfigurationService()
        self.config = await config_service.load_configuration()
        
        # Register configuration in container
        self.container.register_instance("system_config", self.config)
        self.container.register_instance("agent_config", self.config.agent_config)
        self.container.register_instance("model_config", self.config.model_config)
        self.container.register_instance("logging_config", self.config.logging_config)
        self.container.register_instance("database_config", self.config.database_config)
    
    async def _initialize_logging(self) -> None:
        """Initialize logging service."""
        logging_service = LoggingService(self.config.logging_config)
        await logging_service.initialize()
        
        self.logger = logging.getLogger(__name__)
        self.container.register_instance("logging_service", logging_service)
    
    async def _initialize_services(self) -> None:
        """Initialize all application services with proper dependency injection."""
        
        # Resolve all repositories from container
        agent_repo = self.container.resolve("agent_repository")
        message_repo = self.container.resolve("message_repository")
        tool_repo = self.container.resolve("tool_repository")
        session_repo = self.container.resolve("session_repository")
        
        # Resolve executors and services
        agent_executor = self.container.resolve("agent_executor")
        tool_executor = self.container.resolve("tool_executor")
        logging_service = self.container.resolve("logging_service")
        
        # Initialize ProcessMessageUseCase with ALL dependencies
        process_message_use_case = ProcessMessageUseCase(
            agent_repository=agent_repo,  # FIX: Wire up agent repository
            message_repository=message_repo,
            agent_executor=agent_executor,
            logging_service=logging_service,
            config=self.config.agent_config
        )
        
        # Register the use case in container
        self.container.register_instance("process_message_use_case", process_message_use_case)
        
        # Initialize other use cases
        manage_agent_use_case = ManageAgentUseCase(
            agent_repository=agent_repo,
            logging_service=logging_service
        )
        self.container.register_instance("manage_agent_use_case", manage_agent_use_case)
        
        execute_tool_use_case = ExecuteToolUseCase(
            tool_repository=tool_repo,
            tool_executor=tool_executor,
            logging_service=logging_service
        )
        self.container.register_instance("execute_tool_use_case", execute_tool_use_case)
        
        manage_session_use_case = ManageSessionUseCase(
            session_repository=session_repo,
            message_repository=message_repo,
            logging_service=logging_service
        )
        self.container.register_instance("manage_session_use_case", manage_session_use_case)
        
        self.logger.info("All services initialized with proper dependency injection")
    
    async def _initialize_interfaces(self) -> None:
        """Initialize user interfaces."""
        # Initialize web interface
        self.web_interface = GradioInterface(
            process_message_use_case=self.container.resolve("process_message_use_case"),
            config=self.config
        )
        
        # Initialize CLI interface
        self.cli_interface = CLIInterface(
            process_message_use_case=self.container.resolve("process_message_use_case"),
            config=self.config
        )
        
        self.logger.info("User interfaces initialized successfully")
    
    async def run_web(self) -> None:
        """Run the web interface."""
        if not self.web_interface:
            raise RuntimeError("Web interface not initialized")
        
        self.logger.info(f"Starting web interface on {self.config.api_host}:{self.config.api_port}")
        await self.web_interface.run()
    
    async def run_cli(self) -> None:
        """Run the CLI interface."""
        if not self.cli_interface:
            raise RuntimeError("CLI interface not initialized")
        
        self.logger.info("Starting CLI interface")
        await self.cli_interface.run()
    
    async def shutdown(self) -> None:
        """Shutdown the application gracefully."""
        self.logger.info("Shutting down AI Agent application")
        
        # Cleanup resources
        if self.web_interface:
            await self.web_interface.shutdown()
        
        if self.cli_interface:
            await self.cli_interface.shutdown()
        
        # Clear container
        self.container.clear()
        
        self.logger.info("Application shutdown complete")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Agent Application")
    parser.add_argument("--mode", choices=["web", "cli"], default="web", help="Run mode")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Create and initialize application
    app = AIAgentApplication()
    await app.initialize()
    
    try:
        # Run appropriate interface
        if args.mode == "web":
            await app.run_web()
        elif args.mode == "cli":
            await app.run_cli()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main()) 