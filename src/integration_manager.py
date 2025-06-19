"""
Integration Manager for AI Agent
Coordinates initialization and management of all integrations including
Supabase, LangChain, CrewAI, LlamaIndex, and GAIA components.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio

try:
    from .config.integrations import integration_config
except ImportError:
    try:
        from config.integrations import integration_config
    except ImportError:
        # Fallback for when running as standalone script
        integration_config = None
        logging.warning("Could not import integration_config - using defaults")

logger = logging.getLogger(__name__)

class IntegrationManager:
    """Manages initialization and coordination of all integrations"""
    
    def __init__(self):
        self.config = integration_config
        self.components = {}
        self._initialized = False
        
    async def initialize_all(self):
        """Initialize all components in correct order"""
        
        if self._initialized:
            logger.info("Integration manager already initialized")
            return self.components
        
        # Validate config first
        is_valid, issues = self.config.validate()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {issues}")
        
        logger.info("Starting integration initialization...")
        
        try:
            # 1. Initialize Supabase if configured
            if self.config.supabase.is_configured():
                logger.info("Initializing Supabase components...")
                from .database_enhanced import initialize_supabase_enhanced
                self.components['supabase'] = await initialize_supabase_enhanced()
                logger.info("✅ Supabase components initialized")
            else:
                logger.info("⚠️ Supabase not configured, skipping")
            
            # 2. Initialize LlamaIndex
            if self.config.llamaindex.enable_hierarchical_indexing:
                logger.info("Initializing LlamaIndex components...")
                from .llamaindex_enhanced import create_gaia_knowledge_base
                self.components['llamaindex'] = create_gaia_knowledge_base()
                logger.info("✅ LlamaIndex components initialized")
            else:
                logger.info("⚠️ LlamaIndex disabled, skipping")
            
            # 3. Initialize LangChain
            if self.config.langchain.enable_memory:
                logger.info("Initializing LangChain components...")
                try:
                    from .langchain_enhanced import initialize_enhanced_agent
                    tools = self._get_available_tools()
                    self.components['langchain'] = initialize_enhanced_agent(tools)
                    logger.info("✅ LangChain components initialized")
                except ImportError as e:
                    logger.warning(f"⚠️ LangChain not available: {e}")
            else:
                logger.info("⚠️ LangChain disabled, skipping")
            
            # 4. Initialize CrewAI
            if self.config.crewai.enable_multi_agent:
                logger.info("Initializing CrewAI components...")
                try:
                    from .crew_enhanced import initialize_crew_enhanced
                    tools = self._get_available_tools()
                    self.components['crewai'] = initialize_crew_enhanced(tools)
                    logger.info("✅ CrewAI components initialized")
                except ImportError as e:
                    logger.warning(f"⚠️ CrewAI not available: {e}")
            else:
                logger.info("⚠️ CrewAI disabled, skipping")
            
            self._initialized = True
            logger.info("🎉 All integrations initialized successfully")
            
            return self.components
            
        except Exception as e:
            logger.error(f"❌ Integration initialization failed: {e}")
            raise
    
    def _get_available_tools(self) -> List[Any]:
        """Get list of available tools for agents"""
        tools = []
        
        # Add basic tools
        try:
            from .tools import (
                SemanticSearchTool, PythonInterpreterTool, 
                FileReaderTool, WeatherTool
            )
            tools.extend([
                SemanticSearchTool(),
                PythonInterpreterTool(),
                FileReaderTool(),
                WeatherTool()
            ])
        except ImportError as e:
            logger.warning(f"Could not load basic tools: {e}")
        
        # Add GAIA tools if enabled
        if self.config.gaia.enable_gaia_tools:
            try:
                from .gaia.tools.gaia_specialized import (
                    GAIAVideoAnalyzer, GAIAChessLogicTool, 
                    GAIAImageAnalyzer, GAIAMusicAnalyzer
                )
                tools.extend([
                    GAIAVideoAnalyzer(),
                    GAIAChessLogicTool(),
                    GAIAImageAnalyzer(),
                    GAIAMusicAnalyzer()
                ])
            except ImportError as e:
                logger.warning(f"Could not load GAIA tools: {e}")
        
        return tools
    
    def get_component(self, name: str) -> Optional[Any]:
        """Get specific component by name"""
        return self.components.get(name)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all components"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "initialized": self._initialized,
            "config_valid": self.config.validate()[0],
            "components": {}
        }
        
        for name, component in self.components.items():
            status["components"][name] = {
                "available": component is not None,
                "type": type(component).__name__ if component else None
            }
        
        return status
    
    async def shutdown(self):
        """Gracefully shutdown all components"""
        logger.info("Shutting down integrations...")
        
        # Close Supabase connections
        if 'supabase' in self.components:
            try:
                connection_pool = self.components['supabase'].get('connection_pool')
                if connection_pool:
                    await connection_pool.close()
                logger.info("✅ Supabase connections closed")
            except Exception as e:
                logger.error(f"❌ Error closing Supabase connections: {e}")
        
        # Clear components
        self.components.clear()
        self._initialized = False
        logger.info("✅ Integration manager shutdown complete")

# Global integration manager instance
integration_manager = IntegrationManager()

async def get_integration_manager() -> IntegrationManager:
    """Get or initialize the global integration manager"""
    if not integration_manager._initialized:
        await integration_manager.initialize_all()
    return integration_manager 