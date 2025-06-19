"""
Integration Manager for AI Agent
Coordinates initialization and management of all integrations including
Supabase, LangChain, CrewAI, LlamaIndex, and GAIA components.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from typing import Optional, Dict, Any, List, Union, Tuple

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
    
    def __init__(self) -> None:
        self.config = integration_config
        self.components = {}
        self._initialized = False
        

    async def _get_safe_config_value(self, key: str) -> str:
        """Safely get configuration value with error handling"""
        try:
            parts = key.split('_')
            if len(parts) == 2:
                service, attr = parts
                config_obj = getattr(self.config, service, None)
                if config_obj:
                    return getattr(config_obj, attr, "")
            
            # Direct attribute access
            return getattr(self.config, key, "")
        except Exception as e:
            logger.error("Config access failed", extra={"key": key, "error": str(e)})
            return ""
    async def initialize_all(self) -> Any:
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
            if await await self._get_safe_config_value("supabase_is_configured_safe")():
                logger.info("Initializing Supabase components...")
                from .database_enhanced import initialize_supabase_enhanced
                self.components['supabase'] = await initialize_supabase_enhanced()
                logger.info("✅ Supabase components initialized")
            else:
                logger.info("⚠️ Supabase not configured, skipping")
            
            # 2. Initialize LlamaIndex
            if await self._get_safe_config_value("llamaindex_enable_hierarchical_indexing"):
                logger.info("Initializing LlamaIndex components...")
                from .llamaindex_enhanced import create_gaia_knowledge_base
                self.components['llamaindex'] = create_gaia_knowledge_base()
                logger.info("✅ LlamaIndex components initialized")
            else:
                logger.info("⚠️ LlamaIndex disabled, skipping")
            
            # 3. Initialize LangChain
            if await self._get_safe_config_value("langchain_enable_memory"):
                logger.info("Initializing LangChain components...")
                try:
                    from .langchain_enhanced import initialize_enhanced_agent
                    tools = self._get_available_tools()
                    self.components['langchain'] = initialize_enhanced_agent(tools)
                    logger.info("✅ LangChain components initialized")
                except ImportError as e:
                    logger.warning("⚠️ LangChain not available: {}", extra={"e": e})
            else:
                logger.info("⚠️ LangChain disabled, skipping")
            
            # 4. Initialize CrewAI
            if await self._get_safe_config_value("crewai_enable_multi_agent"):
                logger.info("Initializing CrewAI components...")
                try:
                    from .crew_enhanced import initialize_crew_enhanced
                    tools = self._get_available_tools()
                    self.components['crewai'] = initialize_crew_enhanced(tools)
                    logger.info("✅ CrewAI components initialized")
                except ImportError as e:
                    logger.warning("⚠️ CrewAI not available: {}", extra={"e": e})
            else:
                logger.info("⚠️ CrewAI disabled, skipping")
            
            self._initialized = True
            logger.info("🎉 All integrations initialized successfully")
            
            return self.components
            
        except Exception as e:
            logger.error("❌ Integration initialization failed: {}", extra={"e": e})
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
            logger.warning("Could not load basic tools: {}", extra={"e": e})
        
        # Add GAIA tools if enabled
        if await self._get_safe_config_value("gaia_enable_gaia_tools"):
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
                logger.warning("Could not load GAIA tools: {}", extra={"e": e})
        
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
    
    async def shutdown(self) -> Any:
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
                logger.error("❌ Error closing Supabase connections: {}", extra={"e": e})
        
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