from agent import tools
from examples.parallel_execution_example import tool_name
from migrations.env import config

from src.application.tools.tool_factory import advanced_tools
from src.application.tools.tool_factory import default_tools
from src.application.tools.tool_factory import tool_class
from src.application.tools.tool_factory import tool_config
from src.core.optimized_chain_of_thought import configs
from src.database.models import tool
from src.database.models import tool_id
from src.database_extended import tool_names
from src.templates.template_factory import factory
from src.tools_enhanced import FileReaderTool
from src.tools_introspection import name

from src.tools.base_tool import Tool
from src.agents.advanced_hybrid_architecture import SemanticSearchTool
from src.agents.advanced_hybrid_architecture import WeatherTool
from src.gaia_components.adaptive_tool_system import Tool
from src.utils.base_tool import CalculatorTool
from src.utils.base_tool import WebSearchTool
# TODO: Fix undefined variables: Any, Dict, List, Optional, advanced_tools, config, configs, default_tools, e, logging, name, tool_class, tool_config, tool_id, tool_name, tool_names, tools, uuid4
import factory

from src.tools.base_tool import tool
from src.tools.pythonrepltool import PythonReplTool
from src.tools_enhanced import FileReaderTool

# TODO: Fix undefined variables: FileReaderTool, PythonReplTool, advanced_tools, config, configs, default_tools, e, factory, name, self, tool, tool_class, tool_config, tool_id, tool_name, tool_names, tools

"""

from langchain.tools import Tool
Tool Factory for creating different types of tools.
"""

from typing import Optional
from typing import Any
from typing import List

import logging
from typing import Dict, Any, Optional, List
from uuid import uuid4

from src.utils.tools_enhanced import (
    WebSearchTool,
    CalculatorTool,
    FileReaderTool,
    PythonReplTool,
    WeatherTool,
    SemanticSearchTool
)

class ToolFactory:
    """Factory for creating different types of tools"""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        # Registry of tool implementations
        self._tool_registry: Dict[str, type] = {
            "web_search": WebSearchTool,
            "calculator": CalculatorTool,
            "file_reader": FileReaderTool,
            "python_repl": PythonReplTool,
            "weather": WeatherTool,
            "semantic_search": SemanticSearchTool
        }

        # Cache for created tools
        self._tool_cache: Dict[str, Tool] = {}

    def create_tool(self, tool_name: str, config: Optional[Dict[str, Any]] = None) -> Optional[Tool]:
        """Create a tool of the specified type"""

        config = config or {}
        tool_id = f"{tool_name}_{uuid4().hex[:8]}"

        self.logger.debug("Creating tool: {}", extra={"tool_name": tool_name})

        try:
            if tool_name not in self._tool_registry:
                self.logger.warning("Unknown tool type: {}", extra={"tool_name": tool_name})
                return None

            tool_class = self._tool_registry[tool_name]

            # Create tool instance
            if tool_name == "web_search":
                tool = tool_class(
                    api_key=config.get("api_key"),
                    search_engine=config.get("search_engine", "tavily")
                )
            elif tool_name == "calculator":
                tool = tool_class()
            elif tool_name == "file_reader":
                tool = tool_class(
                    supported_formats=config.get("supported_formats", ["txt", "pdf", "docx"])
                )
            elif tool_name == "python_repl":
                tool = tool_class(
                    timeout=config.get("timeout", 30),
                    max_output_length=config.get("max_output_length", 1000)
                )
            elif tool_name == "weather":
                tool = tool_class(
                    api_key=config.get("api_key")
                )
            elif tool_name == "semantic_search":
                tool = tool_class(
                    vector_store=config.get("vector_store"),
                    embedding_model=config.get("embedding_model", "text-embedding-ada-002")
                )
            else:
                tool = tool_class()

            # Set tool properties
            tool.id = tool_id
            tool.name = tool_name
            tool.is_available = True

            # Cache the tool
            self._tool_cache[tool_id] = tool

            self.logger.debug("Successfully created tool: {} (id: {})", extra={"tool_name": tool_name, "tool_id": tool_id})
            return tool

        except Exception as e:
            self.logger.error("Failed to create tool {}: {}", extra={"tool_name": tool_name, "e": e})
            return None

    def get_tool(self, tool_id: str) -> Optional[Tool]:
        """Get cached tool by ID"""
        return self._tool_cache.get(tool_id)

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all cached tools"""
        tools = []
        for tool_id, tool in self._tool_cache.items():
            tools.append({
                "id": tool_id,
                "name": tool.name,
                "type": tool.tool_type.value if hasattr(tool, 'tool_type') else "unknown",
                "is_available": tool.is_available
            })
        return tools

    def get_available_tool_types(self) -> List[str]:
        """Get list of available tool types"""
        return list(self._tool_registry.keys())

    def register_tool_type(self, name: str, tool_class: type) -> None:
        """Register a new tool type"""
        self._tool_registry[name] = tool_class
        self.logger.info("Registered new tool type: {}", extra={"name": name})

    def create_tool_set(self, tool_names: List[str], configs: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Tool]:
        """Create multiple tools at once"""
        tools = []
        configs = configs or {}

        for tool_name in tool_names:
            tool_config = configs.get(tool_name, {})
            tool = self.create_tool(tool_name, tool_config)
            if tool:
                tools.append(tool)

        return tools

    def clear_cache(self) -> None:
        """Clear the tool cache"""
        self._tool_cache.clear()
        self.logger.debug("Tool cache cleared")

# Example usage function
def create_default_tool_set() -> List[Tool]:
    """Create a default set of tools"""
    factory = ToolFactory()

    default_tools = [
        "web_search",
        "calculator",
        "file_reader",
        "python_repl"
    ]

    return factory.create_tool_set(default_tools)

def create_advanced_tool_set() -> List[Tool]:
    """Create an advanced set of tools"""
    factory = ToolFactory()

    advanced_tools = [
        "web_search",
        "calculator",
        "file_reader",
        "python_repl",
        "weather",
        "semantic_search"
    ]

    configs = {
        "web_search": {"search_engine": "tavily"},
        "python_repl": {"timeout": 60, "max_output_length": 2000},
        "semantic_search": {"embedding_model": "text-embedding-ada-002"}
    }

    return factory.create_tool_set(advanced_tools, configs)