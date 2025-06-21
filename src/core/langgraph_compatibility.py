from agent import messages
from agent import tools
from app import error_msg
from app import msg
from examples.enhanced_unified_example import tasks
from examples.parallel_execution_example import results
from examples.parallel_execution_example import tool_calls
from examples.parallel_execution_example import tool_name
from migrations.env import config

from src.core.langchain_enhanced import tool_call
from src.core.langgraph_compatibility import ai_message
from src.core.langgraph_compatibility import formatted_calls
from src.core.langgraph_compatibility import loop
from src.core.langgraph_compatibility import missing
from src.core.langgraph_compatibility import required_imports
from src.core.langgraph_compatibility import tool_args
from src.core.langgraph_compatibility import version
from src.database.models import tool
from src.database.models import tool_input
from src.gaia_components.advanced_reasoning_engine import AIMessage
from src.tools.registry import module
from src.utils.error_category import call

from src.tools.base_tool import Tool

from src.tools.base_tool import BaseTool

from src.tools.base_tool import ToolMessage
# TODO: Fix undefined variables: Any, Dict, LangGraphToolExecutor, List, Optional, ToolMessage, ToolNode, ai_message, attr, call, config, e, error_msg, formatted_calls, langgraph, logging, loop, messages, missing, module, msg, required_imports, result, results, tasks, tool_args, tool_call, tool_calls, tool_calls_list, tool_input, tool_name, tools, version
from src.tools.base_tool import tool


"""
from typing import List
from src.shared.types.di_types import BaseTool
# TODO: Fix undefined variables: AIMessage, LangGraphToolExecutor, ToolMessage, ToolNode, ai_message, attr, call, config, e, error_msg, formatted_calls, langgraph, loop, messages, missing, module, msg, required_imports, result, results, self, tasks, tool, tool_args, tool_call, tool_calls, tool_calls_list, tool_input, tool_name, tools, version

from langchain.schema import AIMessage
from langchain.tools import BaseTool
from unittest.mock import call
LangGraph Compatibility Layer
Handles version differences and missing imports in langgraph
"""

from typing import Optional
from typing import Dict
from typing import Any

import logging

from langgraph.prebuilt import ToolNode
from langchain_core.tools import BaseTool
from langchain_core.messages import ToolMessage, AIMessage

logger = logging.getLogger(__name__)

class ToolExecutor:
    """
    Compatibility implementation of ToolExecutor
    Provides a consistent interface regardless of langgraph version
    """

    def __init__(self, tools: List[BaseTool]):
        """
        Initialize ToolExecutor with a list of tools

        Args:
            tools: List of LangChain tools
        """
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

        # Try to use ToolNode if available (newer versions)
        try:
            self.tool_node = ToolNode(tools)
            self.use_tool_node = True
            logger.info("Using ToolNode from langgraph.prebuilt")
        except Exception as e:
            self.tool_node = None
            self.use_tool_node = False
            logger.info("ToolNode not available, using direct tool execution")

    async def ainvoke(self, tool_calls: List[Dict[str, Any]], config: Optional[Dict] = None) -> List[ToolMessage]:
        """
        Execute tools asynchronously

        Args:
            tool_calls: List of tool call dictionaries with 'name' and 'args'
            config: Optional configuration

        Returns:
            List of ToolMessage results
        """
        if self.use_tool_node:
            # Use ToolNode if available
            try:
                # Create a proper message format for ToolNode
                from langchain_core.messages import AIMessage

                # Convert tool_calls to the format ToolNode expects
                formatted_calls = []
                for call in tool_calls:
                    formatted_calls.append({
                        "id": call.get("id", f"call_{len(formatted_calls)}"),
                        "name": call["name"],
                        "args": call.get("args", {})
                    })

                # Create an AIMessage with tool_calls
                ai_message = AIMessage(
                    content="",
                    tool_calls=formatted_calls
                )

                # Invoke ToolNode
                result = await self.tool_node.ainvoke({"messages": [ai_message]}, config)

                # Extract ToolMessages from result
                if isinstance(result, dict) and "messages" in result:
                    return [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
                elif isinstance(result, list):
                    return [msg for msg in result if isinstance(msg, ToolMessage)]
                else:
                    return []

            except Exception as e:
                logger.warning(f"ToolNode execution failed, falling back to direct execution: {e}")
                self.use_tool_node = False

        # Fallback to direct tool execution
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})

            if tool_name not in self.tool_map:
                error_msg = f"Tool '{tool_name}' not found"
                logger.error(error_msg)
                results.append(ToolMessage(
                    content=error_msg,
                    name=tool_name,
                    tool_call_id=tool_call.get("id", f"error_{tool_name}")
                ))
                continue

            try:
                # Execute tool
                tool = self.tool_map[tool_name]

                # Handle both sync and async tools
                if hasattr(tool, 'ainvoke'):
                    result = await tool.ainvoke(tool_args)
                elif hasattr(tool, 'invoke'):
                    result = tool.invoke(tool_args)
                elif callable(tool):
                    result = tool(**tool_args)
                else:
                    result = str(tool)

                # Create ToolMessage
                results.append(ToolMessage(
                    content=str(result),
                    name=tool_name,
                    tool_call_id=tool_call.get("id", f"call_{tool_name}")
                ))

            except Exception as e:
                error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                logger.error(error_msg)
                results.append(ToolMessage(
                    content=error_msg,
                    name=tool_name,
                    tool_call_id=tool_call.get("id", f"error_{tool_name}")
                ))

        return results

    def invoke(self, tool_calls: List[Dict[str, Any]], config: Optional[Dict] = None) -> List[ToolMessage]:
        """
        Execute tools synchronously

        Args:
            tool_calls: List of tool call dictionaries
            config: Optional configuration

        Returns:
            List of ToolMessage results
        """
        import asyncio

        # Run async version in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.ainvoke(tool_calls, config))
        finally:
            loop.close()

    def batch(self, tool_calls_list: List[List[Dict[str, Any]]], config: Optional[Dict] = None) -> List[List[ToolMessage]]:
        """
        Execute multiple batches of tool calls

        Args:
            tool_calls_list: List of tool call batches
            config: Optional configuration

        Returns:
            List of results for each batch
        """
        results = []
        for tool_calls in tool_calls_list:
            results.append(self.invoke(tool_calls, config))
        return results

    async def abatch(self, tool_calls_list: List[List[Dict[str, Any]]], config: Optional[Dict] = None) -> List[List[ToolMessage]]:
        """
        Execute multiple batches of tool calls asynchronously

        Args:
            tool_calls_list: List of tool call batches
            config: Optional configuration

        Returns:
            List of results for each batch
        """

        tasks = [self.ainvoke(tool_calls, config) for tool_calls in tool_calls_list]
        return await asyncio.gather(*tasks)

# Compatibility imports - try different import paths
def get_tool_executor_class():
    """Get the appropriate ToolExecutor class based on available imports"""

    # First, try to import from langgraph.prebuilt
    try:
        from langgraph.prebuilt import ToolExecutor as LangGraphToolExecutor
        logger.info("Using ToolExecutor from langgraph.prebuilt")
        return LangGraphToolExecutor
    except ImportError:
        pass

    # Try alternative import paths
    try:
        from langgraph.tools import ToolExecutor as LangGraphToolExecutor
        logger.info("Using ToolExecutor from langgraph.tools")
        return LangGraphToolExecutor
    except ImportError:
        pass

    # Try older versions
    try:
        from langgraph.executor import ToolExecutor as LangGraphToolExecutor
        logger.info("Using ToolExecutor from langgraph.executor")
        return LangGraphToolExecutor
    except ImportError:
        pass

    # Use our compatibility implementation
    logger.info("Using compatibility ToolExecutor implementation")
    return ToolExecutor

# Export the appropriate class
ToolExecutor = get_tool_executor_class()

# Additional compatibility helpers
class ToolExecutorAdapter:
    """
    Adapter to ensure consistent interface across different ToolExecutor implementations
    """

    def __init__(self, tools: List[BaseTool]):
        self.executor = ToolExecutor(tools)
        self.tools = tools

    async def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """
        Simple execute interface

        Args:
            tool_name: Name of the tool to execute
            tool_input: Input arguments for the tool

        Returns:
            Tool execution result
        """
        tool_calls = [{
            "name": tool_name,
            "args": tool_input,
            "id": f"call_{tool_name}_{id(tool_input)}"
        }]

        if hasattr(self.executor, 'ainvoke'):
            messages = await self.executor.ainvoke(tool_calls)
        else:
            # Fallback for older versions
            messages = await self._execute_directly(tool_name, tool_input)

        if messages and len(messages) > 0:
            return messages[0].content
        return None

    async def _execute_directly(self, tool_name: str, tool_input: Dict[str, Any]) -> List[ToolMessage]:
        """Direct tool execution fallback"""
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    if hasattr(tool, 'ainvoke'):
                        result = await tool.ainvoke(tool_input)
                    else:
                        result = tool.invoke(tool_input)

                    return [ToolMessage(
                        content=str(result),
                        name=tool_name,
                        tool_call_id=f"direct_{tool_name}"
                    )]
                except Exception as e:
                    return [ToolMessage(
                        content=f"Error: {str(e)}",
                        name=tool_name,
                        tool_call_id=f"error_{tool_name}"
                    )]

        return [ToolMessage(
            content=f"Tool '{tool_name}' not found",
            name=tool_name,
            tool_call_id=f"notfound_{tool_name}"
        )]

# Version detection utilities
def get_langgraph_version():
    """Get the installed langgraph version"""
    try:
        import langgraph
        return getattr(langgraph, '__version__', 'unknown')
    except:
        return 'not_installed'

def check_langgraph_compatibility():
    """Check langgraph compatibility and log warnings if needed"""
    version = get_langgraph_version()
    logger.info(f"LangGraph version: {version}")

    if version == 'not_installed':
        logger.warning("LangGraph is not installed!")
        return False

    if version == 'unknown':
        logger.warning("Could not determine LangGraph version")

    # Check for required imports
    required_imports = [
        ('langgraph.graph', 'StateGraph'),
        ('langgraph.graph', 'END'),
    ]

    missing = []
    for module, attr in required_imports:
        try:
            exec(f"from {module} import {attr}")
        except ImportError:
            missing.append(f"{module}.{attr}")

    if missing:
        logger.warning(f"Missing required imports: {missing}")
        return False

    return True

# Initialize compatibility check on import
compatibility_ok = check_langgraph_compatibility()
if not compatibility_ok:
    logger.warning("LangGraph compatibility issues detected - using fallback implementations")