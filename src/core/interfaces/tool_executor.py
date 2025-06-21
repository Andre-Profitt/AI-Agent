from src.tools.base_tool import Tool
# TODO: Fix undefined variables: ABC, Any, Dict, Optional, UUID, abstractmethod

"""
from abc import abstractmethod
from src.gaia_components.adaptive_tool_system import Tool

from langchain.tools import Tool
Tool executor interface defining the contract for tool execution.
"""

from typing import Any
from typing import Optional

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from uuid import UUID

from src.core.entities.tool import Tool

class ToolExecutor(ABC):
    """
    Abstract interface for tool execution.

    This interface defines the contract that all tool executor
    implementations must follow, ensuring consistency across
    different execution strategies.
    """

    @abstractmethod
    async def execute(self, tool: Tool, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a tool with given parameters.

        Args:
            tool: The tool to execute
            parameters: Tool execution parameters
            context: Optional execution context

        Returns:
            Dictionary containing the execution result

        Raises:
            DomainException: If execution fails
        """
        pass

    @abstractmethod
    async def validate_parameters(self, tool: Tool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tool parameters before execution.

        Args:
            tool: The tool to validate parameters for
            parameters: Parameters to validate

        Returns:
            Dictionary containing validation result
        """
        pass

    @abstractmethod
    async def get_tool_info(self, tool: Tool) -> Dict[str, Any]:
        """
        Get tool information and capabilities.

        Args:
            tool: The tool to query

        Returns:
            Dictionary containing tool information
        """
        pass

    @abstractmethod
    async def execute_batch(self, tools: list[Tool], parameters_list: list[Dict[str, Any]], context: Optional[Dict[str, Any]] = None) -> list[Dict[str, Any]]:
        """
        Execute multiple tools in batch.

        Args:
            tools: List of tools to execute
            parameters_list: List of parameter dictionaries for each tool
            context: Optional execution context

        Returns:
            List of execution results
        """
        pass

    @abstractmethod
    async def cancel_execution(self, execution_id: UUID) -> bool:
        """
        Cancel a running tool execution.

        Args:
            execution_id: The execution to cancel

        Returns:
            True if cancellation was successful, False otherwise
        """
        pass