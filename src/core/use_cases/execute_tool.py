from agent import tools
from examples.enhanced_unified_example import execution_time
from examples.enhanced_unified_example import start_time
from examples.parallel_execution_example import tool_name
from performance_dashboard import stats

from src.application.tools.tool_executor import validation_result
from src.core.use_cases.execute_tool import tool_list
from src.database.models import parameters
from src.database.models import tool
from src.database.models import tool_id
from src.database.models import tool_type
from src.main import logging_service
from src.main import tool_executor

from src.tools.base_tool import Tool

from src.tools.base_tool import ToolType
# TODO: Fix undefined variables: Any, Dict, Optional, UUID, context, e, execution_time, logging, logging_service, parameters, result, start_time, stats, time, tool_executor, tool_id, tool_list, tool_name, tool_repository, tool_type, tools, validation_result
from src.tools.base_tool import tool


"""
from typing import Dict
from src.core.interfaces.tool_executor import ToolExecutor
from src.core.interfaces.tool_repository import ToolRepository
from src.gaia_components.adaptive_tool_system import ToolType
from src.infrastructure.logging.logging_service import LoggingService
from src.shared.exceptions import ValidationException
# TODO: Fix undefined variables: context, e, execution_time, logging_service, parameters, result, self, start_time, stats, tool, tool_executor, tool_id, tool_list, tool_name, tool_repository, tool_type, tools, validation_result

Use case for executing tools.
"""

from typing import Optional
from typing import Any

from uuid import UUID
import logging
import time

from src.core.interfaces.logging_service import LoggingService
from src.shared.exceptions import DomainException, ValidationException

class ExecuteToolUseCase:
    """
    Use case for executing tools.

    This use case handles tool execution, validation,
    and result processing.
    """

    def __init__(
        self,
        tool_repository: ToolRepository,
        tool_executor: ToolExecutor,
        logging_service: LoggingService
    ):
        self.tool_repository = tool_repository
        self.tool_executor = tool_executor
        self.logging_service = logging_service
        self.logger = logging.getLogger(__name__)

    async def execute_tool(
        self,
        tool_id: UUID,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool with given parameters.

        Args:
            tool_id: ID of the tool to execute
            parameters: Tool execution parameters
            context: Optional execution context

        Returns:
            Dictionary containing the execution result
        """
        start_time = time.time()

        try:
            # Validate input
            if not parameters:
                raise ValidationException("Tool parameters cannot be empty")

            # Find tool
            tool = await self.tool_repository.find_by_id(tool_id)
            if not tool:
                raise DomainException(f"Tool {tool_id} not found")

            # Validate tool is available
            if not tool.is_available:
                raise DomainException(f"Tool {tool_id} is not available")

            # Execute tool
            result = await self.tool_executor.execute(tool, parameters, context)

            execution_time = time.time() - start_time

            # Log execution
            await self.logging_service.log_info(
                "tool_executed",
                f"Executed tool {tool_id} successfully",
                {
                    "tool_id": str(tool_id),
                    "tool_name": tool.name,
                    "execution_time": execution_time,
                    "success": result.get("success", False)
                }
            )

            return {
                "success": True,
                "tool_id": str(tool_id),
                "tool_name": tool.name,
                "result": result.get("result"),
                "execution_time": execution_time,
                "metadata": result.get("metadata", {})
            }

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error("Tool execution failed: {}", extra={"str_e_": str(e)})

            await self.logging_service.log_error(
                "tool_execution_failed",
                str(e),
                {
                    "tool_id": str(tool_id),
                    "execution_time": execution_time
                }
            )

            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time
            }

    async def execute_tool_by_name(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            parameters: Tool execution parameters
            context: Optional execution context

        Returns:
            Dictionary containing the execution result
        """
        try:
            # Find tool by name
            tool = await self.tool_repository.find_by_name(tool_name)
            if not tool:
                raise DomainException(f"Tool '{tool_name}' not found")

            # Execute using the found tool
            return await self.execute_tool(tool.id, parameters, context)

        except Exception as e:
            self.logger.error("Failed to execute tool by name '{}': {}", extra={"tool_name": tool_name, "str_e_": str(e)})
            return {"success": False, "error": str(e)}

    async def validate_tool_parameters(
        self,
        tool_id: UUID,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate tool parameters before execution.

        Args:
            tool_id: ID of the tool to validate
            parameters: Parameters to validate

        Returns:
            Dictionary containing validation result
        """
        try:
            # Find tool
            tool = await self.tool_repository.find_by_id(tool_id)
            if not tool:
                return {"success": False, "error": f"Tool {tool_id} not found"}

            # Validate parameters
            validation_result = await self.tool_executor.validate_parameters(tool, parameters)

            return {
                "success": True,
                "valid": validation_result.get("valid", False),
                "errors": validation_result.get("errors", []),
                "warnings": validation_result.get("warnings", [])
            }

        except Exception as e:
            self.logger.error("Parameter validation failed: {}", extra={"str_e_": str(e)})
            return {"success": False, "error": str(e)}

    async def get_tool_info(self, tool_id: UUID) -> Dict[str, Any]:
        """
        Get tool information.

        Args:
            tool_id: ID of the tool to retrieve

        Returns:
            Dictionary containing tool information
        """
        try:
            tool = await self.tool_repository.find_by_id(tool_id)
            if not tool:
                return {"success": False, "error": f"Tool {tool_id} not found"}

            return {
                "success": True,
                "tool": {
                    "id": str(tool.id),
                    "name": tool.name,
                    "description": tool.description,
                    "tool_type": tool.tool_type.value,
                    "is_available": tool.is_available,
                    "parameters_schema": tool.parameters_schema,
                    "created_at": tool.created_at.isoformat() if tool.created_at else None,
                    "updated_at": tool.updated_at.isoformat() if tool.updated_at else None
                }
            }

        except Exception as e:
            self.logger.error("Failed to get tool info {}: {}", extra={"tool_id": tool_id, "str_e_": str(e)})
            return {"success": False, "error": str(e)}

    async def list_available_tools(self, tool_type: Optional[ToolType] = None) -> Dict[str, Any]:
        """
        List available tools, optionally filtered by type.

        Args:
            tool_type: Optional tool type filter

        Returns:
            Dictionary containing the list of tools
        """
        try:
            if tool_type:
                tools = await self.tool_repository.find_by_type(tool_type)
            else:
                tools = await self.tool_repository.find_available()

            tool_list = []
            for tool in tools:
                tool_list.append({
                    "id": str(tool.id),
                    "name": tool.name,
                    "description": tool.description,
                    "tool_type": tool.tool_type.value,
                    "is_available": tool.is_available
                })

            return {
                "success": True,
                "tools": tool_list,
                "count": len(tool_list)
            }

        except Exception as e:
            self.logger.error("Failed to list tools: {}", extra={"str_e_": str(e)})
            return {"success": False, "error": str(e)}

    async def get_tool_statistics(self) -> Dict[str, Any]:
        """
        Get tool repository statistics.

        Returns:
            Dictionary containing tool statistics
        """
        try:
            stats = await self.tool_repository.get_statistics()
            return {"success": True, "statistics": stats}

        except Exception as e:
            self.logger.error("Failed to get tool statistics: {}", extra={"str_e_": str(e)})
            return {"success": False, "error": str(e)}