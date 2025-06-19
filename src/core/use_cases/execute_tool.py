"""
Use case for executing tools.
"""

from typing import Dict, Any, Optional, List
from uuid import UUID
import logging
import time

from src.core.entities.tool import Tool, ToolType
from src.core.interfaces.tool_repository import ToolRepository
from src.core.interfaces.tool_executor import ToolExecutor
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
            self.logger.error(f"Tool execution failed: {str(e)}")
            
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
            self.logger.error(f"Failed to execute tool by name '{tool_name}': {str(e)}")
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
            self.logger.error(f"Parameter validation failed: {str(e)}")
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
            self.logger.error(f"Failed to get tool info {tool_id}: {str(e)}")
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
            self.logger.error(f"Failed to list tools: {str(e)}")
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
            self.logger.error(f"Failed to get tool statistics: {str(e)}")
            return {"success": False, "error": str(e)} 