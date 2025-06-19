"""
Tool executor implementation for executing tools.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4
from datetime import datetime

from src.core.interfaces.tool_executor import ToolExecutor
from src.core.entities.tool import Tool, ToolType
from src.shared.exceptions import DomainException


class ToolExecutorImpl(ToolExecutor):
    """
    Implementation of the tool executor interface.
    
    This class handles the execution of different types of tools
    and manages their lifecycle during processing.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._active_executions: Dict[UUID, Dict[str, Any]] = {}
    
    async def execute(self, tool: Tool, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a tool with given parameters.
        
        Args:
            tool: The tool to execute
            parameters: Tool execution parameters
            context: Optional execution context
            
        Returns:
            Dictionary containing the execution result
        """
        execution_id = uuid4()
        start_time = datetime.now()
        
        try:
            # Register execution
            self._active_executions[execution_id] = {
                "tool_id": tool.id,
                "parameters": parameters,
                "start_time": start_time,
                "status": "running"
            }
            
            self.logger.info(f"Starting tool execution {execution_id} for tool {tool.name}")
            
            # Validate parameters
            validation_result = await self.validate_parameters(tool, parameters)
            if not validation_result.get("valid", False):
                raise DomainException(f"Parameter validation failed: {validation_result.get('errors', [])}")
            
            # Execute based on tool type
            if tool.tool_type == ToolType.SEARCH:
                result = await self._execute_search_tool(tool, parameters, context)
            elif tool.tool_type == ToolType.CALCULATOR:
                result = await self._execute_calculator_tool(tool, parameters, context)
            elif tool.tool_type == ToolType.FILE_PROCESSOR:
                result = await self._execute_file_processor_tool(tool, parameters, context)
            elif tool.tool_type == ToolType.API_CALLER:
                result = await self._execute_api_caller_tool(tool, parameters, context)
            elif tool.tool_type == ToolType.CUSTOM:
                result = await self._execute_custom_tool(tool, parameters, context)
            else:
                raise DomainException(f"Unsupported tool type: {tool.tool_type}")
            
            # Update execution status
            execution_time = (datetime.now() - start_time).total_seconds()
            self._active_executions[execution_id]["status"] = "completed"
            self._active_executions[execution_id]["execution_time"] = execution_time
            
            # Add execution metadata
            result["execution_id"] = str(execution_id)
            result["execution_time"] = execution_time
            
            self.logger.info(f"Tool execution {execution_id} completed successfully in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self._active_executions[execution_id]["status"] = "failed"
            self._active_executions[execution_id]["error"] = str(e)
            self._active_executions[execution_id]["execution_time"] = execution_time
            
            self.logger.error(f"Tool execution {execution_id} failed: {str(e)}")
            raise DomainException(f"Tool execution failed: {str(e)}")
    
    async def validate_parameters(self, tool: Tool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tool parameters before execution.
        
        Args:
            tool: The tool to validate parameters for
            parameters: Parameters to validate
            
        Returns:
            Dictionary containing validation result
        """
        errors = []
        warnings = []
        
        # Check if tool is available
        if not tool.is_available:
            errors.append("Tool is not available for execution")
        
        # Check required parameters
        if hasattr(tool, 'parameters_schema') and tool.parameters_schema:
            required_params = tool.parameters_schema.get("required", [])
            for param in required_params:
                if param not in parameters:
                    errors.append(f"Required parameter '{param}' is missing")
        
        # Check parameter types (basic validation)
        if hasattr(tool, 'parameters_schema') and tool.parameters_schema:
            param_types = tool.parameters_schema.get("properties", {})
            for param_name, param_value in parameters.items():
                if param_name in param_types:
                    expected_type = param_types[param_name].get("type")
                    if expected_type and not self._validate_type(param_value, expected_type):
                        warnings.append(f"Parameter '{param_name}' may have incorrect type")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def get_tool_info(self, tool: Tool) -> Dict[str, Any]:
        """
        Get tool information and capabilities.
        
        Args:
            tool: The tool to query
            
        Returns:
            Dictionary containing tool information
        """
        info = {
            "id": str(tool.id),
            "name": tool.name,
            "description": tool.description,
            "tool_type": tool.tool_type.value,
            "is_available": tool.is_available,
            "parameters_schema": getattr(tool, 'parameters_schema', {}),
            "capabilities": []
        }
        
        # Add capabilities based on tool type
        if tool.tool_type == ToolType.SEARCH:
            info["capabilities"] = ["web_search", "information_retrieval", "query_processing"]
        elif tool.tool_type == ToolType.CALCULATOR:
            info["capabilities"] = ["mathematical_computation", "formula_evaluation", "unit_conversion"]
        elif tool.tool_type == ToolType.FILE_PROCESSOR:
            info["capabilities"] = ["file_reading", "file_writing", "format_conversion", "data_extraction"]
        elif tool.tool_type == ToolType.API_CALLER:
            info["capabilities"] = ["api_integration", "data_fetching", "service_interaction"]
        elif tool.tool_type == ToolType.CUSTOM:
            info["capabilities"] = ["custom_processing", "domain_specific_operations"]
        
        return info
    
    async def execute_batch(self, tools: List[Tool], parameters_list: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute multiple tools in batch.
        
        Args:
            tools: List of tools to execute
            parameters_list: List of parameter dictionaries for each tool
            context: Optional execution context
            
        Returns:
            List of execution results
        """
        if len(tools) != len(parameters_list):
            raise DomainException("Number of tools must match number of parameter sets")
        
        # Execute tools in parallel
        tasks = []
        for tool, parameters in zip(tools, parameters_list):
            task = self.execute(tool, parameters, context)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "tool_id": str(tools[i].id),
                    "tool_name": tools[i].name
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def cancel_execution(self, execution_id: UUID) -> bool:
        """
        Cancel a running tool execution.
        
        Args:
            execution_id: The execution to cancel
            
        Returns:
            True if cancellation was successful, False otherwise
        """
        if execution_id not in self._active_executions:
            return False
        
        execution = self._active_executions[execution_id]
        if execution["status"] == "running":
            execution["status"] = "cancelled"
            execution["end_time"] = datetime.now()
            self.logger.info(f"Tool execution {execution_id} cancelled")
            return True
        
        return False
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Basic type validation."""
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "number":
            return isinstance(value, (int, float))
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        return True  # Unknown type, assume valid
    
    async def _execute_search_tool(self, tool: Tool, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a search tool."""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        query = parameters.get("query", "")
        return {
            "success": True,
            "result": f"Search results for: {query}",
            "metadata": {
                "tool_type": "search",
                "query": query,
                "results_count": 10
            }
        }
    
    async def _execute_calculator_tool(self, tool: Tool, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a calculator tool."""
        await asyncio.sleep(0.05)  # Simulate processing time
        
        expression = parameters.get("expression", "")
        # Basic evaluation (in production, use safe_eval or similar)
        try:
            result = eval(expression)  # Note: This is unsafe, use safe_eval in production
            return {
                "success": True,
                "result": result,
                "metadata": {
                    "tool_type": "calculator",
                    "expression": expression
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Calculation failed: {str(e)}",
                "metadata": {
                    "tool_type": "calculator",
                    "expression": expression
                }
            }
    
    async def _execute_file_processor_tool(self, tool: Tool, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a file processor tool."""
        await asyncio.sleep(0.2)  # Simulate processing time
        
        file_path = parameters.get("file_path", "")
        operation = parameters.get("operation", "read")
        
        return {
            "success": True,
            "result": f"File {operation} completed for: {file_path}",
            "metadata": {
                "tool_type": "file_processor",
                "file_path": file_path,
                "operation": operation
            }
        }
    
    async def _execute_api_caller_tool(self, tool: Tool, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute an API caller tool."""
        await asyncio.sleep(0.3)  # Simulate processing time
        
        url = parameters.get("url", "")
        method = parameters.get("method", "GET")
        
        return {
            "success": True,
            "result": f"API call {method} to {url} completed",
            "metadata": {
                "tool_type": "api_caller",
                "url": url,
                "method": method,
                "status_code": 200
            }
        }
    
    async def _execute_custom_tool(self, tool: Tool, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a custom tool."""
        await asyncio.sleep(0.15)  # Simulate processing time
        
        return {
            "success": True,
            "result": f"Custom tool '{tool.name}' executed with parameters: {parameters}",
            "metadata": {
                "tool_type": "custom",
                "tool_name": tool.name,
                "parameters": parameters
            }
        }


# Alias for backward compatibility
ToolExecutor = ToolExecutorImpl 