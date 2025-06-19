"""
Tool Introspection Module
Enables agents to inspect tool schemas and self-correct invalid calls
"""

from typing import Dict, Any, List, Optional
import json
from dataclasses import dataclass

from langchain_core.tools import StructuredTool, BaseTool
from pydantic import BaseModel, Field


class ToolSchemaInfo(BaseModel):
    """Schema information for a tool"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required_parameters: List[str]
    examples: Optional[List[Dict[str, Any]]] = None


@dataclass
class ToolCallError:
    """Represents an error from a tool call"""
    tool_name: str
    error_type: str  # "parameter_error", "execution_error", "validation_error"
    error_message: str
    attempted_params: Dict[str, Any]
    suggestion: Optional[str] = None


class ToolIntrospector:
    """Provides introspection capabilities for tools"""
    
    def __init__(self, tool_registry: Optional[Dict[str, BaseTool]] = None):
        """
        Initialize with a tool registry
        
        Args:
            tool_registry: Dictionary mapping tool names to tool instances
        """
        self.tool_registry = tool_registry or {}
        self.error_history: List[ToolCallError] = []
    
    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """
        Get the schema for a specific tool
        
        Args:
            tool_name: Name of the tool to inspect
            
        Returns:
            Dictionary containing tool schema information
        """
        if tool_name not in self.tool_registry:
            return {
                "error": f"Tool '{tool_name}' not found in registry",
                "available_tools": list(self.tool_registry.keys())
            }
        
        tool = self.tool_registry[tool_name]
        
        # Extract schema information
        schema_info = {
            "name": tool.name,
            "description": tool.description,
            "parameters": {},
            "required_parameters": [],
            "examples": []
        }
        
        # Get parameter schema if available
        if hasattr(tool, 'args_schema') and tool.args_schema:
            schema = tool.args_schema.schema()
            schema_info["parameters"] = schema.get("properties", {})
            schema_info["required_parameters"] = schema.get("required", [])
            
            # Add parameter descriptions
            for param_name, param_info in schema_info["parameters"].items():
                if "description" not in param_info and hasattr(tool.args_schema, "__fields__"):
                    field = tool.args_schema.__fields__.get(param_name)
                    if field and field.field_info.description:
                        param_info["description"] = field.field_info.description
        
        # Add examples if available
        schema_info["examples"] = self._get_tool_examples(tool_name)
        
        return schema_info
    
    def analyze_tool_error(
        self,
        tool_name: str,
        error_message: str,
        attempted_params: Dict[str, Any]
    ) -> ToolCallError:
        """
        Analyze a tool call error and provide suggestions
        
        Args:
            tool_name: Name of the tool that failed
            error_message: The error message from the tool call
            attempted_params: Parameters that were attempted
            
        Returns:
            ToolCallError with analysis and suggestions
        """
        error = ToolCallError(
            tool_name=tool_name,
            error_type="execution_error",
            error_message=error_message,
            attempted_params=attempted_params
        )
        
        # Analyze error patterns
        if "parameter" in error_message.lower():
            error.error_type = "parameter_error"
            error.suggestion = self._suggest_parameter_fix(tool_name, attempted_params)
        elif "validation" in error_message.lower():
            error.error_type = "validation_error"
            error.suggestion = self._suggest_validation_fix(tool_name, attempted_params)
        else:
            error.suggestion = self._suggest_general_fix(tool_name, error_message)
        
        # Store in history
        self.error_history.append(error)
        
        return error
    
    def _get_tool_examples(self, tool_name: str) -> List[Dict[str, Any]]:
        """Get example usage for a tool"""
        examples = {
            "file_reader": [
                {"filename": "document.txt", "description": "Read a text file"},
                {"filename": "data.csv", "description": "Read a CSV file"}
            ],
            "tavily_search": [
                {"query": "latest news about AI", "description": "Search for recent AI news"},
                {"query": "weather in New York", "description": "Get weather information"}
            ],
            "python_interpreter": [
                {"code": "print('Hello, World!')", "description": "Simple print statement"},
                {"code": "import math; print(math.pi)", "description": "Use math library"}
            ],
            "chess_logic_tool": [
                {"fen_string": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "description": "Starting position"},
                {"fen_string": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "description": "After 1.e4"}
            ]
        }
        
        return examples.get(tool_name, [])
    
    def _suggest_parameter_fix(self, tool_name: str, attempted_params: Dict[str, Any]) -> str:
        """Suggest fixes for parameter errors"""
        schema = self.get_tool_schema(tool_name)
        
        if "error" in schema:
            return f"Tool '{tool_name}' not found. Check available tools."
        
        required_params = schema.get("required_parameters", [])
        missing_params = [param for param in required_params if param not in attempted_params]
        
        if missing_params:
            return f"Missing required parameters: {missing_params}. Please provide these parameters."
        
        return "Check parameter types and values. Use get_tool_schema to see expected format."
    
    def _suggest_validation_fix(self, tool_name: str, attempted_params: Dict[str, Any]) -> str:
        """Suggest fixes for validation errors"""
        schema = self.get_tool_schema(tool_name)
        
        if "error" in schema:
            return f"Tool '{tool_name}' not found. Check available tools."
        
        return "Check parameter validation rules. Ensure all parameters meet the required constraints."
    
    def _suggest_general_fix(self, tool_name: str, error_message: str) -> str:
        """Suggest general fixes for tool errors"""
        return f"General error in {tool_name}: {error_message}. Check tool documentation and try again."
    
    def get_error_history(self) -> List[ToolCallError]:
        """Get the history of tool call errors"""
        return self.error_history.copy()
    
    def clear_error_history(self):
        """Clear the error history"""
        self.error_history.clear()


# Global introspector instance
tool_introspector = ToolIntrospector()


def register_tools(tools: List[BaseTool]):
    """
    Register tools with the introspector
    
    Args:
        tools: List of tools to register
    """
    for tool in tools:
        tool_introspector.tool_registry[tool.name] = tool


def get_tool_schema(tool_name: str) -> Dict[str, Any]:
    """
    Get schema information for a tool
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool schema information
    """
    return tool_introspector.get_tool_schema(tool_name)


def analyze_tool_error(tool_name: str, error_message: str, attempted_params: Dict[str, Any]) -> ToolCallError:
    """
    Analyze a tool call error
    
    Args:
        tool_name: Name of the tool that failed
        error_message: Error message
        attempted_params: Parameters that were attempted
        
    Returns:
        Analysis of the error with suggestions
    """
    return tool_introspector.analyze_tool_error(tool_name, error_message, attempted_params)


def get_available_tools() -> List[str]:
    """
    Get list of available tool names
    
    Returns:
        List of tool names
    """
    return list(tool_introspector.tool_registry.keys())


def get_tool_examples(tool_name: str) -> List[Dict[str, Any]]:
    """
    Get example usage for a tool
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        List of example usages
    """
    return tool_introspector._get_tool_examples(tool_name)


# Tool introspection functions for LangChain tools
def inspect_tool_parameters(tool: BaseTool) -> Dict[str, Any]:
    """
    Inspect the parameters of a LangChain tool
    
    Args:
        tool: The tool to inspect
        
    Returns:
        Parameter information
    """
    if hasattr(tool, 'args_schema') and tool.args_schema:
        schema = tool.args_schema.schema()
        return {
            "parameters": schema.get("properties", {}),
            "required": schema.get("required", []),
            "description": tool.description
        }
    else:
        return {
            "parameters": {},
            "required": [],
            "description": tool.description
        }


def validate_tool_call(tool: BaseTool, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate parameters for a tool call
    
    Args:
        tool: The tool to validate for
        parameters: Parameters to validate
        
    Returns:
        Validation result with any errors
    """
    if hasattr(tool, 'args_schema') and tool.args_schema:
        try:
            # Try to create an instance with the parameters
            validated = tool.args_schema(**parameters)
            return {
                "valid": True,
                "validated_parameters": validated.dict()
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "parameters": parameters
            }
    else:
        return {
            "valid": True,
            "validated_parameters": parameters
        }


def suggest_tool_alternatives(tool_name: str, available_tools: List[BaseTool]) -> List[str]:
    """
    Suggest alternative tools when the requested tool is not available
    
    Args:
        tool_name: Name of the requested tool
        available_tools: List of available tools
        
    Returns:
        List of alternative tool names
    """
    # Simple similarity-based suggestions
    alternatives = []
    tool_name_lower = tool_name.lower()
    
    for tool in available_tools:
        if hasattr(tool, 'name'):
            tool_name_alt = tool.name.lower()
            if (tool_name_lower in tool_name_alt or 
                tool_name_alt in tool_name_lower or
                any(word in tool_name_alt for word in tool_name_lower.split('_'))):
                alternatives.append(tool.name)
    
    return alternatives[:3]  # Return top 3 alternatives


class ToolsIntrospection:
    """Introspection tools class for importing"""
    
    def __init__(self):
        self.introspector = tool_introspector
    
    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """Get tool schema"""
        return self.introspector.get_tool_schema(tool_name)
    
    def analyze_tool_error(self, tool_name: str, error_message: str, attempted_params: Dict[str, Any]) -> ToolCallError:
        """Analyze tool error"""
        return self.introspector.analyze_tool_error(tool_name, error_message, attempted_params)
    
    def get_error_history(self) -> List[ToolCallError]:
        """Get error history"""
        return self.introspector.get_error_history()
    
    def clear_error_history(self):
        """Clear error history"""
        self.introspector.clear_error_history()
    
    def register_tools(self, tools: List[BaseTool]):
        """Register tools"""
        register_tools(tools) 