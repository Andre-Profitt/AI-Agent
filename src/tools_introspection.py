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
            error_message: The error message
            attempted_params: Parameters that were attempted
            
        Returns:
            ToolCallError with analysis and suggestions
        """
        error = ToolCallError(
            tool_name=tool_name,
            error_type=self._classify_error(error_message),
            error_message=error_message,
            attempted_params=attempted_params
        )
        
        # Get tool schema for comparison
        schema = self.get_tool_schema(tool_name)
        
        if "error" not in schema:
            # Analyze parameter mismatches
            suggestion = self._generate_correction_suggestion(
                schema, attempted_params, error_message
            )
            error.suggestion = suggestion
        
        # Store in history for learning
        self.error_history.append(error)
        
        return error
    
    def _classify_error(self, error_message: str) -> str:
        """Classify the type of error"""
        error_lower = error_message.lower()
        
        if any(keyword in error_lower for keyword in ["parameter", "argument", "missing", "required"]):
            return "parameter_error"
        elif any(keyword in error_lower for keyword in ["validation", "invalid", "format"]):
            return "validation_error"
        else:
            return "execution_error"
    
    def _generate_correction_suggestion(
        self,
        schema: Dict[str, Any],
        attempted_params: Dict[str, Any],
        error_message: str
    ) -> str:
        """Generate a suggestion for correcting the tool call"""
        suggestions = []
        
        # Check for missing required parameters
        required = set(schema.get("required_parameters", []))
        provided = set(attempted_params.keys())
        missing = required - provided
        
        if missing:
            suggestions.append(
                f"Missing required parameters: {', '.join(missing)}"
            )
        
        # Check for unknown parameters
        known = set(schema.get("parameters", {}).keys())
        unknown = provided - known
        
        if unknown:
            suggestions.append(
                f"Unknown parameters provided: {', '.join(unknown)}"
            )
        
        # Check for type mismatches
        for param, value in attempted_params.items():
            if param in schema.get("parameters", {}):
                expected_type = schema["parameters"][param].get("type")
                if expected_type and not self._check_type_match(value, expected_type):
                    suggestions.append(
                        f"Parameter '{param}' expects type '{expected_type}', "
                        f"but got '{type(value).__name__}'"
                    )
        
        # Add schema reference
        if suggestions:
            suggestions.append(
                f"Use get_tool_schema('{schema['name']}') to see the full schema"
            )
        
        return "; ".join(suggestions) if suggestions else "No specific suggestion available"
    
    def _check_type_match(self, value: Any, expected_type: str) -> bool:
        """Check if a value matches the expected type"""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected_python_type = type_map.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, assume OK
    
    def _get_tool_examples(self, tool_name: str) -> List[Dict[str, Any]]:
        """Get examples for a specific tool"""
        # Tool-specific examples
        examples = {
            "web_search": [
                {
                    "query": "latest AI developments 2024",
                    "max_results": 5
                }
            ],
            "python_interpreter": [
                {
                    "code": "import numpy as np\nprint(np.array([1, 2, 3]).mean())"
                }
            ],
            "ask_user_for_clarification": [
                {
                    "question": "Which specific date would you like to book the flight for?",
                    "context": "User asked to book a flight but didn't specify dates"
                }
            ]
        }
        
        return examples.get(tool_name, [])
    
    def suggest_alternative_tool(
        self,
        failed_tool: str,
        task_description: str
    ) -> Optional[str]:
        """
        Suggest an alternative tool when one fails
        
        Args:
            failed_tool: Name of the tool that failed
            task_description: Description of what the user is trying to do
            
        Returns:
            Name of alternative tool, or None
        """
        # Simple heuristic-based suggestions
        alternatives = {
            "python_interpreter": ["calculator", "data_analyzer"],
            "web_search": ["knowledge_base_search", "ask_user_for_clarification"],
            "file_reader": ["web_search", "ask_user_for_clarification"]
        }
        
        tool_alternatives = alternatives.get(failed_tool, [])
        
        # Check which alternatives are actually available
        available_alternatives = [
            tool for tool in tool_alternatives 
            if tool in self.tool_registry
        ]
        
        return available_alternatives[0] if available_alternatives else None


# Tool functions

def get_tool_schema(tool_name: str) -> str:
    """
    Get the schema and usage information for a specific tool.
    
    Use this when a tool call fails to understand the correct parameters
    and format for calling the tool.
    
    Args:
        tool_name: Name of the tool to inspect
        
    Returns:
        JSON string containing the tool's schema
    """
    # This will be connected to the actual tool registry in integration
    # For now, return a mock response
    mock_schemas = {
        "web_search": {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5
                }
            },
            "required_parameters": ["query"],
            "examples": [
                {"query": "latest AI news", "max_results": 3}
            ]
        }
    }
    
    schema = mock_schemas.get(tool_name, {"error": f"Unknown tool: {tool_name}"})
    return json.dumps(schema, indent=2)


def analyze_tool_error(
    tool_name: str,
    error_message: str,
    attempted_params: str
) -> str:
    """
    Analyze why a tool call failed and get suggestions for correction.
    
    Use this after a tool call fails to understand what went wrong
    and how to fix it.
    
    Args:
        tool_name: Name of the tool that failed
        error_message: The error message from the failed call
        attempted_params: JSON string of the parameters that were attempted
        
    Returns:
        Analysis and suggestions for fixing the tool call
    """
    try:
        params = json.loads(attempted_params)
    except:
        params = {"raw": attempted_params}
    
    # Mock analysis
    analysis = {
        "tool": tool_name,
        "error_type": "parameter_error",
        "issue": "Missing required parameter 'query'",
        "suggestion": f"The {tool_name} tool requires a 'query' parameter. "
                     f"Use get_tool_schema('{tool_name}') to see all parameters.",
        "corrected_example": {
            "query": "your search query here",
            "max_results": 5
        }
    }
    
    return json.dumps(analysis, indent=2)


# Create tool instances
get_tool_schema_tool = StructuredTool.from_function(
    func=get_tool_schema,
    name="get_tool_schema",
    description="""Get the schema and usage information for a specific tool. 
    Use this when a tool call fails to understand the correct parameters and format."""
)

analyze_tool_error_tool = StructuredTool.from_function(
    func=analyze_tool_error,
    name="analyze_tool_error",
    description="""Analyze why a tool call failed and get suggestions for correction. 
    Use this after a tool call fails to understand what went wrong and how to fix it."""
)


# Self-correction prompt template
SELF_CORRECTION_PROMPT = """
Your previous tool call failed with the following error:
Tool: {tool_name}
Error: {error_message}
Parameters attempted: {attempted_params}

To fix this:
1. First, use get_tool_schema("{tool_name}") to understand the correct parameters
2. Analyze what went wrong using the schema information
3. Formulate a corrected tool call with the proper parameters

Remember:
- Check required vs optional parameters
- Verify parameter types match the schema
- Use the examples provided in the schema as a guide
"""


# Export introspection tools
INTROSPECTION_TOOLS = [
    get_tool_schema_tool,
    analyze_tool_error_tool
] 