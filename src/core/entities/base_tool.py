"""Base Tool Entity - Clean Implementation"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Tool:
    """Tool definition"""
    name: str
    description: str
    parameters: Dict[str, Any]
    
    async def execute(self, **kwargs) -> Any:
        """Execute tool - to be implemented by subclasses"""
        raise NotImplementedError(f"Tool {self.name} must implement execute method")
        
@dataclass
class ToolResult:
    """Tool execution result"""
    tool_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None