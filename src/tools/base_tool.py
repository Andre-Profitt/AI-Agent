"""
Base tool class for all AI Agent tools
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class ToolResult:
    """Result from tool execution"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseTool(ABC):
    """Base class for all tools"""
    
    def __init__(self, name: str, description: str, version: str = "1.0.0"):
        self.name = name
        self.description = description
        self.version = version
        self.metadata = {}
    
    @abstractmethod
    async def arun(self, **kwargs) -> Any:
        """Async execution method"""
        pass
    
    def run(self, **kwargs) -> Any:
        """Sync execution method (default implementation)"""
        import asyncio
        return asyncio.run(self.arun(**kwargs))
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for documentation"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "parameters": self._get_parameters_schema(),
            "metadata": self.metadata
        }
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get parameters schema - override in subclasses"""
        return {}
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate input parameters - override in subclasses"""
        return True
    
    def __str__(self):
        return f"{self.name} (v{self.version})"
    
    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"

def tool(func):
    """Decorator to create a tool from a function"""
    class FunctionTool(BaseTool):
        def __init__(self):
            super().__init__(
                name=func.__name__,
                description=func.__doc__ or "No description available"
            )
            self._func = func
        
        async def arun(self, **kwargs):
            import inspect
            if inspect.iscoroutinefunction(self._func):
                return await self._func(**kwargs)
            else:
                return self._func(**kwargs)
    
    return FunctionTool() 