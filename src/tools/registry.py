"""
Enhanced tool registry with discovery and validation
"""

import inspect
from typing import Dict, List, Any, Optional, Callable, Type
from dataclasses import dataclass
import importlib
import pkgutil
from pathlib import Path

from src.utils.logging import get_logger
from src.tools.base_tool import BaseTool

logger = get_logger(__name__)

@dataclass
class ToolMetadata:
    """Metadata for registered tools"""
    name: str
    description: str
    category: str
    version: str
    author: str
    reliability_score: float = 1.0
    average_execution_time: float = 0.0
    total_executions: int = 0
    last_error: Optional[str] = None

class ToolRegistry:
    """Central registry for all tools with enhanced features"""
    
    def __init__(self):
        self._tools: Dict[str, Any] = {}
        self._metadata: Dict[str, ToolMetadata] = {}
        self._categories: Dict[str, List[str]] = {}
        self._validators: List[Callable] = []
        
        # Auto-discovery settings
        self.auto_discover = True
        self.tool_paths = ["src.tools.implementations"]
    
    def register(
        self,
        tool: Any,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a tool with validation"""
        try:
            # Validate tool
            if not self._validate_tool(tool):
                logger.error("Tool validation failed: {}", extra={"tool": tool})
                return False
            
            # Extract tool info
            tool_name = getattr(tool, 'name', tool.__name__)
            
            # Check for duplicates
            if tool_name in self._tools:
                logger.warning("Tool {} already registered, updating...", extra={"tool_name": tool_name})
            
            # Register tool
            self._tools[tool_name] = tool
            
            # Create metadata
            tool_meta = ToolMetadata(
                name=tool_name,
                description=getattr(tool, 'description', ''),
                category=category,
                version=getattr(tool, '__version__', '1.0.0'),
                author=getattr(tool, '__author__', 'Unknown')
            )
            
            # Update with provided metadata
            if metadata:
                for key, value in metadata.items():
                    if hasattr(tool_meta, key):
                        setattr(tool_meta, key, value)
            
            self._metadata[tool_name] = tool_meta
            
            # Update categories
            if category not in self._categories:
                self._categories[category] = []
            
            if tool_name not in self._categories[category]:
                self._categories[category].append(tool_name)
            
            logger.info("Registered tool: {} in category: {}", extra={"tool_name": tool_name, "category": category})
            return True
            
        except Exception as e:
            logger.error("Failed to register tool: {}", exc_info=True)
            return False
    
    def _validate_tool(self, tool: Any) -> bool:
        """Validate tool meets requirements"""
        # Check if it's a callable or has required methods
        if hasattr(tool, 'run') or hasattr(tool, 'arun'):
            return True
        
        if callable(tool):
            return True
        
        # Check if it's a LangChain tool
        if hasattr(tool, '__class__'):
            if 'Tool' in str(tool.__class__.__mro__):
                return True
        
        # Run custom validators
        for validator in self._validators:
            if not validator(tool):
                return False
        
        return False
    
    def add_validator(self, validator: Callable[[Any], bool]):
        """Add custom tool validator"""
        self._validators.append(validator)
    
    def discover_tools(self, path: Optional[str] = None):
        """Auto-discover tools from modules"""
        if not self.auto_discover:
            return
        
        paths = [path] if path else self.tool_paths
        
        for module_path in paths:
            try:
                # Import the module
                module = importlib.import_module(module_path)
                
                # Get module directory
                if hasattr(module, '__path__'):
                    module_dir = Path(module.__path__[0])
                else:
                    module_dir = Path(module.__file__).parent
                
                # Find all Python files
                for importer, modname, ispkg in pkgutil.iter_modules([str(module_dir)]):
                    if modname.startswith('_'):
                        continue
                    
                    full_module_name = f"{}.{}"
                    
                    try:
                        # Import the module
                        submodule = importlib.import_module(full_module_name)
                        
                        # Find tools in the module
                        for name, obj in inspect.getmembers(submodule):
                            if name.startswith('_'):
                                continue
                            
                            # Check if it's a tool
                            if self._looks_like_tool(obj):
                                # Determine category from module name
                                category = self._get_category_from_module(modname)
                                self.register(obj, category=category)
                    
                    except Exception as e:
                        logger.warning("Failed to import {}: {}", extra={"e": e, "module_path": module_path, "modname": modname, "full_module_name": full_module_name, "e": e})
            
            except Exception as e:
                logger.error("Failed to discover tools in {}: {}", extra={"module_path": module_path, "e": e})
    
    def _looks_like_tool(self, obj: Any) -> bool:
        """Check if object looks like a tool"""
        # Check for tool decorators
        if hasattr(obj, '_is_tool'):
            return True
        
        # Check for tool base classes
        if inspect.isclass(obj):
            if issubclass(obj, BaseTool):
                return True
        
        # Check for tool-like methods
        if hasattr(obj, 'run') or hasattr(obj, 'arun'):
            if hasattr(obj, 'name') and hasattr(obj, 'description'):
                return True
        
        return False
    
    def _get_category_from_module(self, module_name: str) -> str:
        """Determine category from module name"""
        category_map = {
            'file': 'file_operations',
            'web': 'web_tools',
            'search': 'search_tools',
            'python': 'code_execution',
            'audio': 'media_processing',
            'video': 'media_processing',
            'image': 'media_processing',
            'database': 'data_tools',
            'api': 'api_tools'
        }
        
        for keyword, category in category_map.items():
            if keyword in module_name.lower():
                return category
        
        return 'general'
    
    def get_tool(self, name: str) -> Optional[Any]:
        """Get a tool by name"""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names"""
        return list(self._tools.keys())
    
    def list_categories(self) -> List[str]:
        """List all categories"""
        return list(self._categories.keys())
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """Get tools in a specific category"""
        return self._categories.get(category, [])
    
    def get_tool_metadata(self, name: str) -> Optional[ToolMetadata]:
        """Get metadata for a tool"""
        return self._metadata.get(name)
    
    def update_tool_stats(
        self,
        name: str,
        execution_time: float,
        success: bool,
        error: Optional[str] = None
    ):
        """Update tool execution statistics"""
        if name not in self._metadata:
            return
        
        metadata = self._metadata[name]
        metadata.total_executions += 1
        
        # Update average execution time
        if success:
            current_avg = metadata.average_execution_time
            new_avg = (
                (current_avg * (metadata.total_executions - 1) + execution_time)
                / metadata.total_executions
            )
            metadata.average_execution_time = new_avg
        
        # Update reliability score
        if not success:
            # Decrease reliability score
            metadata.reliability_score *= 0.95
            metadata.last_error = error
        else:
            # Slowly increase reliability score
            metadata.reliability_score = min(
                1.0,
                metadata.reliability_score * 1.01
            )
    
    def get_reliable_tools(self, threshold: float = 0.8) -> List[str]:
        """Get tools with reliability above threshold"""
        reliable = []
        
        for name, metadata in self._metadata.items():
            if metadata.reliability_score >= threshold:
                reliable.append(name)
        
        return reliable
    
    def export_registry(self) -> Dict[str, Any]:
        """Export registry data for persistence"""
        return {
            "tools": {
                name: {
                    "category": self._get_tool_category(name),
                    "metadata": {
                        "description": meta.description,
                        "version": meta.version,
                        "author": meta.author,
                        "reliability_score": meta.reliability_score,
                        "average_execution_time": meta.average_execution_time,
                        "total_executions": meta.total_executions
                    }
                }
                for name, meta in self._metadata.items()
            },
            "categories": self._categories
        }
    
    def _get_tool_category(self, tool_name: str) -> str:
        """Get category for a tool"""
        for category, tools in self._categories.items():
            if tool_name in tools:
                return category
        return "uncategorized"
    
    def import_registry(self, data: Dict[str, Any]):
        """Import registry data"""
        # This would restore tool metadata and categories
        # Tools themselves would need to be re-registered
        pass 