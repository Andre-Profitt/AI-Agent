from agent import query
from examples.enhanced_unified_example import output
from examples.enhanced_unified_example import start_time
from examples.parallel_execution_example import results

from src.agents.crew_enhanced import successful_executions
from src.agents.crew_enhanced import total_executions
from src.core.entities.tool import enabled_tools
from src.database.models import parameters
from src.database.models import tool
from src.database.models import tool_id
from src.database.models import tool_type
from src.gaia_components.adaptive_tool_system import available_tools
from src.gaia_components.adaptive_tool_system import total_tools
from src.meta_cognition import query_lower
from src.tools_introspection import name
from src.unified_architecture.enhanced_platform import alpha
from src.utils.tools_introspection import field
from src.utils.tools_introspection import required_params

from src.tools.base_tool import Tool

from src.agents.advanced_agent_fsm import Agent

from src.tools.base_tool import ToolType
from uuid import uuid4
# TODO: Fix undefined variables: Any, Callable, Dict, Enum, List, Optional, ToolExecutionException, UUID, alpha, available_tools, dataclass, datetime, e, enabled_tools, field, name, output, parameters, query, query_lower, required_params, result, results, start_time, successful_executions, tag, tool_id, tool_type, total_executions, total_tools, uuid4
import param

from src.tools.base_tool import tool

# TODO: Fix undefined variables: ToolExecutionException, alpha, available_tools, e, enabled_tools, name, output, param, parameters, query, query_lower, required_params, result, results, self, start_time, successful_executions, tag, tool, tool_id, tool_type, total_executions, total_tools

"""
Core Tool entity representing executable capabilities in the AI Agent system.
"""

from typing import Optional
from dataclasses import field
from typing import Any
from typing import List
from typing import Callable

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from uuid import UUID, uuid4

from src.shared.exceptions import ValidationException, ToolExecutionException

class ToolType(str, Enum):
    """Types of tools available in the system."""
    SEARCH = "search"
    CALCULATION = "calculation"
    FILE_OPERATION = "file_operation"
    API_CALL = "api_call"
    DATABASE_QUERY = "database_query"
    IMAGE_PROCESSING = "image_processing"
    AUDIO_PROCESSING = "audio_processing"
    VIDEO_PROCESSING = "video_processing"
    CUSTOM = "custom"

class ToolStatus(str, Enum):
    """Status of tool execution."""
    IDLE = "idle"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    DISABLED = "disabled"

@dataclass
class ToolResult:
    """
    Result of tool execution.

    This encapsulates the output, metadata, and execution details
    of a tool execution.
    """

    # Identity
    tool_id: UUID
    execution_id: UUID = field(default_factory=uuid4)

    # Results
    success: bool = field(default=True)
    output: Any = field(default=None)
    error_message: Optional[str] = field(default=None)

    # Execution details
    execution_time: float = field(default=0.0)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = field(default=None)

    # Metadata
    input_parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set end time if not provided."""
        if self.end_time is None:
            self.end_time = datetime.now()

    @property
    def duration(self) -> float:
        """Get execution duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "tool_id": str(self.tool_id),
            "execution_id": str(self.execution_id),
            "success": self.success,
            "output": self.output,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "duration": self.duration,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "input_parameters": self.input_parameters,
            "metadata": self.metadata
        }

@dataclass
class Tool:
    """
    Core Tool entity representing an executable capability.

    This entity encapsulates tool functionality, validation,
    execution tracking, and performance metrics.
    """

    # Identity
    id: UUID = field(default_factory=uuid4)
    name: str = field(default="")
    description: str = field(default="")
    tool_type: ToolType = field(default=ToolType.CUSTOM)

    # Functionality
    function: Optional[Callable] = field(default=None)
    parameters_schema: Dict[str, Any] = field(default_factory=dict)
    return_schema: Dict[str, Any] = field(default_factory=dict)

    # Status and configuration
    status: ToolStatus = field(default=ToolStatus.IDLE)
    is_enabled: bool = field(default=True)
    is_public: bool = field(default=True)

    # Performance tracking
    total_executions: int = field(default=0)
    successful_executions: int = field(default=0)
    failed_executions: int = field(default=0)
    average_execution_time: float = field(default=0.0)

    # Metadata
    version: str = field(default="1.0.0")
    author: Optional[str] = field(default=None)
    tags: List[str] = field(default_factory=list)
    documentation: Optional[str] = field(default=None)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_executed_at: Optional[datetime] = field(default=None)

    def __post_init__(self):
        """Validate tool after initialization."""
        if not self.name.strip():
            raise ValidationException("Tool name cannot be empty")

        if not self.description.strip():
            raise ValidationException("Tool description cannot be empty")

        if self.function is None:
            raise ValidationException("Tool must have an executable function")

    def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute the tool with given parameters.

        Args:
            parameters: Input parameters for the tool

        Returns:
            ToolResult containing execution results

        Raises:
            ToolExecutionException: If execution fails
        """
        if not self.is_enabled:
            raise ToolExecutionException(f"Tool '{self.name}' is disabled")

        if self.status == ToolStatus.EXECUTING:
            raise ToolExecutionException(f"Tool '{self.name}' is already executing")

        # Validate parameters
        self._validate_parameters(parameters)

        # Update status
        self.status = ToolStatus.EXECUTING
        self.updated_at = datetime.now()

        start_time = datetime.now()
        result = ToolResult(
            tool_id=self.id,
            input_parameters=parameters,
            start_time=start_time
        )

        try:
            # Execute the function
            output = self.function(**parameters)

            # Update result
            result.success = True
            result.output = output
            result.end_time = datetime.now()
            result.execution_time = (result.end_time - result.start_time).total_seconds()

            # Update tool metrics
            self._update_metrics(result)

            # Update status
            self.status = ToolStatus.COMPLETED
            self.last_executed_at = datetime.now()

        except Exception as e:
            # Handle execution failure
            result.success = False
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.execution_time = (result.end_time - result.start_time).total_seconds()

            # Update tool metrics
            self._update_metrics(result)

            # Update status
            self.status = ToolStatus.FAILED

            raise ToolExecutionException(f"Tool execution failed: {str(e)}")

        finally:
            self.updated_at = datetime.now()

        return result

    def _validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate input parameters against schema."""
        if not self.parameters_schema:
            return  # No schema defined, skip validation

        # Basic validation - in production, use JSON Schema validation
        required_params = self.parameters_schema.get("required", [])
        for param in required_params:
            if param not in parameters:
                raise ValidationException(f"Required parameter '{param}' is missing")

    def _update_metrics(self, result: ToolResult) -> None:
        """Update tool performance metrics."""
        self.total_executions += 1

        if result.success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1

        # Update average execution time
        if self.total_executions == 1:
            self.average_execution_time = result.execution_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_execution_time = (
                alpha * result.execution_time +
                (1 - alpha) * self.average_execution_time
            )

    def enable(self) -> None:
        """Enable the tool."""
        self.is_enabled = True
        self.status = ToolStatus.IDLE
        self.updated_at = datetime.now()

    def disable(self) -> None:
        """Disable the tool."""
        self.is_enabled = False
        self.status = ToolStatus.DISABLED
        self.updated_at = datetime.now()

    def add_tag(self, tag: str) -> None:
        """Add a tag to the tool."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the tool."""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.now()

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    @property
    def is_available(self) -> bool:
        """Check if tool is available for execution."""
        return self.is_enabled and self.status in [ToolStatus.IDLE, ToolStatus.COMPLETED, ToolStatus.FAILED]

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary representation."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "tool_type": self.tool_type.value,
            "parameters_schema": self.parameters_schema,
            "return_schema": self.return_schema,
            "status": self.status.value,
            "is_enabled": self.is_enabled,
            "is_public": self.is_public,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": self.success_rate,
            "average_execution_time": self.average_execution_time,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "documentation": self.documentation,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_executed_at": self.last_executed_at.isoformat() if self.last_executed_at else None,
            "is_available": self.is_available
        }

@dataclass
class ToolRegistry:
    """
    Registry for managing tools in the system.

    This provides centralized tool management, discovery,
    and execution coordination.
    """

    # Tools storage
    tools: Dict[UUID, Tool] = field(default_factory=dict)
    tools_by_name: Dict[str, Tool] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def register_tool(self, tool: Tool) -> None:
        """Register a tool in the registry."""
        if not isinstance(tool, Tool):
            raise ValidationException("Can only register Tool objects")

        # Check for name conflicts
        if tool.name in self.tools_by_name:
            raise ValidationException(f"Tool with name '{tool.name}' already registered")

        # Register tool
        self.tools[tool.id] = tool
        self.tools_by_name[tool.name] = tool

        self.updated_at = datetime.now()

    def unregister_tool(self, tool_id: UUID) -> None:
        """Unregister a tool from the registry."""
        if tool_id not in self.tools:
            raise ValidationException(f"Tool with ID '{tool_id}' not found")

        tool = self.tools[tool_id]
        del self.tools[tool_id]
        del self.tools_by_name[tool.name]

        self.updated_at = datetime.now()

    def get_tool(self, tool_id: UUID) -> Optional[Tool]:
        """Get a tool by ID."""
        return self.tools.get(tool_id)

    def get_tool_by_name(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools_by_name.get(name)

    def get_tools_by_type(self, tool_type: ToolType) -> List[Tool]:
        """Get all tools of a specific type."""
        return [tool for tool in self.tools.values() if tool.tool_type == tool_type]

    def get_available_tools(self) -> List[Tool]:
        """Get all available tools."""
        return [tool for tool in self.tools.values() if tool.is_available]

    def get_enabled_tools(self) -> List[Tool]:
        """Get all enabled tools."""
        return [tool for tool in self.tools.values() if tool.is_enabled]

    def search_tools(self, query: str) -> List[Tool]:
        """Search tools by name, description, or tags."""
        query_lower = query.lower()
        results = []

        for tool in self.tools.values():
            if (query_lower in tool.name.lower() or
                query_lower in tool.description.lower() or
                any(query_lower in tag.lower() for tag in tool.tags)):
                results.append(tool)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total_tools = len(self.tools)
        enabled_tools = len(self.get_enabled_tools())
        available_tools = len(self.get_available_tools())

        total_executions = sum(tool.total_executions for tool in self.tools.values())
        successful_executions = sum(tool.successful_executions for tool in self.tools.values())

        return {
            "total_tools": total_tools,
            "enabled_tools": enabled_tools,
            "available_tools": available_tools,
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0.0,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary representation."""
        return {
            "tools": [tool.to_dict() for tool in self.tools.values()],
            "statistics": self.get_statistics()
        }