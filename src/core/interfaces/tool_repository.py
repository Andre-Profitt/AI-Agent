from src.tools.base_tool import Tool

from src.tools.base_tool import ToolType
# TODO: Fix undefined variables: ABC, List, Optional, UUID, abstractmethod

"""
from abc import abstractmethod
from src.gaia_components.adaptive_tool_system import Tool
from src.gaia_components.adaptive_tool_system import ToolType

from langchain.tools import Tool
Tool repository interface defining the contract for tool persistence.
"""

from typing import Optional

from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from src.core.entities.tool import Tool, ToolType

class ToolRepository(ABC):
    """
    Abstract interface for tool persistence operations.
    """
    @abstractmethod
    async def save(self, tool: Tool) -> Tool:
        pass

    @abstractmethod
    async def find_by_id(self, tool_id: UUID) -> Optional[Tool]:
        pass

    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[Tool]:
        pass

    @abstractmethod
    async def find_by_type(self, tool_type: ToolType) -> List[Tool]:
        pass

    @abstractmethod
    async def delete(self, tool_id: UUID) -> bool:
        pass

    @abstractmethod
    async def get_statistics(self) -> dict:
        pass