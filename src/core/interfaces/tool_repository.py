"""
Tool repository interface defining the contract for tool persistence.
"""

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