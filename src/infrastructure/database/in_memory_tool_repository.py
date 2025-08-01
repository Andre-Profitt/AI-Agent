from src.database.models import tool
from src.database.models import tool_id
from src.database.models import tool_type
from src.tools_introspection import name

from src.tools.base_tool import Tool

from src.tools.base_tool import ToolType
# TODO: Fix undefined variables: Dict, List, Optional, UUID, name, t, tool_id, tool_type
from src.tools.base_tool import tool


"""
from typing import Optional
from src.gaia_components.adaptive_tool_system import Tool
from src.gaia_components.adaptive_tool_system import ToolType
# TODO: Fix undefined variables: name, self, t, tool, tool_id, tool_type

from langchain.tools import Tool
In-memory implementation of the ToolRepository interface.
"""

from typing import Dict

from typing import List, Optional, Dict
from uuid import UUID

from src.core.entities.tool import Tool, ToolType
from src.core.interfaces.tool_repository import ToolRepository

class InMemoryToolRepository(ToolRepository):
    def __init__(self) -> None:
        self._tools: Dict[UUID, Tool] = {}
        self._tools_by_name: Dict[str, Tool] = {}

    async def save(self, tool: Tool) -> Tool:
        self._tools[tool.id] = tool
        self._tools_by_name[tool.name] = tool
        return tool

    async def find_by_id(self, tool_id: UUID) -> Optional[Tool]:
        return self._tools.get(tool_id)

    async def find_by_name(self, name: str) -> Optional[Tool]:
        return self._tools_by_name.get(name)

    async def find_by_type(self, tool_type: ToolType) -> List[Tool]:
        return [tool for tool in self._tools.values() if tool.tool_type == tool_type]

    async def delete(self, tool_id: UUID) -> bool:
        tool = self._tools.get(tool_id)
        if tool:
            del self._tools[tool_id]
            if tool.name in self._tools_by_name:
                del self._tools_by_name[tool.name]
            return True
        return False

    async def get_statistics(self) -> dict:
        return {
            "total_tools": len(self._tools),
            "enabled_tools": len([t for t in self._tools.values() if t.is_enabled]),
        }