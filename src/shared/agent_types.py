"""

from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import Optional
Shared agent types to avoid circular imports.

This module contains type definitions and interfaces that are shared
between multiple modules to prevent circular dependencies.
"""

from typing import Protocol, Any, Dict, List, Optional
from abc import ABC, abstractmethod


class AgentProtocol(Protocol):
    """Protocol defining the interface for Agent objects."""
    
    name: str
    model_name: str
    
    async def run(self, query: str, correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """Run the agent with a query."""
        ...


class BaseAgent(ABC):
    """Abstract base class for agents to avoid circular imports."""
    
    def __init__(self, name: str, model_name: str = "llama-3.3-70b-versatile"):
        self.name = name
        self.model_name = model_name
    
    @abstractmethod
    async def run(self, query: str, correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """Run the agent with a query."""
        pass


# Type alias for Agent
Agent = AgentProtocol