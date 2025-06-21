from src.agents.advanced_agent_fsm import AgentType

from src.agents.advanced_agent_fsm import Agent
# TODO: Fix undefined variables: ABC, Any, Dict, List, Optional, UUID, abstractmethod
from src.core.entities.agent import AgentType


"""
from abc import abstractmethod
from src.agents.multi_agent_system import AgentState
from src.gaia_components.multi_agent_orchestrator import Agent
from src.infrastructure.agents.agent_factory import AgentType

Agent repository interface defining the contract for agent persistence.
"""

from typing import Any
from typing import Dict
from typing import Optional

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID

from src.core.entities.agent import Agent, AgentType, AgentState

class AgentRepository(ABC):
    """
    Abstract interface for agent persistence operations.

    This interface defines the contract that all agent repository
    implementations must follow, ensuring consistency across
    different storage backends.
    """

    @abstractmethod
    async def save(self, agent: Agent) -> Agent:
        """
        Save an agent to the repository.

        Args:
            agent: The agent to save

        Returns:
            The saved agent with updated metadata

        Raises:
            InfrastructureException: If save operation fails
        """
        pass

    @abstractmethod
    async def find_by_id(self, agent_id: UUID) -> Optional[Agent]:
        """
        Find an agent by its ID.

        Args:
            agent_id: The agent's unique identifier

        Returns:
            The agent if found, None otherwise

        Raises:
            InfrastructureException: If query operation fails
        """
        pass

    @abstractmethod
    async def find_by_type(self, agent_type: AgentType) -> List[Agent]:
        """
        Find all agents of a specific type.

        Args:
            agent_type: The type of agents to find

        Returns:
            List of agents matching the type

        Raises:
            InfrastructureException: If query operation fails
        """
        pass

    @abstractmethod
    async def find_available(self) -> List[Agent]:
        """
        Find all available agents (not busy).

        Returns:
            List of available agents

        Raises:
            InfrastructureException: If query operation fails
        """
        pass

    @abstractmethod
    async def update_state(self, agent_id: UUID, state: AgentState) -> bool:
        """
        Update an agent's state.

        Args:
            agent_id: The agent's unique identifier
            state: The new state

        Returns:
            True if update was successful, False otherwise

        Raises:
            InfrastructureException: If update operation fails
        """
        pass

    @abstractmethod
    async def update_performance_metrics(
        self,
        agent_id: UUID,
        metrics: Dict[str, Any]
    ) -> bool:
        """
        Update an agent's performance metrics.

        Args:
            agent_id: The agent's unique identifier
            metrics: Dictionary of metrics to update

        Returns:
            True if update was successful, False otherwise

        Raises:
            InfrastructureException: If update operation fails
        """
        pass

    @abstractmethod
    async def delete(self, agent_id: UUID) -> bool:
        """
        Delete an agent from the repository.

        Args:
            agent_id: The agent's unique identifier

        Returns:
            True if deletion was successful, False otherwise

        Raises:
            InfrastructureException: If deletion operation fails
        """
        pass

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get repository statistics.

        Returns:
            Dictionary containing repository statistics

        Raises:
            InfrastructureException: If query operation fails
        """
        pass