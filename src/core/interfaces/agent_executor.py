from src.agents.advanced_agent_fsm import Agent
# TODO: Fix undefined variables: ABC, Any, Dict, Optional, UUID, abstractmethod

"""
from abc import abstractmethod
from src.gaia_components.multi_agent_orchestrator import Agent

Agent executor interface for executing agent tasks.
"""

from typing import Any
from typing import Optional

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from uuid import UUID

from src.core.entities.agent import Agent
from src.core.entities.message import Message

class AgentExecutor(ABC):
    """
    Abstract interface for agent execution operations.

    This interface defines the contract that all agent executor
    implementations must follow, ensuring consistency across
    different execution strategies.
    """

    @abstractmethod
    async def execute_agent(self, agent: Agent, message: Message) -> Dict[str, Any]:
        """
        Execute an agent with a given message.

        Args:
            agent: The agent to execute
            message: The message to process

        Returns:
            Dictionary containing the execution result

        Raises:
            ApplicationException: If execution fails
        """
        pass

    @abstractmethod
    async def execute_agent_by_id(self, agent_id: UUID, message: Message) -> Dict[str, Any]:
        """
        Execute an agent by ID with a given message.

        Args:
            agent_id: The agent's unique identifier
            message: The message to process

        Returns:
            Dictionary containing the execution result

        Raises:
            ApplicationException: If execution fails
        """
        pass

    @abstractmethod
    async def execute_agent_by_type(self, agent_type: str, message: Message) -> Dict[str, Any]:
        """
        Execute an agent by type with a given message.

        Args:
            agent_type: The type of agent to execute
            message: The message to process

        Returns:
            Dictionary containing the execution result

        Raises:
            ApplicationException: If execution fails
        """
        pass

    @abstractmethod
    async def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """
        Get the status of an execution.

        Args:
            execution_id: The execution identifier

        Returns:
            Dictionary containing the execution status

        Raises:
            ApplicationException: If status retrieval fails
        """
        pass

    @abstractmethod
    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an ongoing execution.

        Args:
            execution_id: The execution identifier

        Returns:
            True if cancellation was successful, False otherwise

        Raises:
            ApplicationException: If cancellation fails
        """
        pass

    @abstractmethod
    async def get_execution_metrics(self, agent_id: Optional[UUID] = None) -> Dict[str, Any]:
        """
        Get execution metrics for agents.

        Args:
            agent_id: Optional agent ID to filter metrics

        Returns:
            Dictionary containing execution metrics

        Raises:
            ApplicationException: If metrics retrieval fails
        """
        pass