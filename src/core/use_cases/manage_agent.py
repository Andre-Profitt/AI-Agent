from migrations.env import configuration
from performance_dashboard import stats
from tests.load_test import success

from src.agents.enhanced_fsm import state
from src.api_server import agent_factory
from src.core.entities.agent import Agent
from src.core.use_cases.manage_agent import all_agents
from src.core.use_cases.manage_agent import core_agent
from src.core.use_cases.manage_agent import factory_agent
from src.core.use_cases.manage_agent import factory_agents
from src.core.use_cases.manage_agent import repo_agent
from src.core.use_cases.manage_agent import repo_agents
from src.core.use_cases.manage_agent import saved_agent
from src.core.use_cases.manage_agent import updated_agent
from src.database.models import agent_id
from src.database.models import agent_type
from src.main import logging_service
from src.tools_introspection import description
from src.tools_introspection import name

from src.agents.advanced_agent_fsm import AgentType

from src.agents.advanced_agent_fsm import Agent

from src.agents.advanced_agent_fsm import AgentFactory
# TODO: Fix undefined variables: Any, Dict, Optional, UUID, agent_factory, agent_id, agent_info, agent_repository, agent_type, all_agents, configuration, core_agent, description, e, factory_agent, factory_agents, logging, logging_service, name, repo_agent, repo_agents, saved_agent, state, stats, success, updated_agent
from tests.test_gaia_agent import agent

from src.core.entities.agent import AgentType


"""
from typing import Dict
from src.agents.multi_agent_system import AgentState
from src.gaia_components.multi_agent_orchestrator import Agent
from src.infrastructure.agents.agent_factory import AgentFactory
from src.infrastructure.agents.agent_factory import AgentType
from src.infrastructure.logging.logging_service import LoggingService
from src.shared.exceptions import ValidationException
# TODO: Fix undefined variables: agent, agent_factory, agent_id, agent_info, agent_repository, agent_type, all_agents, configuration, core_agent, description, e, factory_agent, factory_agents, logging_service, name, repo_agent, repo_agents, saved_agent, self, state, stats, success, updated_agent

Use case for managing AI agents with factory support.
"""

from typing import Optional
from typing import Any

from uuid import UUID
import logging

from src.core.entities.agent import Agent, AgentType, AgentState
from src.core.interfaces.agent_repository import AgentRepository
from src.core.interfaces.logging_service import LoggingService
from src.application.agents.agent_factory import AgentFactory
from src.shared.exceptions import DomainException, ValidationException

class ManageAgentUseCase:
    """
    Use case for managing AI agents with factory support.

    This use case handles agent creation, updates, deletion,
    and lifecycle management operations using the agent factory.
    """

    def __init__(
        self,
        agent_repository: AgentRepository,
        agent_factory: AgentFactory,
        logging_service: LoggingService
    ):
        self.agent_repository = agent_repository
        self.agent_factory = agent_factory
        self.logging_service = logging_service
        self.logger = logging.getLogger(__name__)

    async def create_agent(
        self,
        agent_type: AgentType,
        name: str,
        description: Optional[str] = None,
        configuration: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new agent using the factory.

        Args:
            agent_type: Type of agent to create
            name: Agent name
            description: Optional agent description
            configuration: Optional agent configuration

        Returns:
            Dictionary containing the created agent information
        """
        try:
            # Validate input
            if not name or not name.strip():
                raise ValidationException("Agent name cannot be empty")

            # Create agent using factory
            agent = await self.agent_factory.create_agent(
                agent_type=agent_type,
                name=name,
                config=configuration or {}
            )

            # Create core agent entity for repository
            core_agent = Agent(
                agent_type=agent_type,
                name=name,
                description=description or f"{agent_type.value} agent",
                configuration=configuration or {},
                state=AgentState.READY
            )

            # Save to repository
            saved_agent = await self.agent_repository.save(core_agent)

            # Log creation
            await self.logging_service.log_info(
                "agent_created",
                f"Created {agent_type.value} agent: {name} (ID: {agent.agent_id})",
                {"agent_id": str(saved_agent.id), "agent_type": agent_type.value}
            )

            return {
                "success": True,
                "agent_id": agent.agent_id,
                "name": agent.name,
                "type": agent_type.value,
                "status": agent.status.name,
                "factory_agent": True
            }

        except Exception as e:
            self.logger.error("Failed to create agent: {}", extra={"str_e_": str(e)})
            await self.logging_service.log_error(
                "agent_creation_failed",
                str(e),
                {"agent_type": agent_type.value, "name": name}
            )
            return {"success": False, "error": str(e)}

    async def update_agent(
        self,
        agent_id: UUID,
        name: Optional[str] = None,
        description: Optional[str] = None,
        configuration: Optional[Dict[str, Any]] = None,
        state: Optional[AgentState] = None
    ) -> Dict[str, Any]:
        """
        Update an existing agent.

        Args:
            agent_id: ID of the agent to update
            name: New agent name
            description: New agent description
            configuration: New agent configuration
            state: New agent state

        Returns:
            Dictionary containing the update result
        """
        try:
            # Find agent
            agent = await self.agent_repository.find_by_id(agent_id)
            if not agent:
                raise DomainException(f"Agent {agent_id} not found")

            # Update fields
            if name is not None:
                if not name.strip():
                    raise ValidationException("Agent name cannot be empty")
                agent.name = name

            if description is not None:
                agent.description = description

            if configuration is not None:
                agent.configuration = configuration

            if state is not None:
                agent.state = state

            # Save updated agent
            updated_agent = await self.agent_repository.save(agent)

            # Log update
            await self.logging_service.log_info(
                "agent_updated",
                f"Updated agent {agent_id}",
                {"agent_id": str(agent_id)}
            )

            return {
                "success": True,
                "agent_id": str(updated_agent.id),
                "name": updated_agent.name,
                "state": updated_agent.state.value
            }

        except Exception as e:
            self.logger.error("Failed to update agent {}: {}", extra={"agent_id": agent_id, "str_e_": str(e)})
            await self.logging_service.log_error(
                "agent_update_failed",
                str(e),
                {"agent_id": str(agent_id)}
            )
            return {"success": False, "error": str(e)}

    async def delete_agent(self, agent_id: UUID) -> Dict[str, Any]:
        """
        Delete an agent.

        Args:
            agent_id: ID of the agent to delete

        Returns:
            Dictionary containing the deletion result
        """
        try:
            # Check if agent exists in repository
            agent = await self.agent_repository.find_by_id(agent_id)
            if not agent:
                raise DomainException(f"Agent {agent_id} not found")

            # Check if agent exists in factory cache
            factory_agent = self.agent_factory.get_agent(str(agent_id))
            if factory_agent:
                # Shutdown factory agent
                await factory_agent.shutdown()

            # Delete from repository
            success = await self.agent_repository.delete(agent_id)
            if not success:
                raise DomainException(f"Failed to delete agent {agent_id}")

            # Log deletion
            await self.logging_service.log_info(
                "agent_deleted",
                f"Deleted agent {agent_id}",
                {"agent_id": str(agent_id)}
            )

            return {"success": True, "agent_id": str(agent_id)}

        except Exception as e:
            self.logger.error("Failed to delete agent {}: {}", extra={"agent_id": agent_id, "str_e_": str(e)})
            await self.logging_service.log_error(
                "agent_deletion_failed",
                str(e),
                {"agent_id": str(agent_id)}
            )
            return {"success": False, "error": str(e)}

    async def get_agent(self, agent_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get agent information from factory or repository.

        Args:
            agent_id: ID of the agent to retrieve

        Returns:
            Dictionary containing the agent information
        """
        try:
            # Try factory first
            factory_agent = self.agent_factory.get_agent(str(agent_id))
            if factory_agent:
                return {
                    "success": True,
                    "agent": {
                        "id": factory_agent.agent_id,
                        "name": factory_agent.name,
                        "type": type(factory_agent).__name__,
                        "status": factory_agent.status.name,
                        "source": "factory"
                    }
                }

            # Try repository
            repo_agent = await self.agent_repository.find_by_id(agent_id)
            if repo_agent:
                return {
                    "success": True,
                    "agent": {
                        "id": str(repo_agent.id),
                        "name": repo_agent.name,
                        "description": repo_agent.description,
                        "agent_type": repo_agent.agent_type.value,
                        "state": repo_agent.state.value,
                        "configuration": repo_agent.configuration,
                        "performance_metrics": getattr(repo_agent, 'performance_metrics', {}),
                        "created_at": repo_agent.created_at.isoformat() if repo_agent.created_at else None,
                        "updated_at": repo_agent.updated_at.isoformat() if repo_agent.updated_at else None,
                        "source": "repository"
                    }
                }

            return {"success": False, "error": f"Agent {agent_id} not found"}

        except Exception as e:
            self.logger.error("Failed to get agent {}: {}", extra={"agent_id": agent_id, "str_e_": str(e)})
            return {"success": False, "error": str(e)}

    async def list_agents(self, agent_type: Optional[AgentType] = None) -> Dict[str, Any]:
        """
        List all agents from factory cache and repository.

        Args:
            agent_type: Optional agent type filter

        Returns:
            Dictionary containing the list of agents
        """
        try:
            # Get from factory cache
            factory_agents = self.agent_factory.list_agents()

            # Get from repository
            if agent_type:
                repo_agents = await self.agent_repository.find_by_type(agent_type)
            else:
                # Get all agents from repository
                repo_agents = []
                # Note: We need to add a find_all method to the repository interface
                # For now, we'll use the statistics to get agent info
                stats = await self.agent_repository.get_statistics()
                repo_agents = stats.get("agents", [])

            # Merge and deduplicate
            all_agents = {}

            # Add factory agents
            for agent_info in factory_agents:
                all_agents[agent_info["id"]] = {
                    **agent_info,
                    "source": "factory"
                }

            # Add repository agents
            for agent in repo_agents:
                agent_id = str(agent.id) if hasattr(agent, 'id') else agent.get('id', 'unknown')
                if agent_id not in all_agents:
                    all_agents[agent_id] = {
                        "id": agent_id,
                        "name": agent.name if hasattr(agent, 'name') else agent.get('name', 'Unknown'),
                        "type": agent.agent_type.value if hasattr(agent, 'agent_type') else agent.get('type', 'unknown'),
                        "status": agent.state.value if hasattr(agent, 'state') else agent.get('status', 'unknown'),
                        "source": "repository"
                    }

            return {
                "success": True,
                "agents": list(all_agents.values()),
                "count": len(all_agents)
            }

        except Exception as e:
            self.logger.error("Failed to list agents: {}", extra={"str_e_": str(e)})
            return {"success": False, "error": str(e)}

    async def get_agent_statistics(self) -> Dict[str, Any]:
        """
        Get agent repository statistics.

        Returns:
            Dictionary containing agent statistics
        """
        try:
            stats = await self.agent_repository.get_statistics()

            # Add factory statistics
            factory_agents = self.agent_factory.list_agents()
            stats["factory_agents"] = len(factory_agents)
            stats["factory_agent_types"] = {}

            for agent_info in factory_agents:
                agent_type = agent_info.get("type", "unknown")
                stats["factory_agent_types"][agent_type] = stats["factory_agent_types"].get(agent_type, 0) + 1

            return {"success": True, "statistics": stats}

        except Exception as e:
            self.logger.error("Failed to get agent statistics: {}", extra={"str_e_": str(e)})
            return {"success": False, "error": str(e)}

    async def shutdown_all_agents(self) -> Dict[str, Any]:
        """
        Shutdown all agents in the factory cache.

        Returns:
            Dictionary containing shutdown results
        """
        try:
            await self.agent_factory.shutdown_all()

            await self.logging_service.log_info(
                "agents_shutdown",
                "All factory agents shut down successfully"
            )

            return {
                "success": True,
                "message": "All agents shut down successfully"
            }

        except Exception as e:
            self.logger.error("Failed to shutdown all agents: {}", extra={"str_e_": str(e)})
            await self.logging_service.log_error(
                "agents_shutdown_failed",
                str(e)
            )
            return {"success": False, "error": str(e)}