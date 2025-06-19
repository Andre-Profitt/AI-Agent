"""
Use case for managing AI agents.
"""

from typing import Dict, Any, Optional, List
from uuid import UUID
import logging

from src.core.entities.agent import Agent, AgentType, AgentState
from src.core.interfaces.agent_repository import AgentRepository
from src.core.interfaces.logging_service import LoggingService
from src.shared.exceptions import DomainException, ValidationException


class ManageAgentUseCase:
    """
    Use case for managing AI agents.
    
    This use case handles agent creation, updates, deletion,
    and lifecycle management operations.
    """
    
    def __init__(
        self,
        agent_repository: AgentRepository,
        logging_service: LoggingService
    ):
        self.agent_repository = agent_repository
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
        Create a new agent.
        
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
            
            # Create agent entity
            agent = Agent(
                agent_type=agent_type,
                name=name,
                description=description or f"{agent_type.value} agent",
                configuration=configuration or {},
                state=AgentState.READY
            )
            
            # Save agent
            saved_agent = await self.agent_repository.save(agent)
            
            # Log creation
            await self.logging_service.log_info(
                "agent_created",
                f"Created agent {saved_agent.id} of type {agent_type.value}",
                {"agent_id": str(saved_agent.id), "agent_type": agent_type.value}
            )
            
            return {
                "success": True,
                "agent_id": str(saved_agent.id),
                "agent_type": saved_agent.agent_type.value,
                "name": saved_agent.name,
                "state": saved_agent.state.value
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create agent: {str(e)}")
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
            self.logger.error(f"Failed to update agent {agent_id}: {str(e)}")
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
            # Check if agent exists
            agent = await self.agent_repository.find_by_id(agent_id)
            if not agent:
                raise DomainException(f"Agent {agent_id} not found")
            
            # Delete agent
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
            self.logger.error(f"Failed to delete agent {agent_id}: {str(e)}")
            await self.logging_service.log_error(
                "agent_deletion_failed",
                str(e),
                {"agent_id": str(agent_id)}
            )
            return {"success": False, "error": str(e)}
    
    async def get_agent(self, agent_id: UUID) -> Dict[str, Any]:
        """
        Get agent information.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            Dictionary containing the agent information
        """
        try:
            agent = await self.agent_repository.find_by_id(agent_id)
            if not agent:
                return {"success": False, "error": f"Agent {agent_id} not found"}
            
            return {
                "success": True,
                "agent": {
                    "id": str(agent.id),
                    "name": agent.name,
                    "description": agent.description,
                    "agent_type": agent.agent_type.value,
                    "state": agent.state.value,
                    "configuration": agent.configuration,
                    "performance_metrics": getattr(agent, 'performance_metrics', {}),
                    "created_at": agent.created_at.isoformat() if agent.created_at else None,
                    "updated_at": agent.updated_at.isoformat() if agent.updated_at else None
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get agent {agent_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def list_agents(self, agent_type: Optional[AgentType] = None) -> Dict[str, Any]:
        """
        List all agents, optionally filtered by type.
        
        Args:
            agent_type: Optional agent type filter
            
        Returns:
            Dictionary containing the list of agents
        """
        try:
            if agent_type:
                agents = await self.agent_repository.find_by_type(agent_type)
            else:
                # Get all agents (we'll need to add this method to the repository)
                agents = list(await self.agent_repository.get_statistics().get("agents", []))
            
            agent_list = []
            for agent in agents:
                agent_list.append({
                    "id": str(agent.id),
                    "name": agent.name,
                    "description": agent.description,
                    "agent_type": agent.agent_type.value,
                    "state": agent.state.value,
                    "created_at": agent.created_at.isoformat() if agent.created_at else None
                })
            
            return {
                "success": True,
                "agents": agent_list,
                "count": len(agent_list)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to list agents: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_agent_statistics(self) -> Dict[str, Any]:
        """
        Get agent repository statistics.
        
        Returns:
            Dictionary containing agent statistics
        """
        try:
            stats = await self.agent_repository.get_statistics()
            return {"success": True, "statistics": stats}
            
        except Exception as e:
            self.logger.error(f"Failed to get agent statistics: {str(e)}")
            return {"success": False, "error": str(e)} 