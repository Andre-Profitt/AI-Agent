"""
In-memory implementation of the agent repository.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
import logging
from datetime import datetime

from src.core.interfaces.agent_repository import AgentRepository
from src.core.entities.agent import Agent, AgentType, AgentState
from src.shared.exceptions import InfrastructureException


class InMemoryAgentRepository(AgentRepository):
    """
    In-memory implementation of the agent repository.
    
    This implementation stores agents in memory and is suitable
    for development and testing purposes.
    """
    
    def __init__(self):
        self._agents: Dict[UUID, Agent] = {}
        self.logger = logging.getLogger(__name__)
    
    async def save(self, agent: Agent) -> Agent:
        """Save an agent to the repository."""
        try:
            if not agent.id:
                agent.id = uuid4()
                agent.created_at = datetime.utcnow()
            
            agent.updated_at = datetime.utcnow()
            self._agents[agent.id] = agent
            
            self.logger.debug(f"Saved agent {agent.id} of type {agent.agent_type}")
            return agent
            
        except Exception as e:
            self.logger.error(f"Failed to save agent: {str(e)}")
            raise InfrastructureException(f"Failed to save agent: {str(e)}")
    
    async def find_by_id(self, agent_id: UUID) -> Optional[Agent]:
        """Find an agent by its ID."""
        try:
            agent = self._agents.get(agent_id)
            if agent:
                self.logger.debug(f"Found agent {agent_id}")
            return agent
            
        except Exception as e:
            self.logger.error(f"Failed to find agent {agent_id}: {str(e)}")
            raise InfrastructureException(f"Failed to find agent {agent_id}: {str(e)}")
    
    async def find_by_type(self, agent_type: AgentType) -> List[Agent]:
        """Find all agents of a specific type."""
        try:
            agents = [agent for agent in self._agents.values() if agent.agent_type == agent_type]
            self.logger.debug(f"Found {len(agents)} agents of type {agent_type}")
            return agents
            
        except Exception as e:
            self.logger.error(f"Failed to find agents of type {agent_type}: {str(e)}")
            raise InfrastructureException(f"Failed to find agents of type {agent_type}: {str(e)}")
    
    async def find_available(self) -> List[Agent]:
        """Find all available agents (not busy)."""
        try:
            available_agents = [
                agent for agent in self._agents.values() 
                if agent.state == AgentState.IDLE or agent.state == AgentState.READY
            ]
            self.logger.debug(f"Found {len(available_agents)} available agents")
            return available_agents
            
        except Exception as e:
            self.logger.error(f"Failed to find available agents: {str(e)}")
            raise InfrastructureException(f"Failed to find available agents: {str(e)}")
    
    async def update_state(self, agent_id: UUID, state: AgentState) -> bool:
        """Update an agent's state."""
        try:
            if agent_id not in self._agents:
                self.logger.warning(f"Agent {agent_id} not found for state update")
                return False
            
            self._agents[agent_id].state = state
            self._agents[agent_id].updated_at = datetime.utcnow()
            
            self.logger.debug(f"Updated agent {agent_id} state to {state}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update agent {agent_id} state: {str(e)}")
            raise InfrastructureException(f"Failed to update agent {agent_id} state: {str(e)}")
    
    async def update_performance_metrics(self, agent_id: UUID, metrics: Dict[str, Any]) -> bool:
        """Update an agent's performance metrics."""
        try:
            if agent_id not in self._agents:
                self.logger.warning(f"Agent {agent_id} not found for metrics update")
                return False
            
            agent = self._agents[agent_id]
            if not hasattr(agent, 'performance_metrics'):
                agent.performance_metrics = {}
            
            agent.performance_metrics.update(metrics)
            agent.updated_at = datetime.utcnow()
            
            self.logger.debug(f"Updated performance metrics for agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update metrics for agent {agent_id}: {str(e)}")
            raise InfrastructureException(f"Failed to update metrics for agent {agent_id}: {str(e)}")
    
    async def delete(self, agent_id: UUID) -> bool:
        """Delete an agent from the repository."""
        try:
            if agent_id not in self._agents:
                self.logger.warning(f"Agent {agent_id} not found for deletion")
                return False
            
            del self._agents[agent_id]
            self.logger.debug(f"Deleted agent {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete agent {agent_id}: {str(e)}")
            raise InfrastructureException(f"Failed to delete agent {agent_id}: {str(e)}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        try:
            total_agents = len(self._agents)
            agents_by_type = {}
            agents_by_state = {}
            
            for agent in self._agents.values():
                # Count by type
                agent_type = agent.agent_type.value
                agents_by_type[agent_type] = agents_by_type.get(agent_type, 0) + 1
                
                # Count by state
                agent_state = agent.state.value
                agents_by_state[agent_state] = agents_by_state.get(agent_state, 0) + 1
            
            stats = {
                "total_agents": total_agents,
                "agents_by_type": agents_by_type,
                "agents_by_state": agents_by_state,
                "created_at": datetime.utcnow().isoformat()
            }
            
            self.logger.debug(f"Generated statistics: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {str(e)}")
            raise InfrastructureException(f"Failed to get statistics: {str(e)}") 