"""
Agent Factory

This module provides a factory for creating different types of agents
based on configuration and requirements.
"""

import logging
from typing import Dict, Any, Optional, List, Type
from uuid import uuid4
from enum import Enum

from src.unified_architecture.core import IUnifiedAgent, AgentCapability, AgentMetadata
from src.infrastructure.agents.concrete_agents import (
    FSMReactAgentImpl, NextGenAgentImpl, CrewAgentImpl, SpecializedAgentImpl,
    AgentConfig
)


class AgentType(str, Enum):
    """Supported agent types"""
    FSM_REACT = "fsm_react"
    NEXT_GEN = "next_gen"
    CREW = "crew"
    SPECIALIZED = "specialized"


class AgentFactory:
    """
    Factory for creating different types of agents
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._agent_registry: Dict[str, Type[IUnifiedAgent]] = {
            AgentType.FSM_REACT: FSMReactAgentImpl,
            AgentType.NEXT_GEN: NextGenAgentImpl,
            AgentType.CREW: CrewAgentImpl,
            AgentType.SPECIALIZED: SpecializedAgentImpl
        }
        self._created_agents: Dict[str, IUnifiedAgent] = {}
    
    def register_agent_type(self, agent_type: str, agent_class: Type[IUnifiedAgent]):
        """Register a new agent type"""
        self._agent_registry[agent_type] = agent_class
        self.logger.info(f"Registered agent type: {agent_type}")
    
    def get_available_agent_types(self) -> List[str]:
        """Get list of available agent types"""
        return list(self._agent_registry.keys())
    
    async def create_agent(
        self,
        agent_type: str,
        config: Optional[AgentConfig] = None,
        **kwargs
    ) -> IUnifiedAgent:
        """
        Create an agent of the specified type
        
        Args:
            agent_type: Type of agent to create
            config: Agent configuration
            **kwargs: Additional arguments for agent creation
            
        Returns:
            Created agent instance
            
        Raises:
            ValueError: If agent type is not supported
        """
        if agent_type not in self._agent_registry:
            raise ValueError(f"Unsupported agent type: {agent_type}")
        
        try:
            agent_class = self._agent_registry[agent_type]
            
            # Create agent instance
            if agent_type == AgentType.SPECIALIZED:
                domain = kwargs.get("domain", "general")
                agent = agent_class(domain=domain, config=config)
            else:
                agent = agent_class(config=config)
            
            # Initialize agent
            success = await agent.initialize()
            if not success:
                raise RuntimeError(f"Failed to initialize {agent_type} agent")
            
            # Register created agent
            agent_id = str(agent.agent_id)
            self._created_agents[agent_id] = agent
            
            self.logger.info(f"Created {agent_type} agent: {agent_id}")
            return agent
            
        except Exception as e:
            self.logger.error(f"Failed to create {agent_type} agent: {e}")
            raise
    
    async def create_agent_for_capabilities(
        self,
        required_capabilities: List[AgentCapability],
        config: Optional[AgentConfig] = None,
        **kwargs
    ) -> IUnifiedAgent:
        """
        Create an agent that best matches the required capabilities
        
        Args:
            required_capabilities: List of required capabilities
            config: Agent configuration
            **kwargs: Additional arguments
            
        Returns:
            Best matching agent instance
        """
        # Score each agent type based on capability match
        agent_scores = {}
        
        for agent_type in self._agent_registry.keys():
            # Create temporary agent to check capabilities
            temp_agent = await self.create_agent(agent_type, config, **kwargs)
            agent_capabilities = await temp_agent.get_capabilities()
            
            # Calculate match score
            score = 0
            for required_cap in required_capabilities:
                if required_cap in agent_capabilities:
                    score += 1
            
            agent_scores[agent_type] = score
        
        # Find best matching agent type
        best_agent_type = max(agent_scores.items(), key=lambda x: x[1])[0]
        
        self.logger.info(f"Selected {best_agent_type} for capabilities: {required_capabilities}")
        
        # Create the best matching agent
        return await self.create_agent(best_agent_type, config, **kwargs)
    
    async def create_agent_team(
        self,
        team_config: List[Dict[str, Any]],
        config: Optional[AgentConfig] = None
    ) -> List[IUnifiedAgent]:
        """
        Create a team of agents based on configuration
        
        Args:
            team_config: List of agent configurations
            config: Base configuration for all agents
            
        Returns:
            List of created agents
        """
        agents = []
        
        for agent_spec in team_config:
            agent_type = agent_spec.get("type")
            agent_kwargs = agent_spec.get("kwargs", {})
            
            agent = await self.create_agent(agent_type, config, **agent_kwargs)
            agents.append(agent)
        
        self.logger.info(f"Created agent team with {len(agents)} agents")
        return agents
    
    def get_agent(self, agent_id: str) -> Optional[IUnifiedAgent]:
        """Get a previously created agent by ID"""
        return self._created_agents.get(agent_id)
    
    def get_all_agents(self) -> Dict[str, IUnifiedAgent]:
        """Get all created agents"""
        return self._created_agents.copy()
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """
        Destroy an agent and clean up resources
        
        Args:
            agent_id: ID of agent to destroy
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self._created_agents:
            return False
        
        try:
            agent = self._created_agents[agent_id]
            # Perform cleanup if agent has shutdown method
            if hasattr(agent, 'shutdown'):
                await agent.shutdown()
            
            del self._created_agents[agent_id]
            self.logger.info(f"Destroyed agent: {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to destroy agent {agent_id}: {e}")
            return False
    
    async def destroy_all_agents(self):
        """Destroy all created agents"""
        agent_ids = list(self._created_agents.keys())
        
        for agent_id in agent_ids:
            await self.destroy_agent(agent_id)
        
        self.logger.info("Destroyed all agents")
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about created agents"""
        stats = {
            "total_agents": len(self._created_agents),
            "agent_types": {},
            "status_counts": {}
        }
        
        for agent_id, agent in self._created_agents.items():
            # Count by type
            agent_type = type(agent).__name__
            stats["agent_types"][agent_type] = stats["agent_types"].get(agent_type, 0) + 1
            
            # Count by status
            status = agent.status.value if hasattr(agent.status, 'value') else str(agent.status)
            stats["status_counts"][status] = stats["status_counts"].get(status, 0) + 1
        
        return stats


# Global agent factory instance
_agent_factory: Optional[AgentFactory] = None


def get_agent_factory() -> AgentFactory:
    """Get the global agent factory instance"""
    global _agent_factory
    if _agent_factory is None:
        _agent_factory = AgentFactory()
    return _agent_factory


async def create_default_agents() -> Dict[str, IUnifiedAgent]:
    """
    Create a set of default agents for the platform
    
    Returns:
        Dictionary mapping agent names to agent instances
    """
    factory = get_agent_factory()
    
    default_agents = {}
    
    # Create FSM React Agent
    fsm_agent = await factory.create_agent(AgentType.FSM_REACT)
    default_agents["fsm_react"] = fsm_agent
    
    # Create Next Gen Agent
    next_gen_agent = await factory.create_agent(AgentType.NEXT_GEN)
    default_agents["next_gen"] = next_gen_agent
    
    # Create Crew Agent
    crew_agent = await factory.create_agent(AgentType.CREW)
    default_agents["crew"] = crew_agent
    
    # Create specialized agents
    domains = ["data_analysis", "code_generation", "research", "creative"]
    for domain in domains:
        specialized_agent = await factory.create_agent(
            AgentType.SPECIALIZED, 
            domain=domain
        )
        default_agents[f"specialized_{domain}"] = specialized_agent
    
    return default_agents 