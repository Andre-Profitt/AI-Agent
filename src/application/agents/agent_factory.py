"""
Agent factory for creating different types of agents.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from uuid import uuid4

from src.core.entities.agent import AgentType
from src.infrastructure.agents.concrete_agents import (
    FSMReactAgentImpl, NextGenAgentImpl, CrewAgentImpl, SpecializedAgentImpl
)
from src.application.agents.base_agent import BaseAgent
from src.utils.base_tool import get_enhanced_tools

logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory for creating different types of agents"""
    
    def __init__(self):
        self.created_agents: Dict[str, BaseAgent] = {}
        self.logger = logging.getLogger(__name__)
    
    async def create_agent(
        self,
        agent_type: AgentType,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseAgent:
        """
        Create an agent of the specified type.
        
        Args:
            agent_type: Type of agent to create
            name: Name for the agent
            config: Configuration for the agent
            
        Returns:
            Created agent instance
        """
        config = config or {}
        agent_id = str(uuid4())
        
        try:
            if agent_type == AgentType.FSM_REACT:
                # Get tools for FSM agent
                tools = config.get("tools", [])
                if not tools:
                    tools = get_enhanced_tools()
                
                agent = FSMReactAgentImpl(
                    agent_id=agent_id,
                    name=name,
                    tools=tools
                )
                
            elif agent_type == AgentType.NEXT_GEN:
                model_config = config.get("model_config", {})
                agent = NextGenAgentImpl(
                    agent_id=agent_id,
                    name=name,
                    model_config=model_config
                )
                
            elif agent_type == AgentType.CREW:
                role = config.get("role", "general")
                agent = CrewAgentImpl(
                    agent_id=agent_id,
                    name=name,
                    role=role
                )
                
            elif agent_type == AgentType.SPECIALIZED:
                specialization = config.get("specialization", "general")
                agent = SpecializedAgentImpl(
                    agent_id=agent_id,
                    name=name,
                    specialization=specialization
                )
                
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            # Initialize the agent
            init_config = {
                "agent_type": agent_type.value,
                "model_config": config.get("model_config", {}),
                "max_concurrent_tasks": config.get("max_concurrent_tasks", 5),
                "task_timeout": config.get("task_timeout", 300),
                "enable_learning": config.get("enable_learning", True),
                "enable_collaboration": config.get("enable_collaboration", True),
                "memory_size": config.get("memory_size", 1000),
                **config
            }
            
            success = await agent.initialize(init_config)
            if not success:
                raise RuntimeError(f"Failed to initialize agent {name}")
            
            # Store the agent
            self.created_agents[agent_id] = agent
            
            self.logger.info(f"Created {agent_type.value} agent: {name} ({agent_id})")
            return agent
            
        except Exception as e:
            self.logger.error(f"Failed to create agent {name}: {e}")
            raise
    
    async def create_fsm_agent(
        self,
        name: str,
        tools: Optional[List[Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> FSMReactAgentImpl:
        """Create an FSM React agent with specified tools"""
        agent_config = config or {}
        if tools:
            agent_config["tools"] = tools
        
        return await self.create_agent(AgentType.FSM_REACT, name, agent_config)
    
    async def create_next_gen_agent(
        self,
        name: str,
        model_config: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> NextGenAgentImpl:
        """Create a Next Generation agent with learning capabilities"""
        agent_config = config or {}
        if model_config:
            agent_config["model_config"] = model_config
        
        return await self.create_agent(AgentType.NEXT_GEN, name, agent_config)
    
    async def create_crew_agent(
        self,
        name: str,
        role: str = "general",
        config: Optional[Dict[str, Any]] = None
    ) -> CrewAgentImpl:
        """Create a Crew agent for team collaboration"""
        agent_config = config or {}
        agent_config["role"] = role
        
        return await self.create_agent(AgentType.CREW, name, agent_config)
    
    async def create_specialized_agent(
        self,
        name: str,
        specialization: str,
        config: Optional[Dict[str, Any]] = None
    ) -> SpecializedAgentImpl:
        """Create a specialized agent for domain-specific tasks"""
        agent_config = config or {}
        agent_config["specialization"] = specialization
        
        return await self.create_agent(AgentType.SPECIALIZED, name, agent_config)
    
    async def create_agent_team(
        self,
        team_config: List[Dict[str, Any]]
    ) -> List[BaseAgent]:
        """Create a team of agents based on configuration"""
        agents = []
        
        for member_config in team_config:
            agent_type = AgentType(member_config["type"])
            name = member_config["name"]
            config = member_config.get("config", {})
            
            agent = await self.create_agent(agent_type, name, config)
            agents.append(agent)
        
        self.logger.info(f"Created agent team with {len(agents)} members")
        return agents
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID"""
        return self.created_agents.get(agent_id)
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Get all created agents"""
        return list(self.created_agents.values())
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """Get all agents of a specific type"""
        return [
            agent for agent in self.created_agents.values()
            if hasattr(agent, 'agent_type') and agent.agent_type == agent_type
        ]
    
    async def shutdown_agent(self, agent_id: str) -> bool:
        """Shutdown a specific agent"""
        agent = self.created_agents.get(agent_id)
        if not agent:
            self.logger.warning(f"Agent {agent_id} not found")
            return False
        
        try:
            success = await agent.shutdown()
            if success:
                del self.created_agents[agent_id]
                self.logger.info(f"Agent {agent_id} shut down successfully")
            return success
        except Exception as e:
            self.logger.error(f"Failed to shutdown agent {agent_id}: {e}")
            return False
    
    async def shutdown_all(self):
        """Shutdown all created agents"""
        self.logger.info(f"Shutting down {len(self.created_agents)} agents")
        
        shutdown_tasks = []
        for agent_id, agent in self.created_agents.items():
            task = self.shutdown_agent(agent_id)
            shutdown_tasks.append(task)
        
        # Wait for all shutdowns to complete
        results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        success_count = sum(1 for result in results if result is True)
        self.logger.info(f"Successfully shut down {success_count}/{len(self.created_agents)} agents")
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics about created agents"""
        stats = {
            "total_agents": len(self.created_agents),
            "agents_by_type": {},
            "available_agents": 0,
            "busy_agents": 0,
            "error_agents": 0
        }
        
        for agent in self.created_agents.values():
            # Count by type
            agent_type = getattr(agent, 'agent_type', 'unknown')
            if agent_type not in stats["agents_by_type"]:
                stats["agents_by_type"][agent_type] = 0
            stats["agents_by_type"][agent_type] += 1
            
            # Count by status
            status = getattr(agent, 'status', None)
            if status:
                if status.name in ['AVAILABLE', 'IDLE']:
                    stats["available_agents"] += 1
                elif status.name == 'BUSY':
                    stats["busy_agents"] += 1
                elif status.name == 'ERROR':
                    stats["error_agents"] += 1
        
        return stats


# Example usage function
async def create_agent_team():
    """Example: Create a team of different agent types"""
    
    factory = AgentFactory()
    
    # Create FSM React Agent
    fsm_agent = await factory.create_agent(
        AgentType.FSM_REACT,
        "FSM_Agent_1",
        {"tools": ["web_search", "calculator", "python_repl"]}
    )
    
    # Create Next Gen Agent
    next_gen_agent = await factory.create_agent(
        AgentType.NEXT_GEN,
        "NextGen_Agent_1",
        {"model_config": {"model": "gpt-4", "temperature": 0.8}}
    )
    
    # Create Crew Agents
    coordinator = await factory.create_agent(
        AgentType.CREW,
        "Crew_Coordinator",
        {"role": "coordinator", "team_id": "alpha_team"}
    )
    
    researcher = await factory.create_agent(
        AgentType.CREW,
        "Crew_Researcher", 
        {"role": "researcher", "team_id": "alpha_team"}
    )
    
    # Create Specialized Agents
    data_analyst = await factory.create_agent(
        AgentType.SPECIALIZED,
        "Data_Analyst_1",
        {"specialization": "data_analysis", "expertise_level": "expert"}
    )
    
    code_generator = await factory.create_agent(
        AgentType.SPECIALIZED,
        "Code_Generator_1",
        {"specialization": "code_generation", "expertise_level": "advanced"}
    )
    
    return {
        "fsm_agent": fsm_agent,
        "next_gen_agent": next_gen_agent,
        "coordinator": coordinator,
        "researcher": researcher,
        "data_analyst": data_analyst,
        "code_generator": code_generator
    } 