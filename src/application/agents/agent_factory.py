from agent import tools
from examples.enhanced_unified_example import task
from examples.parallel_execution_example import agents
from examples.parallel_execution_example import results
from migrations.env import config
from performance_dashboard import stats
from tests.load_test import success

from src.agents.crew_workflow import researcher
from src.api_server import fsm_agent
from src.api_server import next_gen_agent
from src.application.agents.agent_factory import code_generator
from src.application.agents.agent_factory import coordinator
from src.application.agents.agent_factory import data_analyst
from src.application.agents.agent_factory import init_config
from src.application.agents.agent_factory import shutdown_tasks
from src.application.agents.agent_factory import specialization
from src.core.entities.agent import Agent
from src.database.models import agent_id
from src.database.models import agent_type
from src.database.models import role
from src.database.models import status
from src.database_extended import success_count
from src.infrastructure.config.configuration_service import agent_config
from src.infrastructure.config.configuration_service import model_config
from src.templates.template_factory import factory
from src.tools_enhanced import get_enhanced_tools
from src.tools_introspection import name

from src.agents.advanced_agent_fsm import AgentType

from src.agents.advanced_agent_fsm import Agent

from src.agents.advanced_agent_fsm import BaseAgent

from src.agents.advanced_agent_fsm import FSMReactAgentImpl

from src.agents.advanced_agent_fsm import AgentFactory
from src.infrastructure.agents.agent_factory import AgentType
from src.infrastructure.agents.concrete_agents import CrewAgentImpl
from src.infrastructure.agents.concrete_agents import FSMReactAgentImpl
from src.infrastructure.agents.concrete_agents import NextGenAgentImpl
from src.infrastructure.agents.concrete_agents import SpecializedAgentImpl
# TODO: Fix undefined variables: Any, Dict, List, Optional, agent_id, agent_type, agents, code_generator, config, coordinator, data_analyst, e, fsm_agent, init_config, logging, member_config, model_config, name, next_gen_agent, researcher, result, results, role, shutdown_tasks, specialization, stats, status, success, success_count, task, team_config, tools, uuid4
from tests.test_complete_system import agent_config
from tests.test_gaia_agent import agent
import factory

from src.utils.tools_enhanced import get_enhanced_tools

# TODO: Fix undefined variables: agent, agent_config, agent_id, agent_type, agents, code_generator, config, coordinator, data_analyst, e, factory, fsm_agent, get_enhanced_tools, init_config, member_config, model_config, name, next_gen_agent, researcher, result, results, role, self, shutdown_tasks, specialization, stats, status, success, success_count, task, team_config, tools

"""

from fastapi import status
Agent factory for creating different types of agents.
"""

from typing import Optional
from typing import Any
from typing import List

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

            self.logger.info("Created {} agent: {} ({})", extra={"agent_type_value": agent_type.value, "name": name, "agent_id": agent_id})
            return agent

        except Exception as e:
            self.logger.error("Failed to create agent {}: {}", extra={"name": name, "e": e})
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

        self.logger.info("Created agent team with {} members", extra={"len_agents_": len(agents)})
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
            self.logger.warning("Agent {} not found", extra={"agent_id": agent_id})
            return False

        try:
            success = await agent.shutdown()
            if success:
                del self.created_agents[agent_id]
                self.logger.info("Agent {} shut down successfully", extra={"agent_id": agent_id})
            return success
        except Exception as e:
            self.logger.error("Failed to shutdown agent {}: {}", extra={"agent_id": agent_id, "e": e})
            return False

    async def shutdown_all(self):
        """Shutdown all created agents"""
        self.logger.info("Shutting down {} agents", extra={"len_self_created_agents_": len(self.created_agents)})

        shutdown_tasks = []
        for agent_id, agent in self.created_agents.items():
            task = self.shutdown_agent(agent_id)
            shutdown_tasks.append(task)

        # Wait for all shutdowns to complete
        results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        success_count = sum(1 for result in results if result is True)
        self.logger.info("Successfully shut down {}/{} agents", extra={"success_count": success_count, "len_self_created_agents_": len(self.created_agents)})

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