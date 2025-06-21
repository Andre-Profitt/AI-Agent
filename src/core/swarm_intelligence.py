"""
Swarm Intelligence System
Massive parallel agent coordination
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import defaultdict
import numpy as np
import networkx as nx
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)

class SwarmRole(Enum):
    EXPLORER = "explorer"          # Discover new solutions
    WORKER = "worker"              # Execute specific tasks
    COORDINATOR = "coordinator"    # Organize other agents
    VALIDATOR = "validator"        # Verify results
    OPTIMIZER = "optimizer"        # Improve solutions
    SCOUT = "scout"               # Quick reconnaissance
    SPECIALIST = "specialist"      # Domain expert

@dataclass
class SwarmTask:
    """Task for swarm execution"""
    id: str
    objective: str
    subtasks: List[Dict[str, Any]]
    priority: float
    deadline: Optional[datetime] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    required_roles: List[SwarmRole] = field(default_factory=list)
    
@dataclass
class SwarmAgent:
    """Individual agent in swarm"""
    id: str
    role: SwarmRole
    capabilities: Set[str]
    state: str = "idle"
    current_task: Optional[str] = None
    performance_score: float = 1.0
    energy: float = 1.0
    position: Tuple[float, float] = (0.0, 0.0)  # For spatial organization
    
class SwarmIntelligence:
    """
    Advanced swarm intelligence system supporting:
    - Massive parallel coordination (1000+ agents)
    - Emergent behavior patterns
    - Self-organization
    - Collective decision making
    - Adaptive task distribution
    """
    
    def __init__(self, initial_agents: int = 100):
        self.agents = {}
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
        self.communication_network = nx.Graph()
        self.pheromone_map = defaultdict(float)  # Stigmergic coordination
        self.swarm_memory = {}
        self.emergence_patterns = []
        
        # Initialize swarm
        self._initialize_swarm(initial_agents)
        
    def _initialize_swarm(self, num_agents: int):
        """Initialize swarm with diverse agents"""
        role_distribution = {
            SwarmRole.WORKER: 0.4,
            SwarmRole.EXPLORER: 0.2,
            SwarmRole.COORDINATOR: 0.1,
            SwarmRole.VALIDATOR: 0.1,
            SwarmRole.OPTIMIZER: 0.1,
            SwarmRole.SCOUT: 0.05,
            SwarmRole.SPECIALIST: 0.05
        }
        
        for i in range(num_agents):
            # Assign role based on distribution
            role = self._select_role(role_distribution)
            
            agent = SwarmAgent(
                id=f"agent_{uuid.uuid4().hex[:8]}",
                role=role,
                capabilities=self._generate_capabilities(role),
                position=(np.random.rand() * 100, np.random.rand() * 100)
            )
            
            self.agents[agent.id] = agent
            self.communication_network.add_node(agent.id)
            
        # Create initial communication topology
        self._create_communication_topology()
        
    def _create_communication_topology(self):
        """Create efficient communication network"""
        agents_list = list(self.agents.keys())
        
        # Small world network for efficient information spread
        for i, agent_id in enumerate(agents_list):
            # Connect to nearby agents
            for j in range(i + 1, min(i + 5, len(agents_list))):
                other_id = agents_list[j]
                self.communication_network.add_edge(agent_id, other_id)
                
            # Random long-range connections
            if np.random.random() < 0.1:
                random_agent = np.random.choice(agents_list)
                if random_agent != agent_id:
                    self.communication_network.add_edge(agent_id, random_agent)
                    
    async def execute_task(self, task: SwarmTask) -> Dict[str, Any]:
        """Execute task using swarm intelligence"""
        logger.info(f"Swarm executing task: {task.objective}")
        
        # Decompose task
        subtasks = await self._decompose_task(task)
        
        # Allocate agents
        allocations = await self._allocate_agents(subtasks, task.required_roles)
        
        # Execute in parallel with coordination
        results = await self._coordinate_execution(subtasks, allocations)
        
        # Aggregate results
        final_result = await self._aggregate_results(results, task)
        
        # Learn from execution
        await self._update_swarm_knowledge(task, final_result)
        
        return final_result
        
    async def _decompose_task(self, task: SwarmTask) -> List[Dict[str, Any]]:
        """Decompose task into subtasks"""
        if task.subtasks:
            return task.subtasks
            
        # Auto-decompose based on task complexity
        complexity = self._estimate_complexity(task)
        
        if complexity < 0.3:
            # Simple task - single subtask
            return [{
                "id": f"{task.id}_1",
                "type": "execute",
                "content": task.objective,
                "required_agents": 1
            }]
        elif complexity < 0.7:
            # Medium complexity - parallel subtasks
            return [
                {
                    "id": f"{task.id}_explore",
                    "type": "explore",
                    "content": f"Find approaches for: {task.objective}",
                    "required_agents": 5
                },
                {
                    "id": f"{task.id}_execute",
                    "type": "execute",
                    "content": f"Implement solution for: {task.objective}",
                    "required_agents": 10
                },
                {
                    "id": f"{task.id}_validate",
                    "type": "validate",
                    "content": f"Verify solution for: {task.objective}",
                    "required_agents": 3
                }
            ]
        else:
            # High complexity - hierarchical decomposition
            return await self._hierarchical_decomposition(task)
            
    async def _allocate_agents(
        self, 
        subtasks: List[Dict[str, Any]], 
        required_roles: List[SwarmRole]
    ) -> Dict[str, List[str]]:
        """Allocate agents to subtasks"""
        allocations = {}
        available_agents = {
            agent_id: agent for agent_id, agent in self.agents.items()
            if agent.state == "idle"
        }
        
        for subtask in subtasks:
            subtask_id = subtask["id"]
            required_count = subtask.get("required_agents", 1)
            task_type = subtask.get("type", "general")
            
            # Select best agents for subtask
            selected_agents = self._select_agents_for_task(
                available_agents,
                task_type,
                required_count,
                required_roles
            )
            
            allocations[subtask_id] = selected_agents
            
            # Mark agents as busy
            for agent_id in selected_agents:
                self.agents[agent_id].state = "working"
                self.agents[agent_id].current_task = subtask_id
                del available_agents[agent_id]
                
        return allocations
        
    def _select_agents_for_task(
        self,
        available_agents: Dict[str, SwarmAgent],
        task_type: str,
        count: int,
        required_roles: List[SwarmRole]
    ) -> List[str]:
        """Select best agents for specific task"""
        # Score agents based on suitability
        agent_scores = []
        
        for agent_id, agent in available_agents.items():
            score = 0.0
            
            # Role matching
            if task_type == "explore" and agent.role == SwarmRole.EXPLORER:
                score += 0.5
            elif task_type == "execute" and agent.role == SwarmRole.WORKER:
                score += 0.5
            elif task_type == "validate" and agent.role == SwarmRole.VALIDATOR:
                score += 0.5
                
            # Performance history
            score += agent.performance_score * 0.3
            
            # Energy level
            score += agent.energy * 0.2
            
            agent_scores.append((agent_id, score))
            
        # Sort by score and select top agents
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        return [agent_id for agent_id, _ in agent_scores[:count]]
        
    async def _coordinate_execution(
        self,
        subtasks: List[Dict[str, Any]],
        allocations: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Coordinate parallel execution with emergence"""
        results = {}
        execution_tasks = []
        
        for subtask in subtasks:
            subtask_id = subtask["id"]
            assigned_agents = allocations.get(subtask_id, [])
            
            if assigned_agents:
                # Create execution coroutine
                exec_task = self._execute_subtask_with_agents(
                    subtask,
                    assigned_agents
                )
                execution_tasks.append(exec_task)
                
        # Execute all subtasks in parallel
        subtask_results = await asyncio.gather(*execution_tasks)
        
        # Combine results
        for subtask, result in zip(subtasks, subtask_results):
            results[subtask["id"]] = result
            
        return results
        
    async def _execute_subtask_with_agents(
        self,
        subtask: Dict[str, Any],
        agent_ids: List[str]
    ) -> Dict[str, Any]:
        """Execute subtask with assigned agents"""
        # Initialize shared workspace
        workspace = {
            "subtask": subtask,
            "partial_results": [],
            "consensus": None,
            "iterations": 0
        }
        
        # Agent execution loop
        max_iterations = 10
        
        for iteration in range(max_iterations):
            # Each agent contributes
            agent_contributions = []
            
            for agent_id in agent_ids:
                agent = self.agents[agent_id]
                contribution = await self._agent_contribute(
                    agent,
                    workspace,
                    iteration
                )
                agent_contributions.append(contribution)
                
                # Deposit pheromone (stigmergic coordination)
                self._deposit_pheromone(subtask["id"], agent.position, contribution["quality"])
                
            # Update workspace with contributions
            workspace["partial_results"].extend(agent_contributions)
            
            # Check for consensus/completion
            if await self._check_completion(workspace, agent_contributions):
                break
                
            # Share information between agents
            await self._share_information(agent_ids, workspace)
            
            workspace["iterations"] = iteration + 1
            
        # Finalize result
        result = await self._finalize_subtask_result(workspace)
        
        # Release agents
        for agent_id in agent_ids:
            self.agents[agent_id].state = "idle"
            self.agents[agent_id].current_task = None
            
        return result
        
    async def _agent_contribute(
        self,
        agent: SwarmAgent,
        workspace: Dict[str, Any],
        iteration: int
    ) -> Dict[str, Any]:
        """Individual agent contribution"""
        subtask = workspace["subtask"]
        
        # Agent processes based on role
        if agent.role == SwarmRole.EXPLORER:
            result = await self._explore_solution_space(subtask, workspace)
        elif agent.role == SwarmRole.WORKER:
            result = await self._execute_work(subtask, workspace)
        elif agent.role == SwarmRole.VALIDATOR:
            result = await self._validate_work(workspace["partial_results"])
        elif agent.role == SwarmRole.OPTIMIZER:
            result = await self._optimize_solution(workspace["partial_results"])
        else:
            result = await self._general_contribution(subtask, workspace)
            
        # Update agent state
        agent.energy *= 0.95  # Energy depletion
        
        return {
            "agent_id": agent.id,
            "role": agent.role.value,
            "iteration": iteration,
            "result": result,
            "quality": self._assess_quality(result),
            "timestamp": datetime.utcnow()
        }
        
    def _deposit_pheromone(self, task_id: str, position: Tuple[float, float], strength: float):
        """Deposit pheromone for stigmergic coordination"""
        key = f"{task_id}_{position[0]:.1f}_{position[1]:.1f}"
        self.pheromone_map[key] += strength
        
        # Evaporation
        for k in list(self.pheromone_map.keys()):
            self.pheromone_map[k] *= 0.99
            if self.pheromone_map[k] < 0.01:
                del self.pheromone_map[k]
                
    async def spawn_agents(self, count: int, role: Optional[SwarmRole] = None):
        """Dynamically spawn new agents"""
        new_agents = []
        
        for _ in range(count):
            agent = SwarmAgent(
                id=f"agent_{uuid.uuid4().hex[:8]}",
                role=role or self._select_role({}),
                capabilities=self._generate_capabilities(role),
                position=(np.random.rand() * 100, np.random.rand() * 100)
            )
            
            self.agents[agent.id] = agent
            self.communication_network.add_node(agent.id)
            new_agents.append(agent.id)
            
        # Connect new agents to network
        for new_agent in new_agents:
            # Connect to nearest agents
            nearest = self._find_nearest_agents(new_agent, 3)
            for neighbor in nearest:
                self.communication_network.add_edge(new_agent, neighbor)
                
        logger.info(f"Spawned {count} new agents")
        
    async def evolve_swarm(self):
        """Evolve swarm behavior based on performance"""
        # Identify successful patterns
        successful_patterns = self._identify_successful_patterns()
        
        # Reproduce successful agents
        for pattern in successful_patterns:
            if pattern["success_rate"] > 0.8:
                # Spawn similar agents
                template_agent = self.agents[pattern["agent_id"]]
                await self.spawn_agents(
                    count=int(pattern["success_rate"] * 5),
                    role=template_agent.role
                )
                
        # Remove underperforming agents
        underperformers = [
            agent_id for agent_id, agent in self.agents.items()
            if agent.performance_score < 0.3 and agent.state == "idle"
        ]
        
        for agent_id in underperformers[:len(underperformers)//10]:  # Remove up to 10%
            del self.agents[agent_id]
            self.communication_network.remove_node(agent_id)
            
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get current swarm status"""
        role_counts = defaultdict(int)
        state_counts = defaultdict(int)
        
        for agent in self.agents.values():
            role_counts[agent.role.value] += 1
            state_counts[agent.state] += 1
            
        return {
            "total_agents": len(self.agents),
            "role_distribution": dict(role_counts),
            "state_distribution": dict(state_counts),
            "active_tasks": len(self.active_tasks),
            "network_connectivity": nx.average_clustering(self.communication_network),
            "pheromone_trails": len(self.pheromone_map),
            "emergence_patterns": len(self.emergence_patterns)
        }
