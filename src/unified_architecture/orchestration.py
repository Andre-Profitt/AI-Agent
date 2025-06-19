"""
Orchestration Engine for Multi-Agent System

This module provides intelligent agent selection and task coordination:
- Agent registration and discovery
- Task-agent matching and selection
- Performance tracking and optimization
- Complex task orchestration
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
import logging

from .core import (
    IUnifiedAgent, AgentMetadata, UnifiedTask, TaskResult, 
    AgentStatus, AgentCapability
)
from .performance import PerformanceTracker

logger = logging.getLogger(__name__)

class OrchestrationEngine:
    """Manages agent selection and coordination"""
    
    def __init__(self):
        self.agents: Dict[str, IUnifiedAgent] = {}
        self.agent_metadata: Dict[str, AgentMetadata] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.execution_history: deque = deque(maxlen=1000)
        self.performance_tracker = PerformanceTracker()
        self.agent_load: Dict[str, int] = defaultdict(int)
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        self.failed_tasks: Dict[str, List[str]] = defaultdict(list)  # agent_id -> failed task_ids
        self.successful_tasks: Dict[str, List[str]] = defaultdict(list)  # agent_id -> successful task_ids
        
        # Configuration
        self.max_concurrent_tasks_per_agent = 5
        self.task_timeout = 300  # 5 minutes
        self.retry_delay = 60  # 1 minute
        self.max_retries = 3
        
        # Statistics
        self.stats = {
            "total_tasks_submitted": 0,
            "total_tasks_completed": 0,
            "total_tasks_failed": 0,
            "average_execution_time": 0.0,
            "agent_utilization": {}
        }
        
    async def register_agent(self, agent: IUnifiedAgent, metadata: AgentMetadata) -> bool:
        """Register an agent with the orchestration engine"""
        try:
            # Validate metadata
            if not metadata.agent_id or not metadata.name:
                logger.error("Invalid agent metadata")
                return False
            
            # Check if agent already registered
            if metadata.agent_id in self.agents:
                logger.warning(f"Agent {metadata.agent_id} already registered")
                return False
            
            # Store agent and metadata
            self.agents[metadata.agent_id] = agent
            self.agent_metadata[metadata.agent_id] = metadata
            
            # Initialize agent
            config = {
                "orchestrator_id": id(self),
                "registration_time": datetime.utcnow().isoformat(),
                "max_concurrent_tasks": self.max_concurrent_tasks_per_agent
            }
            
            success = await agent.initialize(config)
            if not success:
                logger.error(f"Failed to initialize agent {metadata.agent_id}")
                return False
            
            # Update status
            metadata.status = AgentStatus.AVAILABLE
            metadata.last_seen = datetime.utcnow()
            
            # Initialize statistics
            self.stats["agent_utilization"][metadata.agent_id] = {
                "tasks_completed": 0,
                "tasks_failed": 0,
                "total_execution_time": 0.0,
                "last_activity": datetime.utcnow()
            }
            
            logger.info(f"Registered agent: {metadata.name} ({metadata.agent_id}) "
                       f"with capabilities: {[cap.name for cap in metadata.capabilities]}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering agent {metadata.agent_id}: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the orchestration engine"""
        try:
            if agent_id not in self.agents:
                return False
            
            # Shutdown agent
            agent = self.agents[agent_id]
            await agent.shutdown()
            
            # Remove from tracking
            del self.agents[agent_id]
            del self.agent_metadata[agent_id]
            
            if agent_id in self.agent_load:
                del self.agent_load[agent_id]
            
            if agent_id in self.stats["agent_utilization"]:
                del self.stats["agent_utilization"][agent_id]
            
            logger.info(f"Unregistered agent: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error unregistering agent {agent_id}: {e}")
            return False
    
    async def select_agent(self, task: UnifiedTask) -> Optional[str]:
        """Select the best agent for a task"""
        try:
            eligible_agents = []
            
            for agent_id, metadata in self.agent_metadata.items():
                # Check if agent has required capabilities
                if not all(cap in metadata.capabilities for cap in task.required_capabilities):
                    continue
                
                # Check if agent is available
                agent = self.agents[agent_id]
                status = await agent.get_status()
                
                if status not in [AgentStatus.IDLE, AgentStatus.AVAILABLE]:
                    continue
                
                # Check load limit
                if self.agent_load[agent_id] >= self.max_concurrent_tasks_per_agent:
                    continue
                
                # Validate task
                is_valid, error = await agent.validate_task(task)
                if not is_valid:
                    logger.debug(f"Agent {agent_id} cannot handle task {task.task_id}: {error}")
                    continue
                
                # Calculate suitability score
                score = await self._calculate_agent_score(metadata, task, agent_id)
                eligible_agents.append((score, agent_id))
            
            if not eligible_agents:
                logger.warning(f"No eligible agents found for task {task.task_id}")
                return None
            
            # Select agent with highest score
            eligible_agents.sort(reverse=True)
            selected_agent_id = eligible_agents[0][1]
            
            logger.debug(f"Selected agent {selected_agent_id} for task {task.task_id} "
                        f"(score: {eligible_agents[0][0]:.3f})")
            
            return selected_agent_id
            
        except Exception as e:
            logger.error(f"Error selecting agent for task {task.task_id}: {e}")
            return None
    
    async def _calculate_agent_score(self, metadata: AgentMetadata, 
                                   task: UnifiedTask, agent_id: str) -> float:
        """Calculate agent suitability score for a task"""
        score = 0.0
        
        # Reliability score (30%)
        score += metadata.reliability_score * 0.3
        
        # Performance metrics (25%)
        agent_metrics = self.performance_tracker.get_agent_metrics(agent_id)
        avg_performance = agent_metrics.get("success_rate", 0.5)
        score += avg_performance * 0.25
        
        # Capability match (20%)
        capability_overlap = len(set(metadata.capabilities) & set(task.required_capabilities))
        capability_score = capability_overlap / len(task.required_capabilities) if task.required_capabilities else 1.0
        score += capability_score * 0.2
        
        # Load factor (15%) - prefer less loaded agents
        current_load = self.agent_load[agent_id]
        load_score = 1.0 / (1.0 + current_load)
        score += load_score * 0.15
        
        # Task type specialization (10%)
        task_type_performance = agent_metrics.get("task_breakdown", {}).get(task.task_type, {})
        type_success_rate = task_type_performance.get("success_rate", 0.5)
        score += type_success_rate * 0.1
        
        return score
    
    async def execute_task(self, task: UnifiedTask) -> TaskResult:
        """Execute a task using the best available agent"""
        self.stats["total_tasks_submitted"] += 1
        
        agent_id = await self.select_agent(task)
        
        if not agent_id:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                result=None,
                execution_time=0.0,
                agent_id="none",
                error="No suitable agent available"
            )
        
        agent = self.agents[agent_id]
        metadata = self.agent_metadata[agent_id]
        
        # Update agent status and load
        metadata.status = AgentStatus.BUSY
        self.agent_load[agent_id] += 1
        self.task_assignments[task.task_id] = agent_id
        
        try:
            # Execute task with timeout
            start_time = time.time()
            
            if task.timeout:
                timeout = task.timeout
            else:
                timeout = self.task_timeout
            
            # Execute task
            result = await asyncio.wait_for(
                agent.execute(task), 
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            # Update performance tracking
            self.performance_tracker.record_execution(
                agent_id, task.task_type, execution_time, result.success
            )
            
            # Update agent reliability
            metadata.reliability_score = self._update_reliability_score(
                metadata.reliability_score, result.success
            )
            
            # Update statistics
            self._update_execution_stats(agent_id, result, execution_time)
            
            # Record in history
            self.execution_history.append({
                "task_id": task.task_id,
                "agent_id": agent_id,
                "timestamp": datetime.utcnow(),
                "success": result.success,
                "execution_time": execution_time
            })
            
            # Track task outcomes
            if result.success:
                self.successful_tasks[agent_id].append(task.task_id)
            else:
                self.failed_tasks[agent_id].append(task.task_id)
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Task {task.task_id} timed out after {timeout}s")
            return TaskResult(
                task_id=task.task_id,
                success=False,
                result=None,
                execution_time=timeout,
                agent_id=agent_id,
                error="Task execution timed out"
            )
            
        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")
            return TaskResult(
                task_id=task.task_id,
                success=False,
                result=None,
                execution_time=time.time() - start_time,
                agent_id=agent_id,
                error=str(e)
            )
            
        finally:
            # Reset agent status and load
            metadata.status = AgentStatus.AVAILABLE
            metadata.last_seen = datetime.utcnow()
            self.agent_load[agent_id] = max(0, self.agent_load[agent_id] - 1)
            
            if task.task_id in self.task_assignments:
                del self.task_assignments[task.task_id]
    
    def _update_reliability_score(self, current_score: float, success: bool) -> float:
        """Update agent reliability score using exponential moving average"""
        alpha = 0.1  # Learning rate
        new_value = 1.0 if success else 0.0
        return (1 - alpha) * current_score + alpha * new_value
    
    def _update_execution_stats(self, agent_id: str, result: TaskResult, execution_time: float):
        """Update execution statistics"""
        if agent_id not in self.stats["agent_utilization"]:
            self.stats["agent_utilization"][agent_id] = {
                "tasks_completed": 0,
                "tasks_failed": 0,
                "total_execution_time": 0.0,
                "last_activity": datetime.utcnow()
            }
        
        stats = self.stats["agent_utilization"][agent_id]
        
        if result.success:
            stats["tasks_completed"] += 1
            self.stats["total_tasks_completed"] += 1
        else:
            stats["tasks_failed"] += 1
            self.stats["total_tasks_failed"] += 1
        
        stats["total_execution_time"] += execution_time
        stats["last_activity"] = datetime.utcnow()
        
        # Update overall average execution time
        total_completed = self.stats["total_tasks_completed"]
        if total_completed > 0:
            self.stats["average_execution_time"] = (
                (self.stats["average_execution_time"] * (total_completed - 1) + execution_time) / total_completed
            )
    
    async def orchestrate_complex_task(self, task: UnifiedTask, 
                                     subtasks: List[UnifiedTask]) -> List[TaskResult]:
        """Orchestrate execution of a complex task with multiple subtasks"""
        try:
            # Build dependency graph
            dep_graph = self._build_dependency_graph(subtasks)
            
            # Check for cycles
            if not nx.is_directed_acyclic_graph(dep_graph):
                logger.error("Dependency graph contains cycles")
                return []
            
            # Execute tasks in topological order
            results = {}
            execution_order = list(nx.topological_sort(dep_graph))
            
            logger.info(f"Executing complex task {task.task_id} with {len(subtasks)} subtasks")
            
            for subtask_id in execution_order:
                subtask = next(t for t in subtasks if t.task_id == subtask_id)
                
                # Wait for dependencies
                for dep_id in subtask.dependencies:
                    if dep_id in results:
                        while not results[dep_id].success:
                            await asyncio.sleep(0.1)
                
                # Execute subtask
                result = await self.execute_task(subtask)
                results[subtask_id] = result
                
                # Check if we should continue
                if not result.success and subtask.max_retries > 0:
                    # Retry logic could be implemented here
                    logger.warning(f"Subtask {subtask_id} failed, retries available: {subtask.max_retries}")
            
            return list(results.values())
            
        except Exception as e:
            logger.error(f"Error orchestrating complex task {task.task_id}: {e}")
            return []
    
    def _build_dependency_graph(self, tasks: List[UnifiedTask]) -> nx.DiGraph:
        """Build dependency graph from tasks"""
        graph = nx.DiGraph()
        
        for task in tasks:
            graph.add_node(task.task_id)
            for dep in task.dependencies:
                graph.add_edge(dep, task.task_id)
        
        return graph
    
    async def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of an agent"""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        metadata = self.agent_metadata[agent_id]
        
        # Get performance metrics
        metrics = self.performance_tracker.get_agent_metrics(agent_id)
        
        # Get resource usage
        resource_usage = await agent.get_resource_usage()
        
        return {
            "agent_id": agent_id,
            "name": metadata.name,
            "status": metadata.status.name,
            "capabilities": [cap.name for cap in metadata.capabilities],
            "reliability_score": metadata.reliability_score,
            "current_load": self.agent_load[agent_id],
            "performance_metrics": metrics,
            "resource_usage": resource_usage,
            "last_seen": metadata.last_seen.isoformat(),
            "total_tasks_completed": len(self.successful_tasks[agent_id]),
            "total_tasks_failed": len(self.failed_tasks[agent_id])
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        total_agents = len(self.agents)
        available_agents = sum(1 for m in self.agent_metadata.values() 
                             if m.status in [AgentStatus.IDLE, AgentStatus.AVAILABLE])
        
        return {
            "total_agents": total_agents,
            "available_agents": available_agents,
            "total_load": sum(self.agent_load.values()),
            "task_stats": {
                "submitted": self.stats["total_tasks_submitted"],
                "completed": self.stats["total_tasks_completed"],
                "failed": self.stats["total_tasks_failed"],
                "success_rate": (self.stats["total_tasks_completed"] / 
                               max(1, self.stats["total_tasks_submitted"]))
            },
            "average_execution_time": self.stats["average_execution_time"],
            "agent_utilization": self.stats["agent_utilization"]
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all agents"""
        health_results = {}
        
        for agent_id, agent in self.agents.items():
            try:
                health = await agent.health_check()
                health_results[agent_id] = {
                    "healthy": health.get("healthy", False),
                    "status": health.get("status", "unknown"),
                    "uptime": health.get("uptime", 0)
                }
            except Exception as e:
                health_results[agent_id] = {
                    "healthy": False,
                    "status": "error",
                    "error": str(e)
                }
        
        return health_results 