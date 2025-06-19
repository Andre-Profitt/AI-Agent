"""
Task Distribution for Multi-Agent System

This module provides intelligent task distribution:
- Multiple distribution strategies
- Load balancing and optimization
- Task routing and prioritization
- Performance-based agent selection
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import defaultdict, deque
from enum import Enum, auto
import numpy as np
import logging

from .core import UnifiedTask, AgentMetadata, AgentStatus, AgentCapability
from .registry import AgentRegistry

logger = logging.getLogger(__name__)

class TaskDistributionStrategy(Enum):
    """Task distribution strategies"""
    ROUND_ROBIN = auto()
    LEAST_LOADED = auto()
    CAPABILITY_BASED = auto()
    PERFORMANCE_BASED = auto()
    HYBRID = auto()
    PRIORITY_BASED = auto()
    COST_OPTIMIZED = auto()

class TaskDistributor:
    """Intelligent task distribution system"""
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.task_queue = asyncio.PriorityQueue()
        self.agent_load: Dict[str, int] = defaultdict(int)
        self.distribution_strategy = TaskDistributionStrategy.HYBRID
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        self.task_routing_rules: List[Callable] = []
        self.task_history: deque = deque(maxlen=10000)
        
        # Configuration
        self.max_queue_size = 10000
        self.default_timeout = 300  # 5 minutes
        self.retry_attempts = 3
        self.retry_delay = 60  # 1 minute
        
        # Statistics
        self.stats = {
            "tasks_submitted": 0,
            "tasks_distributed": 0,
            "tasks_failed": 0,
            "average_distribution_time": 0.0,
            "strategy_usage": defaultdict(int)
        }
        
    async def submit_task(self, task: UnifiedTask) -> str:
        """Submit a task for distribution"""
        self.stats["tasks_submitted"] += 1
        
        # Apply routing rules
        for rule in self.task_routing_rules:
            if rule(task):
                return await self._direct_route_task(task, rule)
        
        # Add to queue for distribution
        priority = -task.get_priority_score()  # Negative for max-heap behavior
        await self.task_queue.put((priority, task))
        
        logger.info(f"Task {task.task_id} submitted for distribution (priority: {task.priority})")
        return task.task_id
    
    async def distribute_tasks(self):
        """Main task distribution loop"""
        while True:
            try:
                # Get next task
                priority, task = await self.task_queue.get()
                
                # Find suitable agents
                suitable_agents = await self._find_suitable_agents(task)
                
                if not suitable_agents:
                    logger.warning(f"No suitable agents for task {task.task_id}")
                    # Could implement retry logic here
                    self.stats["tasks_failed"] += 1
                    continue
                
                # Select agent based on strategy
                selected_agent = await self._select_agent(suitable_agents, task)
                
                # Assign task to agent
                await self._assign_task(selected_agent, task)
                
                self.stats["tasks_distributed"] += 1
                self.stats["strategy_usage"][self.distribution_strategy.name] += 1
                
            except asyncio.CancelledError:
                logger.info("Task distribution loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in task distribution: {e}")
                self.stats["tasks_failed"] += 1
    
    async def _find_suitable_agents(self, task: UnifiedTask) -> List[AgentMetadata]:
        """Find agents suitable for a task"""
        # Discover agents with required capabilities
        agents = await self.registry.discover(
            capabilities=task.required_capabilities,
            status=AgentStatus.AVAILABLE,
            min_reliability=0.7  # Minimum reliability threshold
        )
        
        # Filter by additional criteria
        suitable = []
        for agent in agents:
            # Check reliability threshold
            if agent.reliability_score < 0.7:
                continue
            
            # Check if agent can handle the task type
            if not self._agent_can_handle_task_type(agent, task):
                continue
            
            # Check resource requirements (simplified)
            if not self._check_resource_requirements(agent, task):
                continue
            
            suitable.append(agent)
        
        return suitable
    
    def _agent_can_handle_task_type(self, agent: AgentMetadata, task: UnifiedTask) -> bool:
        """Check if agent can handle the specific task type"""
        # Check performance history for this task type
        agent_performance = self.performance_history.get(agent.agent_id, [])
        
        if agent_performance:
            # If agent has failed this task type multiple times, skip
            recent_failures = sum(1 for score in agent_performance[-10:] if score < 0.5)
            if recent_failures > 5:  # More than 50% failures in last 10 tasks
                return False
        
        return True
    
    def _check_resource_requirements(self, agent: AgentMetadata, task: UnifiedTask) -> bool:
        """Check if agent has sufficient resources for the task"""
        # This is a simplified check - in practice would check actual resource availability
        current_load = self.agent_load[agent.agent_id]
        max_load = agent.performance_metrics.get("max_concurrent_tasks", 5)
        
        return current_load < max_load
    
    async def _select_agent(self, agents: List[AgentMetadata], 
                          task: UnifiedTask) -> AgentMetadata:
        """Select the best agent based on distribution strategy"""
        if not agents:
            raise ValueError("No agents available for selection")
        
        if self.distribution_strategy == TaskDistributionStrategy.ROUND_ROBIN:
            # Simple round-robin
            return agents[0]
        
        elif self.distribution_strategy == TaskDistributionStrategy.LEAST_LOADED:
            # Select least loaded agent
            return min(agents, key=lambda a: self.agent_load[a.agent_id])
        
        elif self.distribution_strategy == TaskDistributionStrategy.PERFORMANCE_BASED:
            # Select based on past performance
            return max(agents, key=lambda a: np.mean(self.performance_history[a.agent_id]) 
                      if self.performance_history[a.agent_id] else 0.5)
        
        elif self.distribution_strategy == TaskDistributionStrategy.PRIORITY_BASED:
            # Select based on agent priority and task priority
            return max(agents, key=lambda a: a.performance_metrics.get("priority", 5) * task.priority)
        
        elif self.distribution_strategy == TaskDistributionStrategy.COST_OPTIMIZED:
            # Select based on cost efficiency
            return min(agents, key=lambda a: a.performance_metrics.get("cost_per_task", 1.0))
        
        elif self.distribution_strategy == TaskDistributionStrategy.HYBRID:
            # Hybrid approach combining multiple factors
            scores = []
            for agent in agents:
                score = self._calculate_hybrid_score(agent, task)
                scores.append((score, agent))
            
            scores.sort(reverse=True)
            return scores[0][1]
        
        else:
            # Default to first available
            return agents[0]
    
    def _calculate_hybrid_score(self, agent: AgentMetadata, 
                               task: UnifiedTask) -> float:
        """Calculate hybrid score for agent selection"""
        score = 0.0
        
        # Load factor (25%)
        load = self.agent_load[agent.agent_id]
        max_load = agent.performance_metrics.get("max_concurrent_tasks", 5)
        load_score = 1.0 - (load / max_load)  # Less load = higher score
        score += load_score * 0.25
        
        # Performance history (25%)
        if self.performance_history[agent.agent_id]:
            perf_score = np.mean(self.performance_history[agent.agent_id])
        else:
            perf_score = 0.5
        score += perf_score * 0.25
        
        # Reliability (20%)
        score += agent.reliability_score * 0.20
        
        # Capability match (15%)
        capability_overlap = len(set(agent.capabilities) & set(task.required_capabilities))
        capability_score = capability_overlap / len(task.required_capabilities) if task.required_capabilities else 1.0
        score += capability_score * 0.15
        
        # Task type specialization (10%)
        task_type_performance = agent.performance_metrics.get("task_type_performance", {})
        type_score = task_type_performance.get(task.task_type, 0.5)
        score += type_score * 0.10
        
        # Cost efficiency (5%)
        cost_per_task = agent.performance_metrics.get("cost_per_task", 1.0)
        cost_score = 1.0 / (1.0 + cost_per_task)  # Lower cost = higher score
        score += cost_score * 0.05
        
        return score
    
    async def _assign_task(self, agent: AgentMetadata, task: UnifiedTask):
        """Assign a task to an agent"""
        # Update load tracking
        self.agent_load[agent.agent_id] += 1
        
        # Record task assignment
        assignment_record = {
            "task_id": task.task_id,
            "agent_id": agent.agent_id,
            "timestamp": time.time(),
            "strategy": self.distribution_strategy.name,
            "task_priority": task.priority,
            "agent_load": self.agent_load[agent.agent_id]
        }
        self.task_history.append(assignment_record)
        
        # Send task to agent (through communication protocol)
        # This would integrate with the communication system
        message = {
            "action": "execute_task",
            "task": task.to_dict(),
            "assigned_at": time.time()
        }
        
        logger.info(f"Assigned task {task.task_id} to agent {agent.agent_id} "
                   f"(load: {self.agent_load[agent.agent_id]})")
    
    def add_routing_rule(self, rule: Callable):
        """Add a custom routing rule"""
        self.task_routing_rules.append(rule)
        logger.info("Added custom routing rule")
    
    async def _direct_route_task(self, task: UnifiedTask, rule: Callable) -> str:
        """Directly route a task based on a rule"""
        # Rule should return agent_id or None
        agent_id = rule(task)
        if agent_id:
            agent = await self.registry.get_agent(agent_id)
            if agent and agent.status == AgentStatus.AVAILABLE:
                await self._assign_task(agent, task)
                return task.task_id
        
        # Fallback to normal distribution
        priority = -task.get_priority_score()
        await self.task_queue.put((priority, task))
        return task.task_id
    
    async def record_task_result(self, task_id: str, agent_id: str, 
                               success: bool, execution_time: float,
                               quality_score: Optional[float] = None):
        """Record the result of a task execution"""
        # Update agent load
        self.agent_load[agent_id] = max(0, self.agent_load[agent_id] - 1)
        
        # Update performance history
        if success:
            score = quality_score if quality_score is not None else 1.0
        else:
            score = 0.0
        
        self.performance_history[agent_id].append(score)
        
        # Keep only recent history
        if len(self.performance_history[agent_id]) > 100:
            self.performance_history[agent_id] = self.performance_history[agent_id][-100:]
        
        # Update task history
        for record in self.task_history:
            if record["task_id"] == task_id and record["agent_id"] == agent_id:
                record["completed_at"] = time.time()
                record["success"] = success
                record["execution_time"] = execution_time
                record["quality_score"] = quality_score
                break
    
    def get_distribution_stats(self) -> Dict[str, Any]:
        """Get task distribution statistics"""
        # Calculate average distribution time
        if self.task_history:
            distribution_times = []
            for record in self.task_history:
                if "completed_at" in record:
                    dist_time = record["completed_at"] - record["timestamp"]
                    distribution_times.append(dist_time)
            
            if distribution_times:
                self.stats["average_distribution_time"] = np.mean(distribution_times)
        
        # Calculate success rates by strategy
        strategy_success = defaultdict(lambda: {"total": 0, "successful": 0})
        for record in self.task_history:
            if "success" in record:
                strategy = record.get("strategy", "unknown")
                strategy_success[strategy]["total"] += 1
                if record["success"]:
                    strategy_success[strategy]["successful"] += 1
        
        success_rates = {}
        for strategy, counts in strategy_success.items():
            if counts["total"] > 0:
                success_rates[strategy] = counts["successful"] / counts["total"]
        
        return {
            **self.stats,
            "queue_size": self.task_queue.qsize(),
            "active_agents": len([aid for aid, load in self.agent_load.items() if load > 0]),
            "strategy_success_rates": success_rates,
            "agent_load_distribution": dict(self.agent_load)
        }
    
    async def get_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Get performance statistics for a specific agent"""
        performance_history = self.performance_history.get(agent_id, [])
        
        if not performance_history:
            return {
                "agent_id": agent_id,
                "total_tasks": 0,
                "success_rate": 0.0,
                "average_score": 0.0,
                "current_load": self.agent_load[agent_id]
            }
        
        return {
            "agent_id": agent_id,
            "total_tasks": len(performance_history),
            "success_rate": sum(1 for score in performance_history if score > 0.5) / len(performance_history),
            "average_score": np.mean(performance_history),
            "recent_performance": np.mean(performance_history[-10:]) if len(performance_history) >= 10 else np.mean(performance_history),
            "current_load": self.agent_load[agent_id],
            "performance_trend": self._calculate_performance_trend(performance_history)
        }
    
    def _calculate_performance_trend(self, performance_history: List[float]) -> str:
        """Calculate performance trend (improving, declining, stable)"""
        if len(performance_history) < 10:
            return "insufficient_data"
        
        recent = performance_history[-10:]
        earlier = performance_history[-20:-10] if len(performance_history) >= 20 else performance_history[:-10]
        
        recent_avg = np.mean(recent)
        earlier_avg = np.mean(earlier)
        
        if recent_avg > earlier_avg + 0.1:
            return "improving"
        elif recent_avg < earlier_avg - 0.1:
            return "declining"
        else:
            return "stable"
    
    async def optimize_distribution(self) -> List[Dict[str, Any]]:
        """Analyze and suggest distribution optimizations"""
        optimizations = []
        
        # Check for load imbalance
        loads = list(self.agent_load.values())
        if loads:
            avg_load = np.mean(loads)
            max_load = max(loads)
            
            if max_load > avg_load * 2:  # Significant imbalance
                overloaded_agents = [aid for aid, load in self.agent_load.items() if load > avg_load * 1.5]
                optimizations.append({
                    "type": "load_balance",
                    "description": f"Load imbalance detected: {len(overloaded_agents)} agents overloaded",
                    "agents": overloaded_agents,
                    "suggested_action": "Consider adding more agents or redistributing tasks"
                })
        
        # Check strategy effectiveness
        strategy_stats = self.get_distribution_stats()
        for strategy, success_rate in strategy_stats.get("strategy_success_rates", {}).items():
            if success_rate < 0.7:  # Low success rate
                optimizations.append({
                    "type": "strategy_optimization",
                    "description": f"Low success rate for {strategy} strategy: {success_rate:.2f}",
                    "suggested_action": "Consider switching to a different strategy or improving agent selection"
                })
        
        return optimizations
    
    async def set_distribution_strategy(self, strategy: TaskDistributionStrategy):
        """Change the distribution strategy"""
        self.distribution_strategy = strategy
        logger.info(f"Changed distribution strategy to: {strategy.name}")
    
    async def clear_task_history(self):
        """Clear task history"""
        self.task_history.clear()
        logger.info("Cleared task history") 