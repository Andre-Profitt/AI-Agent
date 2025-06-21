from app import history
from examples.enhanced_unified_example import metrics
from examples.enhanced_unified_example import task
from examples.parallel_execution_example import agents
from setup_environment import value
from tests.load_test import data

from src.agents.enhanced_fsm import state
from src.api_server import conflict
from src.api_server import resource
from src.collaboration.realtime_collaboration import session
from src.core.monitoring import key
from src.database.models import action
from src.database.models import agent_id
from src.database.models import priority
from src.gaia_components.advanced_reasoning_engine import strategy
from src.gaia_components.enhanced_memory_system import current_time
from src.gaia_components.multi_agent_orchestrator import agent_scores
from src.meta_cognition import score
from src.services.integration_hub import limit
from src.unified_architecture.conflict_resolution import agent_priorities
from src.unified_architecture.conflict_resolution import avg_score
from src.unified_architecture.conflict_resolution import common_elements
from src.unified_architecture.conflict_resolution import competing_agents
from src.unified_architecture.conflict_resolution import conflict_id
from src.unified_architecture.conflict_resolution import conflicts_to_remove
from src.unified_architecture.conflict_resolution import consensus
from src.unified_architecture.conflict_resolution import consensus_state
from src.unified_architecture.conflict_resolution import failed_agents
from src.unified_architecture.conflict_resolution import failure_type
from src.unified_architecture.conflict_resolution import goals
from src.unified_architecture.conflict_resolution import importance
from src.unified_architecture.conflict_resolution import most_common
from src.unified_architecture.conflict_resolution import negotiation_id
from src.unified_architecture.conflict_resolution import proposal
from src.unified_architecture.conflict_resolution import proposals
from src.unified_architecture.conflict_resolution import ranked_goals
from src.unified_architecture.conflict_resolution import resolution
from src.unified_architecture.conflict_resolution import resolution_record
from src.unified_architecture.conflict_resolution import resolution_time
from src.unified_architecture.conflict_resolution import retry_config
from src.unified_architecture.conflict_resolution import selected_index
from src.unified_architecture.conflict_resolution import severity
from src.unified_architecture.conflict_resolution import state_counts
from src.unified_architecture.conflict_resolution import state_str
from src.unified_architecture.conflict_resolution import states
from src.unified_architecture.conflict_resolution import total_resolved
from src.unified_architecture.conflict_resolution import violating_agent
from src.unified_architecture.conflict_resolution import violation_type
from src.unified_architecture.conflict_resolution import winner
from src.unified_architecture.task_distribution import agent_performance
from src.unified_architecture.task_distribution import selected_agent
from src.utils.tools_introspection import field

from src.agents.advanced_agent_fsm import Agent
# TODO: Fix undefined variables: Any, Callable, Dict, Enum, List, Optional, action, agent_id, agent_performance, agent_priorities, agent_scores, agents, auto, avg_score, cls, common_elements, competing_agents, conflict, conflict_id, conflict_type, conflicts_to_remove, consensus, consensus_state, current_time, data, dataclass, defaultdict, deque, e, failed_agents, failure_type, field, goal, goals, history, importance, key, limit, logging, max_age, metrics, most_common, negotiation_config, negotiation_id, priority, proposal, proposal_data, proposals, ranked_goals, record, resolution, resolution_record, resolution_time, resource, retry_config, s, score, selected_agent, selected_index, session, severity, state, state_counts, state_str, states, strategy, task, time, total_resolved, uuid, value, value_counts, violating_agent, violation_type, winner, winner_agent, winner_goal, winner_importance, x
from src.infrastructure.monitoring.decorators import performance_metrics


"""

from collections import deque
from dataclasses import field
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
Conflict Resolution for Multi-Agent System

This module provides automated conflict resolution:
- Conflict detection and classification
- Resolution strategies for different conflict types
- Negotiation and consensus mechanisms
- Conflict history and learning
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)

class ConflictType(Enum):
    """Types of conflicts that can occur"""
    RESOURCE_CONTENTION = auto()
    TASK_ASSIGNMENT = auto()
    STATE_INCONSISTENCY = auto()
    GOAL_CONFLICT = auto()
    PRIORITY_DISPUTE = auto()
    COMMUNICATION_FAILURE = auto()
    PERFORMANCE_DISPUTE = auto()
    POLICY_VIOLATION = auto()

@dataclass
class Conflict:
    """Represents a conflict between agents"""
    conflict_id: str
    conflict_type: ConflictType
    involved_agents: List[str]
    description: str
    context: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    resolved: bool = False
    resolution: Optional[Dict[str, Any]] = None
    resolution_time: Optional[float] = None
    severity: str = "medium"  # low, medium, high, critical
    priority: int = 5  # 1-10, higher is more important
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type.name,
            "involved_agents": self.involved_agents,
            "description": self.description,
            "context": self.context,
            "created_at": self.created_at,
            "resolved": self.resolved,
            "resolution": self.resolution,
            "resolution_time": self.resolution_time,
            "severity": self.severity,
            "priority": self.priority
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conflict':
        """Create from dictionary"""
        data = data.copy()
        data["conflict_type"] = ConflictType[data["conflict_type"]]
        return cls(**data)

class ConflictResolver:
    """Automated conflict resolution system"""
    
    def __init__(self):
        self.active_conflicts: Dict[str, Conflict] = {}
        self.resolution_strategies: Dict[ConflictType, Callable] = {
            ConflictType.RESOURCE_CONTENTION: self._resolve_resource_contention,
            ConflictType.TASK_ASSIGNMENT: self._resolve_task_assignment,
            ConflictType.STATE_INCONSISTENCY: self._resolve_state_inconsistency,
            ConflictType.GOAL_CONFLICT: self._resolve_goal_conflict,
            ConflictType.PRIORITY_DISPUTE: self._resolve_priority_dispute,
            ConflictType.COMMUNICATION_FAILURE: self._resolve_communication_failure,
            ConflictType.PERFORMANCE_DISPUTE: self._resolve_performance_dispute,
            ConflictType.POLICY_VIOLATION: self._resolve_policy_violation
        }
        self.resolution_history: deque = deque(maxlen=1000)
        self.negotiation_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.auto_resolve_enabled = True
        self.negotiation_timeout = 300  # 5 minutes
        self.max_negotiation_rounds = 10
        
        # Statistics
        self.stats = {
            "total_conflicts": 0,
            "resolved_conflicts": 0,
            "failed_resolutions": 0,
            "average_resolution_time": 0.0,
            "conflicts_by_type": defaultdict(int)
        }
        
    async def report_conflict(self, conflict: Conflict) -> str:
        """Report a conflict for resolution"""
        self.active_conflicts[conflict.conflict_id] = conflict
        self.stats["total_conflicts"] += 1
        self.stats["conflicts_by_type"][conflict.conflict_type.name] += 1
        
        logger.info(f"Conflict reported: {conflict.conflict_type.name} "
                   f"involving {conflict.involved_agents} (severity: {conflict.severity})")
        
        # Attempt automatic resolution
        if self.auto_resolve_enabled:
            await self.resolve_conflict(conflict.conflict_id)
        
        return conflict.conflict_id
    
    async def resolve_conflict(self, conflict_id: str) -> bool:
        """Attempt to resolve a conflict"""
        if conflict_id not in self.active_conflicts:
            return False
        
        conflict = self.active_conflicts[conflict_id]
        start_time = time.time()
        
        # Get resolution strategy
        strategy = self.resolution_strategies.get(conflict.conflict_type)
        if not strategy:
            logger.error("No resolution strategy for {}", extra={"conflict_conflict_type": conflict.conflict_type})
            self.stats["failed_resolutions"] += 1
            return False
        
        try:
            # Apply resolution strategy
            resolution = await strategy(conflict)
            
            if resolution:
                conflict.resolved = True
                conflict.resolution = resolution
                conflict.resolution_time = time.time()
                
                # Record in history
                resolution_record = {
                    "conflict_id": conflict_id,
                    "resolution": resolution,
                    "resolution_time": conflict.resolution_time - conflict.created_at,
                    "timestamp": time.time()
                }
                self.resolution_history.append(resolution_record)
                
                # Update statistics
                self.stats["resolved_conflicts"] += 1
                resolution_time = conflict.resolution_time - conflict.created_at
                
                # Update average resolution time
                total_resolved = self.stats["resolved_conflicts"]
                self.stats["average_resolution_time"] = (
                    (self.stats["average_resolution_time"] * (total_resolved - 1) + resolution_time) / total_resolved
                )
                
                # Remove from active conflicts
                del self.active_conflicts[conflict_id]
                
                logger.info("Conflict {} resolved: {}", extra={"conflict_id": conflict_id, "resolution_get__strategy____unknown__": resolution.get('strategy', 'unknown')})
                return True
            else:
                self.stats["failed_resolutions"] += 1
                logger.warning("Failed to resolve conflict {}", extra={"conflict_id": conflict_id})
                return False
                
        except Exception as e:
            logger.error("Error resolving conflict {}: {}", extra={"conflict_id": conflict_id, "e": e})
            self.stats["failed_resolutions"] += 1
            return False
    
    async def _resolve_resource_contention(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Resolve resource contention conflicts"""
        # Extract resource information
        resource = conflict.context.get("resource")
        competing_agents = conflict.involved_agents
        
        # Priority-based resolution
        agent_priorities = []
        for agent_id in competing_agents:
            priority = conflict.context.get(f"{agent_id}_priority", 5)
            agent_priorities.append((priority, agent_id))
        
        # Sort by priority (higher first)
        agent_priorities.sort(reverse=True)
        
        # Allocate to highest priority agent
        winner = agent_priorities[0][1]
        
        return {
            "strategy": "priority_based",
            "winner": winner,
            "resource": resource,
            "reason": f"Agent {winner} has highest priority ({agent_priorities[0][0]})",
            "allocation": {
                "agent_id": winner,
                "resource": resource,
                "duration": conflict.context.get("duration", 300)
            }
        }
    
    async def _resolve_task_assignment(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Resolve task assignment conflicts"""
        task = conflict.context.get("task")
        competing_agents = conflict.involved_agents
        
        # Performance-based resolution
        agent_scores = []
        for agent_id in competing_agents:
            # Get agent performance score
            score = conflict.context.get(f"{agent_id}_performance", 0.5)
            agent_scores.append((score, agent_id))
        
        # Sort by score (higher first)
        agent_scores.sort(reverse=True)
        
        # Assign to best performing agent
        winner = agent_scores[0][1]
        
        return {
            "strategy": "performance_based",
            "winner": winner,
            "task": task,
            "reason": f"Agent {winner} has best performance score ({agent_scores[0][0]:.2f})",
            "assignment": {
                "agent_id": winner,
                "task_id": task.get("task_id") if task else None,
                "estimated_duration": task.get("estimated_duration") if task else None
            }
        }
    
    async def _resolve_state_inconsistency(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Resolve state inconsistency conflicts"""
        # Use consensus mechanism
        states = conflict.context.get("states", {})
        
        # Find most common state value
        state_counts = defaultdict(int)
        for agent_id, state in states.items():
            state_str = str(state)
            state_counts[state_str] += 1
        
        # Get consensus state
        consensus_state = max(state_counts.items(), key=lambda x: x[1])[0]
        
        return {
            "strategy": "consensus",
            "consensus_state": consensus_state,
            "votes": dict(state_counts),
            "reason": f"Majority consensus ({state_counts[consensus_state]} votes)",
            "state_update": {
                "new_state": consensus_state,
                "applied_to": list(states.keys())
            }
        }
    
    async def _resolve_goal_conflict(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Resolve goal conflicts between agents"""
        # Hierarchical resolution based on goal importance
        goals = conflict.context.get("goals", {})
        
        # Rank goals by importance
        ranked_goals = []
        for agent_id, goal in goals.items():
            importance = goal.get("importance", 5)
            ranked_goals.append((importance, agent_id, goal))
        
        ranked_goals.sort(reverse=True)
        
        # Select most important goal
        winner_importance, winner_agent, winner_goal = ranked_goals[0]
        
        return {
            "strategy": "hierarchical",
            "selected_goal": winner_goal,
            "selected_agent": winner_agent,
            "reason": f"Goal has highest importance: {winner_importance}",
            "goal_execution": {
                "agent_id": winner_agent,
                "goal": winner_goal,
                "priority": winner_importance
            }
        }
    
    async def _resolve_priority_dispute(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Resolve priority disputes"""
        # Use round-robin or fair queuing
        agents = conflict.involved_agents
        
        # Simple round-robin assignment
        current_time = int(time.time())
        selected_index = current_time % len(agents)
        selected_agent = agents[selected_index]
        
        return {
            "strategy": "round_robin",
            "selected_agent": selected_agent,
            "reason": "Fair rotation among agents",
            "rotation": {
                "agent_id": selected_agent,
                "rotation_index": selected_index,
                "total_agents": len(agents)
            }
        }
    
    async def _resolve_communication_failure(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Resolve communication failures"""
        failed_agents = conflict.involved_agents
        failure_type = conflict.context.get("failure_type", "timeout")
        
        # Retry with exponential backoff
        retry_config = {
            "max_retries": 3,
            "base_delay": 5,
            "max_delay": 60
        }
        
        return {
            "strategy": "retry_with_backoff",
            "retry_config": retry_config,
            "reason": f"Communication failure: {failure_type}",
            "recovery": {
                "action": "retry_communication",
                "agents": failed_agents,
                "backoff_delay": retry_config["base_delay"]
            }
        }
    
    async def _resolve_performance_dispute(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Resolve performance disputes"""
        agents = conflict.involved_agents
        performance_metrics = conflict.context.get("performance_metrics", {})
        
        # Compare performance metrics
        agent_performance = []
        for agent_id in agents:
            metrics = performance_metrics.get(agent_id, {})
            avg_score = metrics.get("average_score", 0.5)
            agent_performance.append((avg_score, agent_id))
        
        # Select best performing agent
        agent_performance.sort(reverse=True)
        winner = agent_performance[0][1]
        
        return {
            "strategy": "performance_comparison",
            "winner": winner,
            "reason": f"Agent {winner} has best performance ({agent_performance[0][0]:.2f})",
            "performance_analysis": {
                "winner": winner,
                "performance_scores": dict(agent_performance),
                "recommendation": "Use winner for similar tasks"
            }
        }
    
    async def _resolve_policy_violation(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Resolve policy violations"""
        violating_agent = conflict.involved_agents[0]
        violation_type = conflict.context.get("violation_type", "unknown")
        severity = conflict.context.get("severity", "medium")
        
        # Determine action based on severity
        if severity == "critical":
            action = "suspend_agent"
        elif severity == "high":
            action = "warn_and_monitor"
        else:
            action = "warn"
        
        return {
            "strategy": "policy_enforcement",
            "action": action,
            "violating_agent": violating_agent,
            "reason": f"Policy violation: {violation_type} (severity: {severity})",
            "enforcement": {
                "agent_id": violating_agent,
                "action": action,
                "violation_type": violation_type,
                "severity": severity,
                "duration": 3600 if action == "suspend_agent" else None
            }
        }
    
    async def start_negotiation(self, conflict_id: str, 
                              negotiation_config: Dict[str, Any]) -> str:
        """Start a negotiation session for conflict resolution"""
        if conflict_id not in self.active_conflicts:
            raise ValueError(f"Conflict {conflict_id} not found")
        
        negotiation_id = f"negotiation_{uuid.uuid4().hex[:8]}"
        
        self.negotiation_sessions[negotiation_id] = {
            "conflict_id": conflict_id,
            "start_time": time.time(),
            "rounds": 0,
            "proposals": {},
            "config": negotiation_config,
            "status": "active"
        }
        
        logger.info("Started negotiation {} for conflict {}", extra={"negotiation_id": negotiation_id, "conflict_id": conflict_id})
        return negotiation_id
    
    async def submit_proposal(self, negotiation_id: str, agent_id: str, 
                            proposal: Dict[str, Any]) -> bool:
        """Submit a proposal in a negotiation session"""
        if negotiation_id not in self.negotiation_sessions:
            return False
        
        session = self.negotiation_sessions[negotiation_id]
        session["proposals"][agent_id] = {
            "proposal": proposal,
            "timestamp": time.time(),
            "round": session["rounds"]
        }
        
        logger.debug("Agent {} submitted proposal in negotiation {}", extra={"agent_id": agent_id, "negotiation_id": negotiation_id})
        return True
    
    async def evaluate_negotiation(self, negotiation_id: str) -> Optional[Dict[str, Any]]:
        """Evaluate a negotiation session and determine outcome"""
        if negotiation_id not in self.negotiation_sessions:
            return None
        
        session = self.negotiation_sessions[negotiation_id]
        proposals = session["proposals"]
        
        if len(proposals) < 2:
            return None  # Need at least 2 proposals
        
        # Simple evaluation - find most common elements
        common_elements = {}
        for agent_id, proposal_data in proposals.items():
            proposal = proposal_data["proposal"]
            for key, value in proposal.items():
                if key not in common_elements:
                    common_elements[key] = defaultdict(int)
                common_elements[key][str(value)] += 1
        
        # Build consensus proposal
        consensus = {}
        for key, value_counts in common_elements.items():
            most_common = max(value_counts.items(), key=lambda x: x[1])
            consensus[key] = most_common[0]
        
        return {
            "negotiation_id": negotiation_id,
            "consensus": consensus,
            "participants": list(proposals.keys()),
            "rounds": session["rounds"]
        }
    
    def add_resolution_strategy(self, conflict_type: ConflictType,
                              strategy: Callable):
        """Add a custom resolution strategy"""
        self.resolution_strategies[conflict_type] = strategy
        logger.info("Added custom resolution strategy for {}", extra={"conflict_type_name": conflict_type.name})
    
    def get_conflict_stats(self) -> Dict[str, Any]:
        """Get conflict resolution statistics"""
        return {
            **self.stats,
            "active_conflicts": len(self.active_conflicts),
            "active_negotiations": len([s for s in self.negotiation_sessions.values() 
                                      if s["status"] == "active"]),
            "resolution_success_rate": (self.stats["resolved_conflicts"] / 
                                      max(1, self.stats["total_conflicts"]))
        }
    
    async def get_conflict_history(self, agent_id: Optional[str] = None,
                                 conflict_type: Optional[ConflictType] = None,
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """Get conflict resolution history"""
        history = []
        
        # Get resolved conflicts from history
        for record in self.resolution_history:
            conflict_id = record["conflict_id"]
            if conflict_id in self.active_conflicts:
                conflict = self.active_conflicts[conflict_id]
            else:
                # Conflict was resolved and removed
                continue
            
            # Apply filters
            if agent_id and agent_id not in conflict.involved_agents:
                continue
            
            if conflict_type and conflict.conflict_type != conflict_type:
                continue
            
            history.append({
                "conflict_id": conflict_id,
                "conflict_type": conflict.conflict_type.name,
                "involved_agents": conflict.involved_agents,
                "description": conflict.description,
                "resolution": record["resolution"],
                "resolution_time": record["resolution_time"],
                "timestamp": record["timestamp"]
            })
            
            if len(history) >= limit:
                break
        
        return history
    
    async def cleanup_old_conflicts(self, max_age: float = 86400):
        """Clean up old resolved conflicts"""
        current_time = time.time()
        conflicts_to_remove = []
        
        for conflict_id, conflict in self.active_conflicts.items():
            if conflict.resolved and (current_time - conflict.created_at) > max_age:
                conflicts_to_remove.append(conflict_id)
        
        for conflict_id in conflicts_to_remove:
            del self.active_conflicts[conflict_id]
        
        if conflicts_to_remove:
            logger.info("Cleaned up {} old conflicts", extra={"len_conflicts_to_remove_": len(conflicts_to_remove)}) 