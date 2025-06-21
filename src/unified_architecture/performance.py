from app import history
from benchmarks.cot_performance import duration
from examples.enhanced_unified_example import execution_time
from examples.enhanced_unified_example import metrics
from examples.parallel_execution_example import summary
from performance_dashboard import alerts
from performance_dashboard import cutoff_time
from tests.load_test import success

from src.core.monitoring import key
from src.core.optimized_chain_of_thought import n
from src.database.models import agent_id
from src.database.models import resource_type
from src.gaia_components.adaptive_tool_system import recent_success_rate
from src.gaia_components.enhanced_memory_system import current_time
from src.gaia_components.monitoring import values
from src.gaia_components.performance_optimization import entry
from src.gaia_components.performance_optimization import successes
from src.services.integration_hub import limit
from src.tools.registry import current_avg
from src.unified_architecture.dashboard import all_exec_times
from src.unified_architecture.dashboard import all_successes
from src.unified_architecture.dashboard import partner
from src.unified_architecture.enhanced_platform import agent1
from src.unified_architecture.enhanced_platform import agent2
from src.unified_architecture.enhanced_platform import all_times
from src.unified_architecture.enhanced_platform import exec_times
from src.unified_architecture.enhanced_platform import task_type
from src.unified_architecture.performance import all_costs
from src.unified_architecture.performance import all_quality_scores
from src.unified_architecture.performance import avg_usage
from src.unified_architecture.performance import collaborators
from src.unified_architecture.performance import costs
from src.unified_architecture.performance import history_entry
from src.unified_architecture.performance import quality_metrics
from src.unified_architecture.performance import quality_scores
from src.unified_architecture.performance import recent_history
from src.unified_architecture.performance import recent_successes
from src.unified_architecture.performance import recent_times
from src.unified_architecture.performance import resource_types
from src.unified_architecture.performance import resource_usage
from src.unified_architecture.performance import task_metrics
from src.unified_architecture.performance import total_execs
from src.unified_architecture.performance import usage_data
from src.unified_architecture.resource_management import usage

from src.agents.advanced_agent_fsm import Agent
# TODO: Fix undefined variables: Any, Dict, List, Optional, Tuple, a, agent1, agent1_id, agent2, agent2_id, agent_id, alert, alert_type, alerts, all_costs, all_exec_times, all_quality_scores, all_successes, all_times, avg_usage, collaboration_score, collaborators, cost, costs, current_avg, current_time, cutoff_time, defaultdict, deque, duration, entry, exec_times, execution_time, history, history_entry, history_size, key, limit, logging, m, metrics, n, partner, quality_metrics, quality_score, quality_scores, recent_history, recent_success_rate, recent_successes, recent_times, resource_type, resource_types, resource_usage, success, successes, summary, task_metrics, task_type, time, time_window, total_execs, usage, usage_data, values, x

"""
from collections import deque
from typing import Optional
# TODO: Fix undefined variables: a, agent1, agent1_id, agent2, agent2_id, agent_id, alert, alert_type, alerts, all_costs, all_exec_times, all_quality_scores, all_successes, all_times, avg_usage, collaboration_score, collaborators, cost, costs, current_avg, current_time, cutoff_time, duration, entry, exec_times, execution_time, history, history_entry, history_size, key, limit, m, metrics, n, partner, quality_metrics, quality_score, quality_scores, recent_history, recent_success_rate, recent_successes, recent_times, resource_type, resource_types, resource_usage, self, success, successes, summary, task_metrics, task_type, time_window, total_execs, usage, usage_data, values, x

Performance Tracking for Multi-Agent System

This module provides comprehensive performance monitoring:
- Task execution metrics
- Agent performance tracking
- Collaboration effectiveness
- Resource utilization monitoring
"""

from typing import Tuple
from typing import Any
from typing import List

import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque

import logging

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Tracks performance metrics for agents and tasks"""

    def __init__(self, history_size: int = 1000) -> None:
        self.execution_times: Dict[str, List[float]] = defaultdict(list)
        self.success_rates: Dict[str, List[bool]] = defaultdict(list)
        self.resource_usage: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        self.collaboration_metrics: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
        self.task_type_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.agent_performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))

        # Performance thresholds
        self.performance_thresholds = {
            "min_success_rate": 0.7,
            "max_avg_execution_time": 300.0,  # 5 minutes
            "min_reliability_score": 0.8
        }

        # Alert tracking
        self.performance_alerts: deque = deque(maxlen=100)

    def record_execution(self, agent_id: str, task_type: str,
                        execution_time: float, success: bool,
                        resource_usage: Optional[Dict[str, float]] = None,
                        quality_score: Optional[float] = None,
                        cost: Optional[float] = None) -> Any:
        """Record task execution metrics"""
        key = f"{agent_id}:{task_type}"

        # Record basic metrics
        self.execution_times[key].append(execution_time)
        self.success_rates[key].append(success)

        # Record resource usage if provided
        if resource_usage:
            self.resource_usage[key].append(resource_usage)

        # Record in agent history
        history_entry = {
            "timestamp": time.time(),
            "task_type": task_type,
            "execution_time": execution_time,
            "success": success,
            "quality_score": quality_score,
            "cost": cost,
            "resource_usage": resource_usage
        }
        self.agent_performance_history[agent_id].append(history_entry)

        # Update task type metrics
        if task_type not in self.task_type_metrics:
            self.task_type_metrics[task_type] = {
                "total_executions": 0,
                "successful_executions": 0,
                "total_execution_time": 0.0,
                "avg_execution_time": 0.0,
                "success_rate": 0.0,
                "total_cost": 0.0,
                "avg_quality_score": 0.0
            }

        task_metrics = self.task_type_metrics[task_type]
        task_metrics["total_executions"] += 1
        task_metrics["total_execution_time"] += execution_time

        if success:
            task_metrics["successful_executions"] += 1

        if cost:
            task_metrics["total_cost"] += cost

        if quality_score:
            current_avg = task_metrics["avg_quality_score"]
            total_execs = task_metrics["total_executions"]
            task_metrics["avg_quality_score"] = (
                (current_avg * (total_execs - 1) + quality_score) / total_execs
            )

        # Update averages
        task_metrics["avg_execution_time"] = task_metrics["total_execution_time"] / task_metrics["total_executions"]
        task_metrics["success_rate"] = task_metrics["successful_executions"] / task_metrics["total_executions"]

        # Check for performance alerts
        self._check_performance_alerts(agent_id, task_type, execution_time, success)

        # Keep only recent history
        if len(self.execution_times[key]) > 1000:
            self.execution_times[key] = self.execution_times[key][-1000:]
            self.success_rates[key] = self.success_rates[key][-1000:]

    def get_agent_metrics(self, agent_id: str,
                         time_window: Optional[float] = None) -> Dict[str, Any]:
        """Get performance metrics for an agent"""
        metrics = {
            "avg_execution_time": 0.0,
            "success_rate": 0.0,
            "total_tasks": 0,
            "task_breakdown": {},
            "recent_performance": {},
            "resource_usage": {},
            "quality_metrics": {}
        }

        # Filter by time window if specified
        current_time = time.time()
        if time_window:
            cutoff_time = current_time - time_window
        else:
            cutoff_time = 0

        # Aggregate metrics across all task types
        for key in self.execution_times:
            if key.startswith(f"{agent_id}:"):
                task_type = key.split(":")[1]

                # Filter by time window
                if time_window:
                    # This is a simplified approach - in practice you'd need timestamps
                    # For now, we'll use all data
                    pass

                exec_times = self.execution_times[key]
                successes = self.success_rates[key]

                if exec_times:
                    task_metrics = {
                        "avg_time": np.mean(exec_times),
                        "success_rate": sum(successes) / len(successes),
                        "count": len(exec_times),
                        "min_time": min(exec_times),
                        "max_time": max(exec_times),
                        "std_time": np.std(exec_times)
                    }

                    metrics["task_breakdown"][task_type] = task_metrics
                    metrics["total_tasks"] += len(exec_times)

        # Calculate overall metrics
        if metrics["total_tasks"] > 0:
            all_times = []
            all_successes = []

            for key in self.execution_times:
                if key.startswith(f"{agent_id}:"):
                    all_times.extend(self.execution_times[key])
                    all_successes.extend(self.success_rates[key])

            metrics["avg_execution_time"] = np.mean(all_times)
            metrics["success_rate"] = sum(all_successes) / len(all_successes)
            metrics["min_execution_time"] = min(all_times)
            metrics["max_execution_time"] = max(all_times)
            metrics["std_execution_time"] = np.std(all_times)

        # Get recent performance (last 10 tasks)
        recent_history = list(self.agent_performance_history[agent_id])[-10:]
        if recent_history:
            recent_times = [entry["execution_time"] for entry in recent_history]
            recent_successes = [entry["success"] for entry in recent_history]

            metrics["recent_performance"] = {
                "avg_execution_time": np.mean(recent_times),
                "success_rate": sum(recent_successes) / len(recent_successes),
                "task_count": len(recent_history)
            }

        # Get resource usage patterns
        resource_usage = self._get_agent_resource_usage(agent_id)
        metrics["resource_usage"] = resource_usage

        # Get quality metrics
        quality_metrics = self._get_agent_quality_metrics(agent_id)
        metrics["quality_metrics"] = quality_metrics

        return metrics

    def _get_agent_resource_usage(self, agent_id: str) -> Dict[str, Any]:
        """Get resource usage patterns for an agent"""
        usage_data = []

        for key in self.resource_usage:
            if key.startswith(f"{agent_id}:"):
                usage_data.extend(self.resource_usage[key])

        if not usage_data:
            return {}

        # Calculate averages for each resource type
        resource_types = set()
        for usage in usage_data:
            resource_types.update(usage.keys())

        avg_usage = {}
        for resource_type in resource_types:
            values = [usage.get(resource_type, 0) for usage in usage_data]
            avg_usage[resource_type] = {
                "average": np.mean(values),
                "max": max(values),
                "min": min(values),
                "std": np.std(values)
            }

        return avg_usage

    def _get_agent_quality_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get quality metrics for an agent"""
        quality_scores = []
        costs = []

        for entry in self.agent_performance_history[agent_id]:
            if entry.get("quality_score") is not None:
                quality_scores.append(entry["quality_score"])
            if entry.get("cost") is not None:
                costs.append(entry["cost"])

        metrics = {}

        if quality_scores:
            metrics["avg_quality_score"] = np.mean(quality_scores)
            metrics["min_quality_score"] = min(quality_scores)
            metrics["max_quality_score"] = max(quality_scores)
            metrics["quality_consistency"] = 1.0 - np.std(quality_scores)

        if costs:
            metrics["avg_cost"] = np.mean(costs)
            metrics["total_cost"] = sum(costs)
            metrics["cost_efficiency"] = np.mean(quality_scores) / np.mean(costs) if quality_scores else 0

        return metrics

    def record_collaboration(self, agent1_id: str, agent2_id: str,
                           collaboration_score: float,
                           task_type: str,
                           duration: float) -> Any:
        """Record collaboration effectiveness between agents"""
        key = tuple(sorted([agent1_id, agent2_id]))

        if key not in self.collaboration_metrics:
            self.collaboration_metrics[key] = {
                "total_collaborations": 0,
                "avg_score": 0.0,
                "total_duration": 0.0,
                "task_type_breakdown": defaultdict(int),
                "recent_scores": deque(maxlen=50)
            }

        metrics = self.collaboration_metrics[key]
        metrics["total_collaborations"] += 1
        metrics["total_duration"] += duration
        metrics["task_type_breakdown"][task_type] += 1
        metrics["recent_scores"].append(collaboration_score)

        # Update average score
        n = metrics["total_collaborations"]
        metrics["avg_score"] = ((n - 1) * metrics["avg_score"] + collaboration_score) / n

    def get_collaboration_metrics(self, agent1_id: str, agent2_id: str) -> Optional[Dict[str, Any]]:
        """Get collaboration metrics between two agents"""
        key = tuple(sorted([agent1_id, agent2_id]))
        return self.collaboration_metrics.get(key)

    def get_top_collaborators(self, agent_id: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Get top collaborators for an agent"""
        collaborators = []

        for (agent1, agent2), metrics in self.collaboration_metrics.items():
            if agent_id in (agent1, agent2):
                partner = agent2 if agent1 == agent_id else agent1
                collaborators.append((partner, metrics["avg_score"]))

        # Sort by collaboration score
        collaborators.sort(key=lambda x: x[1], reverse=True)
        return collaborators[:limit]

    def get_task_type_metrics(self, task_type: str) -> Dict[str, Any]:
        """Get metrics for a specific task type"""
        return self.task_type_metrics.get(task_type, {})

    def get_all_task_type_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all task types"""
        return dict(self.task_type_metrics)

    def _check_performance_alerts(self, agent_id: str, task_type: str,
                                execution_time: float, success: bool) -> Any:
        """Check for performance issues and generate alerts"""
        alerts = []

        # Check success rate
        key = f"{agent_id}:{task_type}"
        if key in self.success_rates:
            recent_successes = self.success_rates[key][-10:]  # Last 10 tasks
            if len(recent_successes) >= 5:
                recent_success_rate = sum(recent_successes) / len(recent_successes)
                if recent_success_rate < self.performance_thresholds["min_success_rate"]:
                    alerts.append({
                        "type": "low_success_rate",
                        "agent_id": agent_id,
                        "task_type": task_type,
                        "value": recent_success_rate,
                        "threshold": self.performance_thresholds["min_success_rate"],
                        "timestamp": time.time()
                    })

        # Check execution time
        if execution_time > self.performance_thresholds["max_avg_execution_time"]:
            alerts.append({
                "type": "high_execution_time",
                "agent_id": agent_id,
                "task_type": task_type,
                "value": execution_time,
                "threshold": self.performance_thresholds["max_avg_execution_time"],
                "timestamp": time.time()
            })

        # Add alerts to tracking
        for alert in alerts:
            self.performance_alerts.append(alert)
            logger.warning("Performance alert: {}", extra={"alert": alert})

    def get_performance_alerts(self, agent_id: Optional[str] = None,
                             alert_type: Optional[str] = None,
                             time_window: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get performance alerts with optional filtering"""
        alerts = list(self.performance_alerts)

        # Filter by agent
        if agent_id:
            alerts = [a for a in alerts if a.get("agent_id") == agent_id]

        # Filter by type
        if alert_type:
            alerts = [a for a in alerts if a.get("type") == alert_type]

        # Filter by time window
        if time_window:
            cutoff_time = time.time() - time_window
            alerts = [a for a in alerts if a.get("timestamp", 0) > cutoff_time]

        return alerts

    def get_system_performance_summary(self) -> Dict[str, Any]:
        """Get overall system performance summary"""
        all_exec_times = []
        all_successes = []
        all_costs = []
        all_quality_scores = []

        # Aggregate all metrics
        for key in self.execution_times:
            all_exec_times.extend(self.execution_times[key])
            all_successes.extend(self.success_rates[key])

        for history in self.agent_performance_history.values():
            for entry in history:
                if entry.get("cost"):
                    all_costs.append(entry["cost"])
                if entry.get("quality_score"):
                    all_quality_scores.append(entry["quality_score"])

        if not all_exec_times:
            return {
                "total_tasks": 0,
                "overall_success_rate": 0.0,
                "avg_execution_time": 0.0,
                "total_cost": 0.0,
                "avg_quality_score": 0.0
            }

        summary = {
            "total_tasks": len(all_exec_times),
            "overall_success_rate": sum(all_successes) / len(all_successes),
            "avg_execution_time": np.mean(all_exec_times),
            "execution_time_percentiles": {
                "p50": np.percentile(all_exec_times, 50),
                "p90": np.percentile(all_exec_times, 90),
                "p99": np.percentile(all_exec_times, 99)
            },
            "total_cost": sum(all_costs) if all_costs else 0.0,
            "avg_quality_score": np.mean(all_quality_scores) if all_quality_scores else 0.0,
            "active_agents": len(self.agent_performance_history),
            "total_collaborations": sum(m["total_collaborations"] for m in self.collaboration_metrics.values()),
            "recent_alerts": len([a for a in self.performance_alerts
                                if time.time() - a.get("timestamp", 0) < 3600])  # Last hour
        }

        return summary

    def reset_metrics(self, agent_id: Optional[str] = None) -> Any:
        """Reset metrics for an agent or all agents"""
        if agent_id:
            # Reset specific agent
            for key in list(self.execution_times.keys()):
                if key.startswith(f"{agent_id}:"):
                    del self.execution_times[key]
                    del self.success_rates[key]

            if agent_id in self.resource_usage:
                del self.resource_usage[agent_id]

            if agent_id in self.agent_performance_history:
                del self.agent_performance_history[agent_id]

            logger.info("Reset metrics for agent {}", extra={"agent_id": agent_id})
        else:
            # Reset all metrics
            self.execution_times.clear()
            self.success_rates.clear()
            self.resource_usage.clear()
            self.collaboration_metrics.clear()
            self.task_type_metrics.clear()
            self.agent_performance_history.clear()
            self.performance_alerts.clear()

            logger.info("Reset all performance metrics")