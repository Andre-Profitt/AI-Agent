"""
Collaboration Dashboard for Multi-Agent System

This module provides monitoring and analytics:
- System-wide collaboration metrics
- Agent performance visualization
- Network analysis and clustering
- Real-time monitoring capabilities
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging

from .registry import AgentRegistry
from .performance import PerformanceTracker

logger = logging.getLogger(__name__)

class CollaborationDashboard:
    """Dashboard for monitoring multi-agent collaboration"""
    
    def __init__(self, registry: AgentRegistry, 
                 performance_tracker: PerformanceTracker):
        self.registry = registry
        self.performance_tracker = performance_tracker
        self.metrics_cache: Dict[str, Any] = {}
        self.cache_ttl = 60  # 1 minute
        self.last_cache_update = 0
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_interval = 30  # seconds
        self.alert_thresholds = {
            "low_success_rate": 0.7,
            "high_latency": 300.0,  # 5 minutes
            "resource_utilization": 0.9,  # 90%
            "agent_failures": 3  # consecutive failures
        }
        
        # Historical data
        self.historical_metrics: deque = deque(maxlen=1000)
        
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get system-wide collaboration metrics"""
        # Check cache
        if time.time() - self.last_cache_update < self.cache_ttl:
            return self.metrics_cache
        
        # Calculate fresh metrics
        agents = list(self.registry.registry.values())
        
        overview = {
            "total_agents": len(agents),
            "active_agents": sum(1 for a in agents if a.status.value in [1, 2]),  # IDLE, AVAILABLE
            "busy_agents": sum(1 for a in agents if a.status.value == 2),  # BUSY
            "offline_agents": sum(1 for a in agents if a.status.value == 3),  # OFFLINE
            "agent_breakdown": self._get_agent_breakdown(agents),
            "collaboration_network": self._get_collaboration_network(),
            "performance_summary": self._get_performance_summary(),
            "resource_utilization": self._get_resource_utilization(),
            "alerts": await self._get_active_alerts(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Update cache
        self.metrics_cache = overview
        self.last_cache_update = time.time()
        
        # Store in historical data
        self.historical_metrics.append({
            "timestamp": time.time(),
            "overview": overview
        })
        
        return overview
    
    def _get_agent_breakdown(self, agents: List[Any]) -> Dict[str, Any]:
        """Get breakdown of agents by capability and status"""
        capability_counts = defaultdict(int)
        status_counts = defaultdict(int)
        
        for agent in agents:
            for capability in agent.capabilities:
                capability_counts[capability.name] += 1
            
            status_counts[agent.status.name] += 1
        
        return {
            "by_capability": dict(capability_counts),
            "by_status": dict(status_counts)
        }
    
    def _get_collaboration_network(self) -> Dict[str, Any]:
        """Get collaboration network metrics"""
        network = {
            "nodes": [],
            "edges": [],
            "clusters": []
        }
        
        # Add nodes (agents)
        for agent_id in self.registry.registry:
            metadata = self.registry.registry[agent_id]
            metrics = self.performance_tracker.get_agent_metrics(agent_id)
            
            network["nodes"].append({
                "id": agent_id,
                "name": metadata.name,
                "status": metadata.status.name,
                "capabilities": [cap.name for cap in metadata.capabilities],
                "success_rate": metrics.get("success_rate", 0.0),
                "total_tasks": metrics.get("total_tasks", 0),
                "reliability_score": metadata.reliability_score
            })
        
        # Add edges (collaborations)
        for (agent1, agent2), metrics in self.performance_tracker.collaboration_metrics.items():
            network["edges"].append({
                "source": agent1,
                "target": agent2,
                "weight": metrics.get("avg_score", 0.0),
                "count": metrics.get("total_collaborations", 0)
            })
        
        # Identify clusters (simplified - agents with same capabilities)
        capability_clusters = defaultdict(list)
        for agent_id, metadata in self.registry.registry.items():
            cap_key = tuple(sorted(cap.name for cap in metadata.capabilities))
            capability_clusters[cap_key].append(agent_id)
        
        for cap_key, agent_ids in capability_clusters.items():
            if len(agent_ids) > 1:
                network["clusters"].append({
                    "capabilities": list(cap_key),
                    "agents": agent_ids,
                    "size": len(agent_ids)
                })
        
        return network
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        all_exec_times = []
        all_successes = []
        
        for times in self.performance_tracker.execution_times.values():
            all_exec_times.extend(times)
        
        for successes in self.performance_tracker.success_rates.values():
            all_successes.extend(successes)
        
        if not all_exec_times:
            return {
                "avg_execution_time": 0.0,
                "overall_success_rate": 0.0,
                "total_tasks_completed": 0,
                "execution_time_percentiles": {
                    "p50": 0.0,
                    "p90": 0.0,
                    "p99": 0.0
                }
            }
        
        return {
            "avg_execution_time": np.mean(all_exec_times),
            "overall_success_rate": sum(all_successes) / len(all_successes),
            "total_tasks_completed": len(all_exec_times),
            "execution_time_percentiles": {
                "p50": np.percentile(all_exec_times, 50),
                "p90": np.percentile(all_exec_times, 90),
                "p99": np.percentile(all_exec_times, 99)
            }
        }
    
    def _get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization"""
        # This would integrate with resource manager
        # For now, return simplified metrics
        return {
            "cpu_utilization": 0.0,
            "memory_utilization": 0.0,
            "gpu_utilization": 0.0,
            "network_utilization": 0.0
        }
    
    async def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active system alerts"""
        alerts = []
        
        # Check performance alerts
        performance_alerts = self.performance_tracker.get_performance_alerts(
            time_window=3600  # Last hour
        )
        
        for alert in performance_alerts:
            alerts.append({
                "type": "performance",
                "severity": "warning",
                "message": f"Performance alert: {alert.get('type', 'unknown')}",
                "agent_id": alert.get("agent_id"),
                "timestamp": alert.get("timestamp", time.time())
            })
        
        # Check agent health
        health_results = await self.registry.check_agent_health()
        for agent_id, health in health_results.items():
            if health.get("status") == "offline":
                alerts.append({
                    "type": "agent_health",
                    "severity": "error",
                    "message": f"Agent {agent_id} is offline",
                    "agent_id": agent_id,
                    "timestamp": time.time()
                })
        
        return alerts
    
    async def get_agent_details(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed metrics for a specific agent"""
        metadata = await self.registry.get_agent(agent_id)
        if not metadata:
            return None
        
        metrics = self.performance_tracker.get_agent_metrics(agent_id)
        
        # Get collaboration partners
        collaborations = []
        for (agent1, agent2), collab_metrics in self.performance_tracker.collaboration_metrics.items():
            if agent_id in (agent1, agent2):
                partner = agent2 if agent1 == agent_id else agent1
                collaborations.append({
                    "partner": partner,
                    "score": collab_metrics.get("avg_score", 0.0),
                    "count": collab_metrics.get("total_collaborations", 0)
                })
        
        # Get recent task history
        recent_tasks = []
        agent_history = self.performance_tracker.agent_performance_history.get(agent_id, [])
        for entry in list(agent_history)[-10:]:  # Last 10 tasks
            recent_tasks.append({
                "task_type": entry.get("task_type"),
                "execution_time": entry.get("execution_time"),
                "success": entry.get("success"),
                "timestamp": entry.get("timestamp")
            })
        
        return {
            "metadata": {
                "name": metadata.name,
                "version": metadata.version,
                "capabilities": [cap.name for cap in metadata.capabilities],
                "status": metadata.status.name,
                "reliability_score": metadata.reliability_score,
                "description": metadata.description
            },
            "performance": metrics,
            "collaborations": collaborations,
            "recent_tasks": recent_tasks,
            "last_seen": metadata.last_seen.isoformat()
        }
    
    async def get_capability_analysis(self, capability: str) -> Dict[str, Any]:
        """Get analysis for a specific capability"""
        agents_with_capability = await self.registry.get_agents_by_capability(
            getattr(self.registry.registry[list(self.registry.registry.keys())[0]].capabilities[0].__class__, capability, None)
        )
        
        if not agents_with_capability:
            return {
                "capability": capability,
                "total_agents": 0,
                "performance": {},
                "utilization": 0.0
            }
        
        # Aggregate performance metrics
        total_tasks = 0
        total_successes = 0
        total_execution_time = 0.0
        
        for agent in agents_with_capability:
            metrics = self.performance_tracker.get_agent_metrics(agent.agent_id)
            total_tasks += metrics.get("total_tasks", 0)
            total_successes += int(metrics.get("success_rate", 0) * metrics.get("total_tasks", 0))
            total_execution_time += metrics.get("avg_execution_time", 0) * metrics.get("total_tasks", 0)
        
        return {
            "capability": capability,
            "total_agents": len(agents_with_capability),
            "performance": {
                "total_tasks": total_tasks,
                "success_rate": total_successes / max(1, total_tasks),
                "avg_execution_time": total_execution_time / max(1, total_tasks)
            },
            "utilization": len([a for a in agents_with_capability if a.status.value in [1, 2]]) / len(agents_with_capability)
        }
    
    async def get_trend_analysis(self, metric: str, time_window: int = 3600) -> Dict[str, Any]:
        """Get trend analysis for a specific metric"""
        if not self.historical_metrics:
            return {
                "metric": metric,
                "trend": "insufficient_data",
                "values": [],
                "timestamps": []
            }
        
        # Extract metric values from historical data
        values = []
        timestamps = []
        cutoff_time = time.time() - time_window
        
        for record in self.historical_metrics:
            if record["timestamp"] >= cutoff_time:
                overview = record["overview"]
                
                if metric == "total_agents":
                    values.append(overview["total_agents"])
                elif metric == "active_agents":
                    values.append(overview["active_agents"])
                elif metric == "success_rate":
                    values.append(overview["performance_summary"]["overall_success_rate"])
                elif metric == "avg_execution_time":
                    values.append(overview["performance_summary"]["avg_execution_time"])
                
                timestamps.append(record["timestamp"])
        
        if len(values) < 2:
            return {
                "metric": metric,
                "trend": "insufficient_data",
                "values": values,
                "timestamps": timestamps
            }
        
        # Calculate trend
        if values[-1] > values[0] * 1.1:
            trend = "increasing"
        elif values[-1] < values[0] * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "metric": metric,
            "trend": trend,
            "values": values,
            "timestamps": timestamps,
            "current_value": values[-1] if values else 0,
            "change_percent": ((values[-1] - values[0]) / values[0] * 100) if values and values[0] != 0 else 0
        }
    
    async def start_monitoring(self):
        """Start real-time monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("Started real-time monitoring")
        
        while self.monitoring_active:
            try:
                # Update metrics
                await self.get_system_overview()
                
                # Check for alerts
                alerts = await self._get_active_alerts()
                if alerts:
                    logger.warning(f"Active alerts: {len(alerts)}")
                    for alert in alerts:
                        logger.warning(f"Alert: {alert['message']}")
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        logger.info("Stopped real-time monitoring")
    
    async def export_dashboard_data(self, format: str = "json") -> Dict[str, Any]:
        """Export dashboard data for external analysis"""
        overview = await self.get_system_overview()
        
        export_data = {
            "overview": overview,
            "agents": {},
            "capabilities": {},
            "trends": {}
        }
        
        # Export agent details
        for agent_id in self.registry.registry:
            agent_details = await self.get_agent_details(agent_id)
            if agent_details:
                export_data["agents"][agent_id] = agent_details
        
        # Export capability analysis
        for capability in ["REASONING", "TOOL_USE", "COLLABORATION", "PLANNING"]:
            capability_analysis = await self.get_capability_analysis(capability)
            export_data["capabilities"][capability] = capability_analysis
        
        # Export trend analysis
        for metric in ["total_agents", "active_agents", "success_rate", "avg_execution_time"]:
            trend_analysis = await self.get_trend_analysis(metric)
            export_data["trends"][metric] = trend_analysis
        
        export_data["exported_at"] = datetime.utcnow().isoformat()
        
        return export_data
    
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics"""
        return {
            "cache_size": len(self.metrics_cache),
            "historical_records": len(self.historical_metrics),
            "monitoring_active": self.monitoring_active,
            "last_update": self.last_cache_update,
            "cache_ttl": self.cache_ttl
        } 