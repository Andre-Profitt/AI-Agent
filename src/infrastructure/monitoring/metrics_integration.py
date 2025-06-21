from benchmarks.cot_performance import duration
from examples.enhanced_unified_example import metrics
from examples.enhanced_unified_example import start_time
from examples.enhanced_unified_example import task
from examples.enhanced_unified_example import tasks
from examples.parallel_execution_example import summary
from tests.load_test import args
from tests.performance.performance_test import agent_data
from tests.performance.performance_test import task_data
from tests.unit.simple_test import func

from src.api_server import resources
from src.application.tools.tool_executor import operation
from src.database.models import agent_id
from src.database.models import agent_type
from src.database.models import priority
from src.database.models import status
from src.gaia_components.multi_agent_orchestrator import task_id
from src.infrastructure.monitoring.metrics_integration import agent_metadata
from src.infrastructure.monitoring.metrics_integration import available_resources
from src.infrastructure.monitoring.metrics_integration import platform_stats
from src.infrastructure.monitoring.metrics_integration import total_resources
from src.tools_introspection import error
from src.tools_introspection import error_type
from src.unified_architecture.enhanced_platform import task_type
from src.unified_architecture.resource_management import utilization

from src.agents.advanced_agent_fsm import AgentStatus

from src.agents.advanced_agent_fsm import Agent

from src.agents.advanced_agent_fsm import IUnifiedAgent
# TODO: Fix undefined variables: Any, Dict, Optional, activity_type, agent_data, agent_id, agent_metadata, agent_type, args, asynccontextmanager, available_resources, datetime, duration, e, error, error_type, func, kwargs, logging, metrics, metrics_integration, operation, operation_type, platform_stats, priority, resources, result, start_time, status, summary, task, task_data, task_id, task_type, tasks, time, total_resources, utilization
from tests.test_gaia_agent import agent

from src.infrastructure.monitoring.metrics import record_agent_availability
from src.infrastructure.monitoring.metrics import record_agent_registration
from src.infrastructure.monitoring.metrics import record_error
from src.infrastructure.monitoring.metrics import record_task_completion
from src.infrastructure.monitoring.metrics import record_task_duration
from src.infrastructure.monitoring.metrics import record_task_execution
from src.infrastructure.monitoring.metrics import record_task_submission
from src.infrastructure.monitoring.metrics import update_resource_utilization
from src.infrastructure.monitoring.metrics import update_task_queue_size
from src.shared.types.di_types import MetricsCollector
from src.unified_architecture.enhanced_platform import IUnifiedAgent
from src.unified_architecture.enhanced_platform import UnifiedTask


"""

from fastapi import status
from sqlalchemy import func
from typing import Any
from typing import Optional
Metrics Integration

This module integrates Prometheus metrics with the platform and agents
to provide comprehensive monitoring and observability.
"""

import logging
import time
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from src.infrastructure.monitoring.metrics import (
    MetricsCollector, record_agent_registration, record_task_execution,
    record_task_duration, record_agent_availability, record_task_submission,
    record_task_completion, record_error, update_resource_utilization,
    update_task_queue_size, AGENT_REGISTRATIONS, AGENT_TASK_EXECUTIONS,
    AGENT_TASK_DURATION, AGENT_AVAILABILITY, TASKS_SUBMITTED, TASKS_COMPLETED,
    TASK_QUEUE_SIZE, ERRORS_TOTAL, RESOURCE_UTILIZATION
)
from src.unified_architecture.core import IUnifiedAgent, UnifiedTask, TaskResult, AgentStatus


class MetricsIntegration:
    """
    Integrates metrics collection with the platform and agents
    """
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.metrics_collector = MetricsCollector()
        self._agent_metrics: Dict[str, Dict[str, Any]] = {}
        self._task_metrics: Dict[str, Dict[str, Any]] = {}
        self._platform_metrics: Dict[str, Any] = {}
    
    async def record_agent_activity(self, agent: IUnifiedAgent, activity_type: str, **kwargs) -> Any:
        """Record agent activity metrics"""
        try:
            agent_id = str(agent.agent_id)
            agent_metadata = await agent.get_metadata()
            agent_type = type(agent).__name__
            
            if activity_type == "registration":
                record_agent_registration(agent_type, "success")
                self._agent_metrics[agent_id] = {
                    "type": agent_type,
                    "name": agent_metadata.name,
                    "registered_at": datetime.utcnow(),
                    "activities": 0
                }
                
            elif activity_type == "status_change":
                status = await agent.get_status()
                record_agent_availability(agent_id, status.value)
                
                if agent_id in self._agent_metrics:
                    self._agent_metrics[agent_id]["last_status"] = status.value
                    self._agent_metrics[agent_id]["last_activity"] = datetime.utcnow()
                
            elif activity_type == "task_start":
                if agent_id in self._agent_metrics:
                    self._agent_metrics[agent_id]["activities"] += 1
                    self._agent_metrics[agent_id]["current_task"] = kwargs.get("task_id")
                
            elif activity_type == "task_complete":
                task_id = kwargs.get("task_id")
                task_type = kwargs.get("task_type", "unknown")
                status = kwargs.get("status", "unknown")
                duration = kwargs.get("duration", 0)
                
                record_task_execution(agent_id, task_type, status)
                record_task_duration(agent_id, task_type, duration)
                
                if agent_id in self._agent_metrics:
                    self._agent_metrics[agent_id]["completed_tasks"] = \
                        self._agent_metrics[agent_id].get("completed_tasks", 0) + 1
                    self._agent_metrics[agent_id]["total_execution_time"] = \
                        self._agent_metrics[agent_id].get("total_execution_time", 0) + duration
                
            elif activity_type == "error":
                error_type = kwargs.get("error_type", "unknown")
                record_error(error_type, agent_type, "error")
                
                if agent_id in self._agent_metrics:
                    self._agent_metrics[agent_id]["errors"] = \
                        self._agent_metrics[agent_id].get("errors", 0) + 1
            
            self.logger.debug("Recorded {} for agent {}", extra={"activity_type": activity_type, "agent_id": agent_id})
            
        except Exception as e:
            self.logger.error("Failed to record agent activity: {}", extra={"e": e})
    
    async def record_task_activity(self, task: UnifiedTask, activity_type: str, **kwargs) -> Any:
        """Record task activity metrics"""
        try:
            task_id = str(task.task_id)
            task_type = task.title.lower().replace(" ", "_")
            
            if activity_type == "submission":
                record_task_submission(task_type, task.priority)
                self._task_metrics[task_id] = {
                    "type": task_type,
                    "priority": task.priority,
                    "submitted_at": datetime.utcnow(),
                    "status": "submitted"
                }
                
            elif activity_type == "assignment":
                agent_id = kwargs.get("agent_id")
                if task_id in self._task_metrics:
                    self._task_metrics[task_id]["assigned_agent"] = agent_id
                    self._task_metrics[task_id]["assigned_at"] = datetime.utcnow()
                    self._task_metrics[task_id]["status"] = "assigned"
                
            elif activity_type == "completion":
                status = kwargs.get("status", "completed")
                duration = kwargs.get("duration", 0)
                
                record_task_completion(task_type, status)
                
                if task_id in self._task_metrics:
                    self._task_metrics[task_id]["completed_at"] = datetime.utcnow()
                    self._task_metrics[task_id]["status"] = status
                    self._task_metrics[task_id]["duration"] = duration
                
            elif activity_type == "failure":
                error = kwargs.get("error", "unknown")
                record_task_completion(task_type, "failed")
                record_error("task_failure", task_type, "error")
                
                if task_id in self._task_metrics:
                    self._task_metrics[task_id]["failed_at"] = datetime.utcnow()
                    self._task_metrics[task_id]["status"] = "failed"
                    self._task_metrics[task_id]["error"] = error
            
            self.logger.debug("Recorded {} for task {}", extra={"activity_type": activity_type, "task_id": task_id})
            
        except Exception as e:
            self.logger.error("Failed to record task activity: {}", extra={"e": e})
    
    async def record_platform_metrics(self, platform_stats: Dict[str, Any]) -> Any:
        """Record platform-level metrics"""
        try:
            # Update task queue sizes
            if "tasks" in platform_stats:
                tasks = platform_stats["tasks"]
                update_task_queue_size("pending", tasks.get("pending", 0))
                update_task_queue_size("in_progress", tasks.get("in_progress", 0))
                update_task_queue_size("completed", tasks.get("completed", 0))
                update_task_queue_size("failed", tasks.get("failed", 0))
            
            # Update resource utilization
            if "resources" in platform_stats:
                resources = platform_stats["resources"]
                total_resources = resources.get("total", 0)
                available_resources = resources.get("available", 0)
                
                if total_resources > 0:
                    utilization = ((total_resources - available_resources) / total_resources) * 100
                    update_resource_utilization("platform", "overall", utilization)
            
            # Store platform metrics
            self._platform_metrics.update(platform_stats)
            self._platform_metrics["last_updated"] = datetime.utcnow()
            
            self.logger.debug("Recorded platform metrics")
            
        except Exception as e:
            self.logger.error("Failed to record platform metrics: {}", extra={"e": e})
    
    @asynccontextmanager
    async def track_agent_task(self, agent: IUnifiedAgent, task: UnifiedTask) -> Any:
        """Context manager to track agent task execution"""
        start_time = time.time()
        task_id = str(task.task_id)
        
        try:
            # Record task start
            await self.record_agent_activity(agent, "task_start", task_id=task_id)
            await self.record_task_activity(task, "assignment", agent_id=str(agent.agent_id))
            
            yield
            
            # Record successful completion
            duration = time.time() - start_time
            await self.record_agent_activity(
                agent, "task_complete", 
                task_id=task_id, 
                task_type=task.title.lower().replace(" ", "_"),
                status="success",
                duration=duration
            )
            await self.record_task_activity(
                task, "completion", 
                status="completed",
                duration=duration
            )
            
        except Exception as e:
            # Record failure
            duration = time.time() - start_time
            await self.record_agent_activity(
                agent, "error",
                error_type=type(e).__name__
            )
            await self.record_task_activity(
                task, "failure",
                error=str(e)
            )
            raise
    
    def get_agent_metrics(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get agent metrics"""
        if agent_id:
            return self._agent_metrics.get(agent_id, {})
        return self._agent_metrics
    
    def get_task_metrics(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Get task metrics"""
        if task_id:
            return self._task_metrics.get(task_id, {})
        return self._task_metrics
    
    def get_platform_metrics(self) -> Dict[str, Any]:
        """Get platform metrics"""
        return self._platform_metrics.copy()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        summary = {
            "agents": {
                "total": len(self._agent_metrics),
                "by_type": {},
                "by_status": {}
            },
            "tasks": {
                "total": len(self._task_metrics),
                "by_status": {},
                "by_priority": {}
            },
            "platform": self._platform_metrics,
            "timestamp": datetime.utcnow()
        }
        
        # Aggregate agent metrics
        for agent_data in self._agent_metrics.values():
            agent_type = agent_data.get("type", "unknown")
            status = agent_data.get("last_status", "unknown")
            
            summary["agents"]["by_type"][agent_type] = \
                summary["agents"]["by_type"].get(agent_type, 0) + 1
            summary["agents"]["by_status"][status] = \
                summary["agents"]["by_status"].get(status, 0) + 1
        
        # Aggregate task metrics
        for task_data in self._task_metrics.values():
            status = task_data.get("status", "unknown")
            priority = task_data.get("priority", 0)
            
            summary["tasks"]["by_status"][status] = \
                summary["tasks"]["by_status"].get(status, 0) + 1
            summary["tasks"]["by_priority"][priority] = \
                summary["tasks"]["by_priority"].get(priority, 0) + 1
        
        return summary


# Global metrics integration instance
_metrics_integration: Optional[MetricsIntegration] = None


def get_metrics_integration() -> MetricsIntegration:
    """Get the global metrics integration instance"""
    global _metrics_integration
    if _metrics_integration is None:
        _metrics_integration = MetricsIntegration()
    return _metrics_integration


class MetricsMiddleware:
    """Middleware for automatic metrics collection"""
    
    def __init__(self, metrics_integration: MetricsIntegration) -> None:
        self.metrics_integration = metrics_integration
        self.logger = logging.getLogger(__name__)
    
    async def track_agent_operation(self, agent: IUnifiedAgent, operation: str, **kwargs) -> Any:
        """Track agent operations automatically"""
        try:
            if operation == "initialize":
                await self.metrics_integration.record_agent_activity(agent, "registration")
            elif operation == "status_change":
                await self.metrics_integration.record_agent_activity(agent, "status_change")
            elif operation == "task_execution":
                task = kwargs.get("task")
                if task:
                    async with self.metrics_integration.track_agent_task(agent, task):
                        yield
            elif operation == "error":
                await self.metrics_integration.record_agent_activity(
                    agent, "error", 
                    error_type=kwargs.get("error_type", "unknown")
                )
        except Exception as e:
            self.logger.error("Metrics middleware error: {}", extra={"e": e})
            raise
    
    async def track_platform_operation(self, operation: str, **kwargs) -> Any:
        """Track platform operations automatically"""
        try:
            if operation == "stats_update":
                platform_stats = kwargs.get("stats", {})
                await self.metrics_integration.record_platform_metrics(platform_stats)
            elif operation == "task_submission":
                task = kwargs.get("task")
                if task:
                    await self.metrics_integration.record_task_activity(task, "submission")
        except Exception as e:
            self.logger.error("Platform metrics middleware error: {}", extra={"e": e})
            raise


# Decorator for automatic metrics collection
def with_metrics(operation_type: str) -> Any:
    """Decorator to automatically collect metrics for operations"""
    def decorator(func) -> Any:
        async def wrapper(*args, **kwargs) -> Any:
            metrics = get_metrics_integration()
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record success metrics
                if operation_type == "agent_task":
                    agent = args[0] if args else None
                    task = kwargs.get("task")
                    if agent and task:
                        duration = time.time() - start_time
                        await metrics.record_agent_activity(
                            agent, "task_complete",
                            task_id=str(task.task_id),
                            task_type=task.title.lower().replace(" ", "_"),
                            status="success",
                            duration=duration
                        )
                
                return result
                
            except Exception as e:
                # Record error metrics
                if operation_type == "agent_task":
                    agent = args[0] if args else None
                    if agent:
                        await metrics.record_agent_activity(
                            agent, "error",
                            error_type=type(e).__name__
                        )
                raise
        
        return wrapper
    return decorator 