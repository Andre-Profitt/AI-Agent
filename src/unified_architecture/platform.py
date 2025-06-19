"""
Multi-Agent Platform for Unified Architecture

This module provides the main platform that integrates all unified architecture
components into a comprehensive multi-agent collaboration system.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from dataclasses import dataclass, field
from uuid import UUID, uuid4
from contextlib import asynccontextmanager

from .core import (
    AgentCapability, AgentStatus, AgentMetadata, 
    UnifiedTask, TaskResult, IUnifiedAgent
)
from .orchestration import OrchestrationEngine
from .state_management import StateManager
from .communication import CommunicationProtocol, AgentMessage, MessageType
from .resource_management import ResourceManager, ResourceType, ResourceAllocation
from .registry import AgentRegistry
from .task_distribution import TaskDistributor, TaskDistributionStrategy
from .shared_memory import SharedMemorySystem, MemoryType, MemoryEntry
from .conflict_resolution import ConflictResolver, Conflict, ConflictType
from .performance import PerformanceTracker
from .dashboard import CollaborationDashboard
from .marketplace import AgentMarketplace, AgentListing, ListingStatus


@dataclass
class PlatformConfig:
    """Configuration for the multi-agent platform"""
    max_concurrent_tasks: int = 100
    task_timeout: int = 300  # seconds
    heartbeat_interval: int = 30  # seconds
    cleanup_interval: int = 3600  # seconds
    enable_marketplace: bool = True
    enable_dashboard: bool = True
    enable_performance_tracking: bool = True
    enable_conflict_resolution: bool = True
    storage_backend: str = "memory"  # memory, redis, database
    log_level: str = "INFO"


@dataclass
class PlatformStats:
    """Platform statistics"""
    total_agents: int = 0
    active_agents: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    active_tasks: int = 0
    total_collaborations: int = 0
    platform_uptime: float = 0.0
    last_activity: datetime = field(default_factory=datetime.utcnow)
    resource_utilization: Dict[str, float] = field(default_factory=dict)


class MultiAgentPlatform:
    """
    Main platform that integrates all unified architecture components
    """
    
    def __init__(self, config: Optional[PlatformConfig] = None):
        self.config = config or PlatformConfig()
        self.logger = logging.getLogger(__name__)
        self.platform_id = uuid4()
        self.start_time = datetime.utcnow()
        
        # Initialize core components
        self._initialize_components()
        
        # Platform state
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._background_tasks: Set[asyncio.Task] = set()
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        # Statistics
        self._stats = PlatformStats()
        
        self.logger.info(f"MultiAgentPlatform {self.platform_id} initialized")
    
    def _initialize_components(self):
        """Initialize all platform components"""
        try:
            # Core orchestration
            self.orchestration = OrchestrationEngine()
            
            # State management
            self.state_manager = StateManager(
                storage_backend=self.config.storage_backend
            )
            
            # Communication
            self.communication = CommunicationProtocol()
            
            # Resource management
            self.resource_manager = ResourceManager()
            
            # Registry
            self.registry = AgentRegistry()
            
            # Task distribution
            self.task_distributor = TaskDistributor()
            
            # Shared memory
            self.shared_memory = SharedMemorySystem()
            
            # Performance tracking
            if self.config.enable_performance_tracking:
                self.performance_tracker = PerformanceTracker()
            
            # Conflict resolution
            if self.config.enable_conflict_resolution:
                self.conflict_resolver = ConflictResolver()
            
            # Dashboard
            if self.config.enable_dashboard:
                self.dashboard = CollaborationDashboard()
            
            # Marketplace
            if self.config.enable_marketplace:
                self.marketplace = AgentMarketplace(
                    storage_backend=self.config.storage_backend
                )
            
            self.logger.info("All platform components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize platform components: {e}")
            raise
    
    async def start(self):
        """Start the platform"""
        try:
            if self._running:
                self.logger.warning("Platform is already running")
                return
            
            self.logger.info("Starting MultiAgentPlatform...")
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Initialize components
            await self._initialize_platform_components()
            
            # Start event loop
            self._running = True
            self.start_time = datetime.utcnow()
            
            self.logger.info("MultiAgentPlatform started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start platform: {e}")
            raise
    
    async def stop(self):
        """Stop the platform"""
        try:
            if not self._running:
                self.logger.warning("Platform is not running")
                return
            
            self.logger.info("Stopping MultiAgentPlatform...")
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Stop components
            await self._stop_platform_components()
            
            self._running = False
            self.logger.info("MultiAgentPlatform stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to stop platform: {e}")
            raise
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        try:
            # Heartbeat task
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._background_tasks.add(heartbeat_task)
            
            # Cleanup task
            cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._background_tasks.add(cleanup_task)
            
            # Statistics update task
            stats_task = asyncio.create_task(self._stats_update_loop())
            self._background_tasks.add(stats_task)
            
            self.logger.info("Background tasks started")
            
        except Exception as e:
            self.logger.error(f"Failed to start background tasks: {e}")
            raise
    
    async def _initialize_platform_components(self):
        """Initialize platform-specific components"""
        try:
            # Initialize communication
            await self.communication.initialize()
            
            # Initialize resource monitoring
            await self.resource_manager.start_monitoring()
            
            # Initialize performance tracking
            if self.config.enable_performance_tracking:
                await self.performance_tracker.initialize()
            
            # Initialize dashboard
            if self.config.enable_dashboard:
                await self.dashboard.initialize()
            
            self.logger.info("Platform components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize platform components: {e}")
            raise
    
    async def _stop_platform_components(self):
        """Stop platform components"""
        try:
            # Stop resource monitoring
            await self.resource_manager.stop_monitoring()
            
            # Stop performance tracking
            if self.config.enable_performance_tracking:
                await self.performance_tracker.shutdown()
            
            # Stop dashboard
            if self.config.enable_dashboard:
                await self.dashboard.shutdown()
            
            self.logger.info("Platform components stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop platform components: {e}")
    
    async def register_agent(self, agent: IUnifiedAgent, metadata: AgentMetadata) -> bool:
        """Register an agent with the platform"""
        try:
            # Register with orchestration engine
            success = await self.orchestration.register_agent(agent, metadata)
            if not success:
                return False
            
            # Register with registry
            await self.registry.register_agent(metadata)
            
            # Register with performance tracker
            if self.config.enable_performance_tracking:
                await self.performance_tracker.register_agent(metadata.agent_id)
            
            # Update statistics
            self._stats.total_agents += 1
            self._stats.active_agents += 1
            
            # Emit event
            await self._emit_event("agent_registered", {
                "agent_id": str(metadata.agent_id),
                "capabilities": [cap.value for cap in metadata.capabilities]
            })
            
            self.logger.info(f"Agent {metadata.agent_id} registered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {metadata.agent_id}: {e}")
            return False
    
    async def unregister_agent(self, agent_id: UUID) -> bool:
        """Unregister an agent from the platform"""
        try:
            # Unregister from orchestration
            await self.orchestration.unregister_agent(agent_id)
            
            # Unregister from registry
            await self.registry.unregister_agent(agent_id)
            
            # Unregister from performance tracker
            if self.config.enable_performance_tracking:
                await self.performance_tracker.unregister_agent(agent_id)
            
            # Update statistics
            self._stats.active_agents = max(0, self._stats.active_agents - 1)
            
            # Emit event
            await self._emit_event("agent_unregistered", {
                "agent_id": str(agent_id)
            })
            
            self.logger.info(f"Agent {agent_id} unregistered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    async def submit_task(self, task: UnifiedTask) -> UUID:
        """Submit a task to the platform"""
        try:
            # Validate task
            if not task.description or not task.requirements:
                raise ValueError("Task must have description and requirements")
            
            # Check resource availability
            if not await self.resource_manager.check_availability(task.requirements):
                raise ValueError("Insufficient resources for task")
            
            # Submit to orchestration engine
            task_id = await self.orchestration.submit_task(task)
            
            # Update statistics
            self._stats.total_tasks += 1
            self._stats.active_tasks += 1
            
            # Emit event
            await self._emit_event("task_submitted", {
                "task_id": str(task_id),
                "priority": task.priority.value,
                "requirements": task.requirements
            })
            
            self.logger.info(f"Task {task_id} submitted successfully")
            return task_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit task: {e}")
            raise
    
    async def get_task_status(self, task_id: UUID) -> Optional[Dict[str, Any]]:
        """Get the status of a task"""
        try:
            return await self.orchestration.get_task_status(task_id)
        except Exception as e:
            self.logger.error(f"Failed to get task status {task_id}: {e}")
            return None
    
    async def cancel_task(self, task_id: UUID) -> bool:
        """Cancel a running task"""
        try:
            success = await self.orchestration.cancel_task(task_id)
            if success:
                self._stats.active_tasks = max(0, self._stats.active_tasks - 1)
                
                # Emit event
                await self._emit_event("task_cancelled", {
                    "task_id": str(task_id)
                })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to cancel task {task_id}: {e}")
            return False
    
    async def get_platform_stats(self) -> PlatformStats:
        """Get platform statistics"""
        try:
            # Update uptime
            self._stats.platform_uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            # Get resource utilization
            self._stats.resource_utilization = await self.resource_manager.get_utilization()
            
            # Get task statistics from orchestration
            task_stats = await self.orchestration.get_task_statistics()
            self._stats.completed_tasks = task_stats.get("completed", 0)
            self._stats.failed_tasks = task_stats.get("failed", 0)
            self._stats.active_tasks = task_stats.get("active", 0)
            
            return self._stats
            
        except Exception as e:
            self.logger.error(f"Failed to get platform stats: {e}")
            return self._stats
    
    async def get_agent_performance(self, agent_id: UUID) -> Optional[Dict[str, Any]]:
        """Get performance metrics for an agent"""
        try:
            if not self.config.enable_performance_tracking:
                return None
            
            return await self.performance_tracker.get_agent_metrics(agent_id)
            
        except Exception as e:
            self.logger.error(f"Failed to get agent performance {agent_id}: {e}")
            return None
    
    async def get_collaboration_network(self) -> Dict[str, Any]:
        """Get the collaboration network graph"""
        try:
            if not self.config.enable_dashboard:
                return {}
            
            return await self.dashboard.get_collaboration_network()
            
        except Exception as e:
            self.logger.error(f"Failed to get collaboration network: {e}")
            return {}
    
    async def resolve_conflict(self, conflict: Conflict) -> Optional[Dict[str, Any]]:
        """Resolve a conflict using the conflict resolution system"""
        try:
            if not self.config.enable_conflict_resolution:
                return None
            
            return await self.conflict_resolver.resolve_conflict(conflict)
            
        except Exception as e:
            self.logger.error(f"Failed to resolve conflict: {e}")
            return None
    
    async def share_memory(self, entry: MemoryEntry) -> bool:
        """Share a memory entry with the platform"""
        try:
            success = await self.shared_memory.store_memory(entry)
            if success:
                # Emit event
                await self._emit_event("memory_shared", {
                    "memory_id": str(entry.id),
                    "type": entry.type.value,
                    "source_agent": str(entry.source_agent)
                })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to share memory: {e}")
            return False
    
    async def search_memory(self, query: str, memory_type: Optional[MemoryType] = None) -> List[MemoryEntry]:
        """Search shared memory"""
        try:
            return await self.shared_memory.search_memory(query, memory_type)
        except Exception as e:
            self.logger.error(f"Failed to search memory: {e}")
            return []
    
    async def send_message(self, message: AgentMessage) -> bool:
        """Send a message through the communication protocol"""
        try:
            success = await self.communication.send_message(message)
            if success:
                # Emit event
                await self._emit_event("message_sent", {
                    "message_id": str(message.id),
                    "from_agent": str(message.from_agent),
                    "to_agent": str(message.to_agent),
                    "type": message.type.value
                })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}")
            return False
    
    async def broadcast_message(self, message: AgentMessage) -> bool:
        """Broadcast a message to all agents"""
        try:
            success = await self.communication.broadcast_message(message)
            if success:
                # Emit event
                await self._emit_event("message_broadcast", {
                    "message_id": str(message.id),
                    "from_agent": str(message.from_agent),
                    "type": message.type.value
                })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to broadcast message: {e}")
            return False
    
    async def get_available_agents(self, capabilities: Optional[List[AgentCapability]] = None) -> List[AgentMetadata]:
        """Get available agents with optional capability filtering"""
        try:
            return await self.registry.get_agents(capabilities)
        except Exception as e:
            self.logger.error(f"Failed to get available agents: {e}")
            return []
    
    async def allocate_resources(self, requirements: Dict[ResourceType, float]) -> Optional[ResourceAllocation]:
        """Allocate resources for a task"""
        try:
            return await self.resource_manager.allocate_resources(requirements)
        except Exception as e:
            self.logger.error(f"Failed to allocate resources: {e}")
            return None
    
    async def release_resources(self, allocation_id: UUID) -> bool:
        """Release allocated resources"""
        try:
            return await self.resource_manager.release_resources(allocation_id)
        except Exception as e:
            self.logger.error(f"Failed to release resources: {e}")
            return False
    
    async def _heartbeat_loop(self):
        """Background heartbeat loop"""
        while not self._shutdown_event.is_set():
            try:
                # Update last activity
                self._stats.last_activity = datetime.utcnow()
                
                # Check agent health
                await self.registry.check_agent_health()
                
                # Update performance metrics
                if self.config.enable_performance_tracking:
                    await self.performance_tracker.update_metrics()
                
                # Wait for next heartbeat
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while not self._shutdown_event.is_set():
            try:
                # Clean up old tasks
                await self.orchestration.cleanup_completed_tasks()
                
                # Clean up old memory entries
                await self.shared_memory.cleanup_old_entries()
                
                # Clean up old performance data
                if self.config.enable_performance_tracking:
                    await self.performance_tracker.cleanup_old_data()
                
                # Wait for next cleanup
                await asyncio.sleep(self.config.cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)  # Brief pause on error
    
    async def _stats_update_loop(self):
        """Background statistics update loop"""
        while not self._shutdown_event.is_set():
            try:
                # Update platform statistics
                await self.get_platform_stats()
                
                # Update dashboard
                if self.config.enable_dashboard:
                    await self.dashboard.update_metrics(self._stats)
                
                # Wait for next update
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Stats update loop error: {e}")
                await asyncio.sleep(10)  # Brief pause on error
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to registered handlers"""
        try:
            if event_type in self._event_handlers:
                for handler in self._event_handlers[event_type]:
                    try:
                        await handler(data)
                    except Exception as e:
                        self.logger.error(f"Event handler error: {e}")
                        
        except Exception as e:
            self.logger.error(f"Failed to emit event {event_type}: {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add an event handler"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: str, handler: Callable):
        """Remove an event handler"""
        if event_type in self._event_handlers:
            self._event_handlers[event_type].remove(handler)
    
    @asynccontextmanager
    async def platform_context(self):
        """Context manager for platform lifecycle"""
        try:
            await self.start()
            yield self
        finally:
            await self.stop()
    
    def is_running(self) -> bool:
        """Check if the platform is running"""
        return self._running
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check"""
        try:
            health = {
                "platform_status": "healthy" if self._running else "stopped",
                "components": {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check each component
            components = [
                ("orchestration", self.orchestration),
                ("state_manager", self.state_manager),
                ("communication", self.communication),
                ("resource_manager", self.resource_manager),
                ("registry", self.registry),
                ("task_distributor", self.task_distributor),
                ("shared_memory", self.shared_memory)
            ]
            
            for name, component in components:
                try:
                    if hasattr(component, 'health_check'):
                        health["components"][name] = await component.health_check()
                    else:
                        health["components"][name] = {"status": "unknown"}
                except Exception as e:
                    health["components"][name] = {"status": "error", "error": str(e)}
            
            # Check optional components
            if self.config.enable_performance_tracking:
                try:
                    health["components"]["performance_tracker"] = await self.performance_tracker.health_check()
                except Exception as e:
                    health["components"]["performance_tracker"] = {"status": "error", "error": str(e)}
            
            if self.config.enable_conflict_resolution:
                try:
                    health["components"]["conflict_resolver"] = await self.conflict_resolver.health_check()
                except Exception as e:
                    health["components"]["conflict_resolver"] = {"status": "error", "error": str(e)}
            
            if self.config.enable_dashboard:
                try:
                    health["components"]["dashboard"] = await self.dashboard.health_check()
                except Exception as e:
                    health["components"]["dashboard"] = {"status": "error", "error": str(e)}
            
            if self.config.enable_marketplace:
                try:
                    health["components"]["marketplace"] = {"status": "healthy"}  # Simple check
                except Exception as e:
                    health["components"]["marketplace"] = {"status": "error", "error": str(e)}
            
            return health
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "platform_status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            } 