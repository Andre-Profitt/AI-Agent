"""
Resource Management for Multi-Agent System

This module provides resource allocation and monitoring:
- CPU, memory, GPU resource tracking
- Resource allocation and scheduling
- Resource optimization and load balancing
- Resource usage analytics
"""

import asyncio
import time
import psutil
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

try:
    import pynvml
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False
    pynvml = None

logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """Types of system resources"""
    CPU = auto()
    MEMORY = auto()
    GPU = auto()
    NETWORK = auto()
    STORAGE = auto()
    DISK_IO = auto()

@dataclass
class ResourceAllocation:
    """Resource allocation for an agent"""
    agent_id: str
    cpu_cores: float
    memory_mb: int
    gpu_memory_mb: Optional[int] = None
    network_bandwidth_mbps: Optional[float] = None
    storage_gb: Optional[float] = None
    priority: int = 5  # 1-10, higher is more important
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if allocation has expired"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_id": self.agent_id,
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "gpu_memory_mb": self.gpu_memory_mb,
            "network_bandwidth_mbps": self.network_bandwidth_mbps,
            "storage_gb": self.storage_gb,
            "priority": self.priority,
            "created_at": self.created_at,
            "expires_at": self.expires_at
        }

class ResourceManager:
    """Manages compute and memory resources for agents"""
    
    def __init__(self):
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.resource_limits = self._get_system_resources()
        self.resource_usage: Dict[str, Dict[ResourceType, float]] = defaultdict(dict)
        self.scheduler = ResourceScheduler()
        self.usage_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Configuration
        self.enable_gpu_monitoring = GPU_MONITORING_AVAILABLE
        self.monitoring_interval = 30  # seconds
        self.optimization_threshold = 0.8  # 80% utilization triggers optimization
        
        # Statistics
        self.stats = {
            "total_allocations": 0,
            "active_allocations": 0,
            "failed_allocations": 0,
            "optimizations_performed": 0
        }
        
    def _get_system_resources(self) -> Dict[ResourceType, float]:
        """Get available system resources"""
        resources = {
            ResourceType.CPU: psutil.cpu_count(),
            ResourceType.MEMORY: psutil.virtual_memory().total / (1024 * 1024),  # MB
            ResourceType.STORAGE: psutil.disk_usage('/').free / (1024 * 1024 * 1024),  # GB
            ResourceType.NETWORK: 1000.0,  # Mbps, simplified
            ResourceType.GPU: 0.0,  # Will be updated if GPU monitoring is available
            ResourceType.DISK_IO: 1000.0  # MB/s, simplified
        }
        
        # Get GPU information if available
        if self.enable_gpu_monitoring:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                total_gpu_memory = 0
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_gpu_memory += info.total
                
                resources[ResourceType.GPU] = total_gpu_memory / (1024 * 1024)  # Convert to MB
                logger.info("Detected {} GPU(s) with {}MB total memory", extra={"device_count": device_count, "resources_ResourceType_GPU": resources[ResourceType.GPU]})
            except Exception as e:
                logger.warning("Failed to initialize GPU monitoring: {}", extra={"e": e})
        
        return resources
    
    async def allocate_resources(self, agent_id: str, 
                               requirements: Dict[str, float]) -> bool:
        """Allocate resources to an agent"""
        try:
            # Create allocation
            allocation = ResourceAllocation(
                agent_id=agent_id,
                cpu_cores=requirements.get("cpu_cores", 1.0),
                memory_mb=requirements.get("memory_mb", 512),
                gpu_memory_mb=requirements.get("gpu_memory_mb"),
                network_bandwidth_mbps=requirements.get("network_bandwidth_mbps"),
                storage_gb=requirements.get("storage_gb"),
                priority=requirements.get("priority", 5)
            )
            
            # Check if resources can be allocated
            if self._can_allocate(allocation):
                self.allocations[agent_id] = allocation
                await self._apply_resource_limits(agent_id, allocation)
                
                self.stats["total_allocations"] += 1
                self.stats["active_allocations"] += 1
                
                logger.info("Allocated resources to agent {}: "
                           f"CPU={}, "
                           f"Memory={}MB", extra={"agent_id": agent_id, "allocation_cpu_cores": allocation.cpu_cores, "allocation_memory_mb": allocation.memory_mb})
                return True
            else:
                self.stats["failed_allocations"] += 1
                logger.warning("Insufficient resources for agent {}", extra={"agent_id": agent_id})
                return False
                
        except Exception as e:
            logger.error("Error allocating resources for agent {}: {}", extra={"agent_id": agent_id, "e": e})
            self.stats["failed_allocations"] += 1
            return False
    
    def _can_allocate(self, allocation: ResourceAllocation) -> bool:
        """Check if resources can be allocated"""
        # Calculate total allocated resources
        total_cpu = sum(a.cpu_cores for a in self.allocations.values() if not a.is_expired())
        total_memory = sum(a.memory_mb for a in self.allocations.values() if not a.is_expired())
        total_gpu = sum(a.gpu_memory_mb or 0 for a in self.allocations.values() if not a.is_expired())
        
        # Check if adding this allocation would exceed limits
        if total_cpu + allocation.cpu_cores > self.resource_limits[ResourceType.CPU]:
            logger.debug("CPU limit exceeded: {} > {}", extra={"total_cpu___allocation_cpu_cores": total_cpu + allocation.cpu_cores, "self_resource_limits_ResourceType_CPU": self.resource_limits[ResourceType.CPU]})
            return False
        
        if total_memory + allocation.memory_mb > self.resource_limits[ResourceType.MEMORY]:
            logger.debug("Memory limit exceeded: {} > {}", extra={"total_memory___allocation_memory_mb": total_memory + allocation.memory_mb, "self_resource_limits_ResourceType_MEMORY": self.resource_limits[ResourceType.MEMORY]})
            return False
        
        if allocation.gpu_memory_mb and total_gpu + allocation.gpu_memory_mb > self.resource_limits[ResourceType.GPU]:
            logger.debug("GPU memory limit exceeded: {} > {}", extra={"total_gpu___allocation_gpu_memory_mb": total_gpu + allocation.gpu_memory_mb, "self_resource_limits_ResourceType_GPU": self.resource_limits[ResourceType.GPU]})
            return False
        
        return True
    
    async def _apply_resource_limits(self, agent_id: str, 
                                   allocation: ResourceAllocation):
        """Apply resource limits to an agent process"""
        # This would integrate with container/process management
        # For now, just track the allocation
        self.resource_usage[agent_id] = {
            ResourceType.CPU: allocation.cpu_cores,
            ResourceType.MEMORY: allocation.memory_mb
        }
        
        if allocation.gpu_memory_mb:
            self.resource_usage[agent_id][ResourceType.GPU] = allocation.gpu_memory_mb
    
    async def release_resources(self, agent_id: str):
        """Release resources allocated to an agent"""
        if agent_id in self.allocations:
            del self.allocations[agent_id]
            self.stats["active_allocations"] -= 1
            
        if agent_id in self.resource_usage:
            del self.resource_usage[agent_id]
            
        logger.info("Released resources for agent {}", extra={"agent_id": agent_id})
    
    def get_resource_utilization(self) -> Dict[ResourceType, float]:
        """Get current resource utilization percentages"""
        utilization = {}
        
        # CPU utilization
        total_cpu = sum(a.cpu_cores for a in self.allocations.values() if not a.is_expired())
        utilization[ResourceType.CPU] = (total_cpu / self.resource_limits[ResourceType.CPU]) * 100
        
        # Memory utilization
        total_memory = sum(a.memory_mb for a in self.allocations.values() if not a.is_expired())
        utilization[ResourceType.MEMORY] = (total_memory / self.resource_limits[ResourceType.MEMORY]) * 100
        
        # GPU utilization
        if self.enable_gpu_monitoring:
            total_gpu = sum(a.gpu_memory_mb or 0 for a in self.allocations.values() if not a.is_expired())
            utilization[ResourceType.GPU] = (total_gpu / self.resource_limits[ResourceType.GPU]) * 100 if self.resource_limits[ResourceType.GPU] > 0 else 0
        
        return utilization
    
    async def get_agent_resource_usage(self, agent_id: str) -> Optional[Dict[str, float]]:
        """Get current resource usage for an agent"""
        if agent_id not in self.resource_usage:
            return None
        
        usage = self.resource_usage[agent_id]
        
        # Get actual system usage if possible
        try:
            # This would integrate with process monitoring
            # For now, return allocated resources
            return {
                "cpu_percent": usage.get(ResourceType.CPU, 0),
                "memory_mb": usage.get(ResourceType.MEMORY, 0),
                "gpu_memory_mb": usage.get(ResourceType.GPU, 0)
            }
        except Exception as e:
            logger.error("Error getting resource usage for agent {}: {}", extra={"agent_id": agent_id, "e": e})
            return None
    
    async def optimize_allocations(self) -> List[Dict[str, Any]]:
        """Optimize resource allocations based on usage patterns"""
        optimization_suggestions = []
        
        # Get current utilization
        utilization = self.get_resource_utilization()
        
        # Check for over-utilization
        for resource_type, util_percent in utilization.items():
            if util_percent > self.optimization_threshold * 100:
                logger.warning("High {} utilization: {}%", extra={"resource_type_name": resource_type.name, "util_percent": util_percent})
                
                # Find agents that could be optimized
                for agent_id, allocation in self.allocations.items():
                    if allocation.is_expired():
                        continue
                    
                    # Check if this agent is using this resource type
                    if resource_type == ResourceType.CPU and allocation.cpu_cores > 0.5:
                        optimization_suggestions.append({
                            "agent_id": agent_id,
                            "action": "reduce_cpu",
                            "current": allocation.cpu_cores,
                            "suggested": max(0.5, allocation.cpu_cores * 0.8),
                            "reason": f"High {resource_type.name} utilization"
                        })
                    
                    elif resource_type == ResourceType.MEMORY and allocation.memory_mb > 256:
                        optimization_suggestions.append({
                            "agent_id": agent_id,
                            "action": "reduce_memory",
                            "current": allocation.memory_mb,
                            "suggested": max(256, int(allocation.memory_mb * 0.8)),
                            "reason": f"High {resource_type.name} utilization"
                        })
        
        # Check for under-utilization
        for agent_id, allocation in self.allocations.items():
            if allocation.is_expired():
                continue
            
            # Get actual usage
            actual_usage = await self.get_agent_resource_usage(agent_id)
            if not actual_usage:
                continue
            
            # Check if agent is over-provisioned
            cpu_usage = actual_usage.get("cpu_percent", 0)
            if cpu_usage < allocation.cpu_cores * 0.5:  # Using less than 50%
                optimization_suggestions.append({
                    "agent_id": agent_id,
                    "action": "reduce_cpu",
                    "current": allocation.cpu_cores,
                    "suggested": max(0.5, cpu_usage * 1.2),  # 20% headroom
                    "reason": "Low CPU utilization"
                })
            
            memory_usage = actual_usage.get("memory_mb", 0)
            if memory_usage < allocation.memory_mb * 0.5:  # Using less than 50%
                optimization_suggestions.append({
                    "agent_id": agent_id,
                    "action": "reduce_memory",
                    "current": allocation.memory_mb,
                    "suggested": max(256, int(memory_usage * 1.2)),  # 20% headroom
                    "reason": "Low memory utilization"
                })
        
        return optimization_suggestions
    
    async def apply_optimization(self, agent_id: str, optimization: Dict[str, Any]) -> bool:
        """Apply an optimization suggestion"""
        try:
            if agent_id not in self.allocations:
                return False
            
            allocation = self.allocations[agent_id]
            action = optimization["action"]
            
            if action == "reduce_cpu":
                allocation.cpu_cores = optimization["suggested"]
                self.resource_usage[agent_id][ResourceType.CPU] = allocation.cpu_cores
                
            elif action == "reduce_memory":
                allocation.memory_mb = optimization["suggested"]
                self.resource_usage[agent_id][ResourceType.MEMORY] = allocation.memory_mb
            
            self.stats["optimizations_performed"] += 1
            
            logger.info("Applied optimization for agent {}: {}", extra={"agent_id": agent_id, "action": action})
            return True
            
        except Exception as e:
            logger.error("Error applying optimization for agent {}: {}", extra={"agent_id": agent_id, "e": e})
            return False
    
    def get_allocation_stats(self) -> Dict[str, Any]:
        """Get resource allocation statistics"""
        active_allocations = [a for a in self.allocations.values() if not a.is_expired()]
        
        return {
            **self.stats,
            "total_cpu_allocated": sum(a.cpu_cores for a in active_allocations),
            "total_memory_allocated": sum(a.memory_mb for a in active_allocations),
            "total_gpu_allocated": sum(a.gpu_memory_mb or 0 for a in active_allocations),
            "utilization": self.get_resource_utilization(),
            "resource_limits": {k.name: v for k, v in self.resource_limits.items()}
        }
    
    async def cleanup_expired_allocations(self):
        """Clean up expired resource allocations"""
        expired_agents = []
        
        for agent_id, allocation in self.allocations.items():
            if allocation.is_expired():
                expired_agents.append(agent_id)
        
        for agent_id in expired_agents:
            await self.release_resources(agent_id)
        
        if expired_agents:
            logger.info("Cleaned up {} expired resource allocations", extra={"len_expired_agents_": len(expired_agents)})

class ResourceScheduler:
    """Schedules tasks based on resource availability"""
    
    def __init__(self):
        self.task_queue: List[Tuple[float, Dict[str, Any]]] = []  # Priority queue
        self.running_tasks: Dict[str, Dict[str, Any]] = {}
        self.resource_availability: Dict[ResourceType, float] = {}
        
    def schedule_task(self, task: Dict[str, Any], estimated_resources: Dict[str, float]):
        """Schedule a task for execution"""
        priority = task.get("priority", 5)
        
        # Adjust priority based on deadline
        if "deadline" in task:
            time_until_deadline = task["deadline"] - time.time()
            if time_until_deadline < 300:  # Less than 5 minutes
                priority += 10  # Boost priority
        
        # Add to queue
        self.task_queue.append((-priority, task))  # Negative for max-heap behavior
        self.task_queue.sort(key=lambda x: x[0])  # Sort by priority
        
        logger.debug("Scheduled task {} with priority {}", extra={"task_get__id____unknown__": task.get('id', 'unknown'), "priority": priority})
    
    def get_next_task(self, available_resources: Dict[ResourceType, float]) -> Optional[Dict[str, Any]]:
        """Get the next task that can run with available resources"""
        if not self.task_queue:
            return None
        
        # Check each task in priority order
        for i, (priority, task) in enumerate(self.task_queue):
            if self._can_run_task(task, available_resources):
                # Remove from queue
                self.task_queue.pop(i)
                return task
        
        return None
    
    def _can_run_task(self, task: Dict[str, Any], 
                     available_resources: Dict[ResourceType, float]) -> bool:
        """Check if a task can run with available resources"""
        required_resources = task.get("required_resources", {})
        
        for resource_type, required_amount in required_resources.items():
            try:
                resource_enum = ResourceType[resource_type.upper()]
                available_amount = available_resources.get(resource_enum, 0)
                
                if required_amount > available_amount:
                    return False
            except KeyError:
                logger.warning("Unknown resource type: {}", extra={"resource_type": resource_type})
                return False
        
        return True
    
    def update_resource_availability(self, resources: Dict[ResourceType, float]):
        """Update available resources"""
        self.resource_availability = resources.copy()
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "queue_length": len(self.task_queue),
            "running_tasks": len(self.running_tasks),
            "highest_priority": max([p for p, _ in self.task_queue]) if self.task_queue else 0
        } 