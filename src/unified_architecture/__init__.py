"""
Unified Architecture for Hybrid Agent System and Multi-Agent Collaboration Platform

This package provides a comprehensive framework for:
- Unified agent interfaces and capabilities
- Orchestration and coordination
- State management and communication
- Resource management and conflict resolution
- Performance tracking and collaboration metrics
- Agent marketplace and discovery
"""

from .core import (
    AgentCapability,
    AgentStatus,
    AgentMetadata,
    UnifiedTask,
    TaskResult,
    IUnifiedAgent
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
from .marketplace import AgentMarketplace, AgentListing
from .platform import MultiAgentPlatform

__version__ = "1.0.0"
__all__ = [
    # Core interfaces
    "AgentCapability",
    "AgentStatus", 
    "AgentMetadata",
    "UnifiedTask",
    "TaskResult",
    "IUnifiedAgent",
    
    # Orchestration
    "OrchestrationEngine",
    
    # State and communication
    "StateManager",
    "CommunicationProtocol",
    "AgentMessage",
    "MessageType",
    
    # Resource management
    "ResourceManager",
    "ResourceType",
    "ResourceAllocation",
    
    # Registry and discovery
    "AgentRegistry",
    
    # Task distribution
    "TaskDistributor",
    "TaskDistributionStrategy",
    
    # Shared memory
    "SharedMemorySystem",
    "MemoryType",
    "MemoryEntry",
    
    # Conflict resolution
    "ConflictResolver",
    "Conflict",
    "ConflictType",
    
    # Performance and monitoring
    "PerformanceTracker",
    "CollaborationDashboard",
    
    # Marketplace
    "AgentMarketplace",
    "AgentListing",
    
    # Platform
    "MultiAgentPlatform"
] 