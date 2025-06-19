"""
Core interfaces and data structures for the Unified Architecture

This module defines the fundamental building blocks:
- Agent capabilities and status
- Task and result representations  
- Unified agent interface
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union
from enum import Enum, auto
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# =============================
# ENUMS AND CONSTANTS
# =============================

class AgentCapability(Enum):
    """Standard agent capabilities"""
    REASONING = auto()
    TOOL_USE = auto()
    STATE_BASED = auto()
    MEMORY_ACCESS = auto()
    LEARNING = auto()
    COLLABORATION = auto()
    PLANNING = auto()
    EXECUTION = auto()
    VISION = auto()
    AUDIO = auto()
    TEXT_PROCESSING = auto()
    DATA_ANALYSIS = auto()
    WEB_SCRAPING = auto()
    FILE_PROCESSING = auto()
    MATHEMATICS = auto()
    CREATIVITY = auto()
    DECISION_MAKING = auto()
    OPTIMIZATION = auto()
    SIMULATION = auto()
    MONITORING = auto()

class AgentStatus(Enum):
    """Agent operational status"""
    IDLE = auto()
    BUSY = auto()
    AVAILABLE = auto()
    OFFLINE = auto()
    ERROR = auto()
    MAINTENANCE = auto()
    INITIALIZING = auto()
    SHUTTING_DOWN = auto()

# =============================
# DATA STRUCTURES
# =============================

@dataclass
class AgentMetadata:
    """Metadata for agent registration and discovery"""
    agent_id: str
    name: str
    version: str
    capabilities: List[AgentCapability]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    status: AgentStatus = AgentStatus.IDLE
    reliability_score: float = 1.0
    description: str = ""
    author: str = ""
    contact_info: Optional[str] = None
    documentation_url: Optional[str] = None
    license: str = "MIT"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "version": self.version,
            "capabilities": [cap.name for cap in self.capabilities],
            "performance_metrics": self.performance_metrics,
            "resource_requirements": self.resource_requirements,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "status": self.status.name,
            "reliability_score": self.reliability_score,
            "description": self.description,
            "author": self.author,
            "contact_info": self.contact_info,
            "documentation_url": self.documentation_url,
            "license": self.license
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMetadata':
        """Create from dictionary"""
        data = data.copy()
        data["capabilities"] = [AgentCapability[cap] for cap in data["capabilities"]]
        data["status"] = AgentStatus[data["status"]]
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["last_seen"] = datetime.fromisoformat(data["last_seen"])
        return cls(**data)

@dataclass
class UnifiedTask:
    """Unified task representation"""
    task_id: str
    task_type: str
    priority: int
    payload: Dict[str, Any]
    required_capabilities: List[AgentCapability]
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: Optional[float] = None  # seconds
    max_retries: int = 3
    timeout: Optional[float] = None  # seconds
    cost_limit: Optional[float] = None
    security_level: str = "standard"  # low, standard, high, critical
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate task after initialization"""
        if self.priority < 1 or self.priority > 10:
            raise ValueError("Priority must be between 1 and 10")
        
        if self.max_retries < 0:
            raise ValueError("Max retries must be non-negative")
        
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("Timeout must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "priority": self.priority,
            "payload": self.payload,
            "required_capabilities": [cap.name for cap in self.required_capabilities],
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
            "estimated_duration": self.estimated_duration,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "cost_limit": self.cost_limit,
            "security_level": self.security_level,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedTask':
        """Create from dictionary"""
        data = data.copy()
        data["required_capabilities"] = [AgentCapability[cap] for cap in data["required_capabilities"]]
        if data["deadline"]:
            data["deadline"] = datetime.fromisoformat(data["deadline"])
        return cls(**data)
    
    def is_urgent(self) -> bool:
        """Check if task is urgent based on deadline"""
        if not self.deadline:
            return False
        time_until_deadline = (self.deadline - datetime.utcnow()).total_seconds()
        return time_until_deadline < 300  # 5 minutes
    
    def get_priority_score(self) -> float:
        """Calculate priority score considering deadline"""
        score = self.priority
        
        # Boost priority for urgent tasks
        if self.is_urgent():
            score += 10
        
        # Consider security level
        security_boost = {
            "low": 0,
            "standard": 1,
            "high": 3,
            "critical": 5
        }
        score += security_boost.get(self.security_level, 0)
        
        return score

@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    success: bool
    result: Any
    execution_time: float
    agent_id: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    cost: Optional[float] = None
    quality_score: Optional[float] = None
    confidence: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "result": self.result,
            "execution_time": self.execution_time,
            "agent_id": self.agent_id,
            "error": self.error,
            "metadata": self.metadata,
            "retry_count": self.retry_count,
            "cost": self.cost,
            "quality_score": self.quality_score,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskResult':
        """Create from dictionary"""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)
    
    def is_high_quality(self) -> bool:
        """Check if result meets quality standards"""
        if not self.success:
            return False
        
        if self.quality_score is not None and self.quality_score < 0.8:
            return False
        
        if self.confidence is not None and self.confidence < 0.7:
            return False
        
        return True

# =============================
# UNIFIED AGENT INTERFACE
# =============================

class IUnifiedAgent(ABC):
    """Unified interface for all agents in the system"""
    
    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self._status = AgentStatus.INITIALIZING
        self._capabilities: List[AgentCapability] = []
        self._metadata: Optional[AgentMetadata] = None
        self._initialized = False
        self._shutdown_requested = False
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the agent with configuration"""
        pass
    
    @abstractmethod
    async def execute(self, task: UnifiedTask) -> TaskResult:
        """Execute a task and return result"""
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities"""
        pass
    
    @abstractmethod
    async def get_status(self) -> AgentStatus:
        """Return current agent status"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Gracefully shutdown the agent"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status"""
        pass
    
    async def collaborate(self, other_agent: 'IUnifiedAgent', 
                         task: UnifiedTask) -> TaskResult:
        """Collaborate with another agent on a task"""
        # Default implementation - delegate to other agent
        return await other_agent.execute(task)
    
    async def get_metadata(self) -> Optional[AgentMetadata]:
        """Get agent metadata"""
        return self._metadata
    
    async def update_metadata(self, metadata: AgentMetadata):
        """Update agent metadata"""
        self._metadata = metadata
    
    async def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        if self._metadata:
            return self._metadata.performance_metrics
        return {}
    
    async def update_performance_metrics(self, metrics: Dict[str, float]):
        """Update performance metrics"""
        if self._metadata:
            self._metadata.performance_metrics.update(metrics)
    
    async def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        # Default implementation - override in subclasses
        return {
            "cpu_percent": 0.0,
            "memory_mb": 0.0,
            "gpu_percent": 0.0
        }
    
    async def can_handle_task(self, task: UnifiedTask) -> bool:
        """Check if agent can handle a specific task"""
        agent_capabilities = await self.get_capabilities()
        return all(cap in agent_capabilities for cap in task.required_capabilities)
    
    async def estimate_task_duration(self, task: UnifiedTask) -> Optional[float]:
        """Estimate how long a task will take"""
        # Default implementation - override in subclasses
        return task.estimated_duration
    
    async def validate_task(self, task: UnifiedTask) -> Tuple[bool, Optional[str]]:
        """Validate if a task can be executed"""
        # Check capabilities
        if not await self.can_handle_task(task):
            return False, f"Agent lacks required capabilities: {task.required_capabilities}"
        
        # Check status
        status = await self.get_status()
        if status not in [AgentStatus.IDLE, AgentStatus.AVAILABLE]:
            return False, f"Agent is not available (status: {status.name})"
        
        # Check deadline
        if task.deadline and datetime.utcnow() > task.deadline:
            return False, "Task deadline has passed"
        
        return True, None
    
    def __str__(self) -> str:
        return f"{self.name} ({self.agent_id})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name} ({self.agent_id})>"

# =============================
# UTILITY FUNCTIONS
# =============================

def create_task_id() -> str:
    """Create a unique task ID"""
    return str(uuid.uuid4())

def create_agent_id(prefix: str = "agent") -> str:
    """Create a unique agent ID"""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def validate_capabilities(capabilities: List[AgentCapability]) -> bool:
    """Validate that capabilities are valid"""
    if not capabilities:
        return False
    
    valid_capabilities = set(AgentCapability)
    return all(cap in valid_capabilities for cap in capabilities)

def calculate_task_priority(priority: int, deadline: Optional[datetime] = None,
                          security_level: str = "standard") -> float:
    """Calculate effective task priority"""
    score = priority
    
    # Boost for urgent deadlines
    if deadline:
        time_until_deadline = (deadline - datetime.utcnow()).total_seconds()
        if time_until_deadline < 300:  # 5 minutes
            score += 10
        elif time_until_deadline < 1800:  # 30 minutes
            score += 5
    
    # Boost for security level
    security_boost = {
        "low": 0,
        "standard": 1,
        "high": 3,
        "critical": 5
    }
    score += security_boost.get(security_level, 0)
    
    return score 