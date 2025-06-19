"""
Core Agent entity representing the AI agent domain model.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
from uuid import UUID, uuid4

from src.shared.types import AgentConfig, ModelConfig


class AgentType(str, Enum):
    """Types of agents available in the system."""
    FSM_REACT = "fsm_react"
    NEXT_GEN = "next_gen"
    CREW = "crew"
    SPECIALIZED = "specialized"


class AgentState(str, Enum):
    """Possible states of an agent."""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class Agent:
    """
    Core Agent entity representing an AI agent.
    
    This is a domain entity that encapsulates the business logic
    and rules for AI agents in the system.
    """
    
    # Identity
    id: UUID = field(default_factory=uuid4)
    name: str = field(default="AI Agent")
    agent_type: AgentType = field(default=AgentType.FSM_REACT)
    
    # State
    state: AgentState = field(default=AgentState.IDLE)
    current_task: Optional[str] = field(default=None)
    
    # Configuration
    config: AgentConfig = field(default_factory=AgentConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    
    # Performance tracking
    total_requests: int = field(default=0)
    successful_requests: int = field(default=0)
    failed_requests: int = field(default=0)
    average_response_time: float = field(default=0.0)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate agent after initialization."""
        if not self.name.strip():
            raise ValueError("Agent name cannot be empty")
        
        if self.total_requests < 0:
            raise ValueError("Total requests cannot be negative")
    
    def start_task(self, task_description: str) -> None:
        """Start a new task."""
        if self.state != AgentState.IDLE:
            raise ValueError(f"Cannot start task when agent is in {self.state} state")
        
        self.state = AgentState.THINKING
        self.current_task = task_description
        self.last_active = datetime.now()
    
    def complete_task(self, success: bool = True) -> None:
        """Complete the current task."""
        if self.state not in [AgentState.THINKING, AgentState.EXECUTING]:
            raise ValueError(f"Cannot complete task when agent is in {self.state} state")
        
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.state = AgentState.COMPLETED
        self.current_task = None
        self.last_active = datetime.now()
    
    def enter_error_state(self, error_message: str) -> None:
        """Enter error state."""
        self.state = AgentState.ERROR
        self.failed_requests += 1
        self.current_task = None
        self.last_active = datetime.now()
    
    def reset_to_idle(self) -> None:
        """Reset agent to idle state."""
        self.state = AgentState.IDLE
        self.current_task = None
        self.last_active = datetime.now()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def is_available(self) -> bool:
        """Check if agent is available for new tasks."""
        return self.state in [AgentState.IDLE, AgentState.COMPLETED]
    
    def update_response_time(self, response_time: float) -> None:
        """Update average response time."""
        if self.total_requests == 0:
            self.average_response_time = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_response_time = (
                alpha * response_time + 
                (1 - alpha) * self.average_response_time
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation."""
        return {
            "id": str(self.id),
            "name": self.name,
            "agent_type": self.agent_type.value,
            "state": self.state.value,
            "current_task": self.current_task,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "average_response_time": self.average_response_time,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "is_available": self.is_available
        } 