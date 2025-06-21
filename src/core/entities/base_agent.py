"""Base Agent Entity - Clean Implementation"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

@dataclass
class Agent(ABC):
    """Base agent class for all agent implementations"""
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Agent"
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @abstractmethod
    async def process(self, message: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Process a message and return response"""
        pass
        
    def __post_init__(self):
        """Initialize agent after creation"""
        if not self.agent_id:
            self.agent_id = str(uuid.uuid4())