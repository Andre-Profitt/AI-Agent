"""Base Message Entity - Clean Implementation"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Message:
    """Message class for agent communication"""
    content: str
    role: str  # user, assistant, system
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)