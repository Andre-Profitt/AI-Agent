"""Agent configuration"""

from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class AgentConfig:
    """Agent configuration"""
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    max_iterations: int = 10
    timeout: float = 30.0
    
    # Memory settings
    enable_memory: bool = True
    memory_window_size: int = 100
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_interval: int = 60
    
    # Error handling
    error_threshold: int = 3
    recovery_timeout: float = 5.0
    retry_attempts: int = 3
    
    # Advanced features
    enable_reasoning: bool = True
    enable_learning: bool = False
    enable_multimodal: bool = False
    
    # Performance
    max_concurrent_requests: int = 10
    cache_ttl: int = 300