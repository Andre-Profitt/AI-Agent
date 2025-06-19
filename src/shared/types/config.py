"""
Configuration classes for the AI Agent system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class Environment(str, Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ModelConfig:
    """Configuration for AI models."""
    
    # Model selection
    primary_model: str = "llama-3.3-70b-versatile"
    fallback_model: str = "llama-3.1-8b-instant"
    
    # Model parameters
    temperature: float = 0.1
    max_tokens: int = 4096
    top_p: float = 0.95
    
    # Performance settings
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Rate limiting
    requests_per_minute: int = 60
    burst_allowance: int = 10
    
    def get_model_for_task(self, task_type: str) -> str:
        """Get appropriate model for task type."""
        model_mapping = {
            "reasoning": self.primary_model,
            "fast": self.fallback_model,
            "creative": "llama3-groq-70b-8192-creative",
            "verification": "llama3-groq-70b-8192-verification"
        }
        return model_mapping.get(task_type, self.primary_model)


@dataclass
class AgentConfig:
    """Configuration for AI agents."""
    
    # Agent behavior
    max_steps: int = 15
    max_stagnation: int = 3
    max_retries: int = 3
    
    # Quality settings
    verification_level: str = "thorough"
    confidence_threshold: float = 0.8
    cross_validation_sources: int = 2
    
    # Performance
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_cache_size: int = 1000
    
    # Features
    enable_reflection: bool = True
    enable_meta_cognition: bool = True
    enable_tool_introspection: bool = True
    enable_persistent_learning: bool = True
    
    # Safety
    enable_input_validation: bool = True
    enable_output_filtering: bool = True
    max_input_length: int = 10000


@dataclass
class DatabaseConfig:
    """Database configuration."""
    
    url: str = ""
    api_key: str = ""
    enable_logging: bool = True
    log_table: str = "interactions"
    max_connections: int = 10
    connection_timeout: int = 30


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    level: LogLevel = LogLevel.INFO
    format: str = "[%(asctime)s] %(levelname)-8s [%(name)s:%(lineno)d] %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # File logging
    enable_file_logging: bool = True
    log_file: str = "agent.log"
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    
    # Structured logging
    enable_json_logging: bool = False
    include_correlation_id: bool = True
    
    # Performance
    enable_async_logging: bool = True
    log_queue_size: int = 1000


@dataclass
class SystemConfig:
    """System-wide configuration."""
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug_mode: bool = False
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 7860
    api_workers: int = 4
    
    # Security
    enable_cors: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    api_key_required: bool = False
    
    # Performance
    enable_compression: bool = True
    max_request_size: int = 10485760  # 10MB
    request_timeout: int = 300
    
    # Monitoring
    enable_health_checks: bool = True
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT 