"""
Shared type definitions for the AI Agent system.

This module contains common type definitions, configurations, and data structures
that are used across different layers of the application.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Union, Tuple


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ModelConfig:
    """Configuration for AI models"""
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 4000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3


@dataclass
class AgentConfig:
    """Configuration for AI agents"""
    agent_type: str = "fsm_react"
    name: str = "AI Agent"
    description: Optional[str] = None
    max_concurrent_tasks: int = 5
    task_timeout: int = 300
    enable_learning: bool = True
    enable_collaboration: bool = True
    memory_size: int = 1000
    log_level: LogLevel = LogLevel.INFO
    model_config: ModelConfig = field(default_factory=ModelConfig)
    tools: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = False
    enable_structured_logging: bool = True


@dataclass
class DatabaseConfig:
    """Configuration for database connections"""
    database_type: str = "sqlite"  # sqlite, postgresql, mysql
    host: Optional[str] = None
    port: Optional[int] = None
    database_name: str = "ai_agent.db"
    username: Optional[str] = None
    password: Optional[str] = None
    connection_string: Optional[str] = None
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    ssl_mode: Optional[str] = None


@dataclass
class SystemConfig:
    """Main system configuration"""
    environment: str = "development"  # development, staging, production
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=list)
    secret_key: Optional[str] = None
    session_timeout: int = 3600  # 1 hour
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    
    # Component configurations
    agent_config: AgentConfig = field(default_factory=AgentConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    database_config: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # Feature flags
    enable_monitoring: bool = True
    enable_metrics: bool = True
    enable_health_checks: bool = True
    enable_rate_limiting: bool = False
    enable_caching: bool = True
    
    # Performance settings
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    cache_ttl: int = 300  # 5 minutes
    
    # Security settings
    enable_authentication: bool = False
    enable_authorization: bool = False
    jwt_secret: Optional[str] = None
    jwt_expiration: int = 3600  # 1 hour
    
    # External service configurations
    external_apis: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self) -> Any:
        """Validate configuration after initialization"""
        if self.environment not in ["development", "staging", "production"]:
            raise ValueError("Environment must be one of: development, staging, production")
        
        if self.port < 1 or self.port > 65535:
            raise ValueError("Port must be between 1 and 65535")
        
        if self.workers < 1:
            raise ValueError("Workers must be at least 1")
        
        if self.session_timeout < 0:
            raise ValueError("Session timeout must be non-negative")
        
        if self.max_request_size < 0:
            raise ValueError("Max request size must be non-negative")


@dataclass
class TaskConfig:
    """Configuration for task execution"""
    task_type: str
    priority: int = 5
    timeout: int = 300
    retry_attempts: int = 3
    retry_delay: int = 5
    max_concurrent_executions: int = 1
    requires_approval: bool = False
    notification_on_completion: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolConfig:
    """Configuration for tools"""
    tool_name: str
    tool_type: str
    enabled: bool = True
    timeout: int = 30
    retry_attempts: int = 3
    rate_limit: Optional[int] = None
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for agents and tasks"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    total_execution_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests


@dataclass
class HealthStatus:
    """Health status information"""
    healthy: bool = True
    status: str = "healthy"
    message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    checks: Dict[str, bool] = field(default_factory=dict)
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    details: Dict[str, Any] = field(default_factory=dict)


# Type aliases for common use cases
ConfigDict = Dict[str, Any]
MetadataDict = Dict[str, Any]
ResultDict = Dict[str, Any]
ErrorDict = Dict[str, Any] 