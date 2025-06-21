# TODO: Fix undefined variables: issues, self
"""
Application settings with environment-based configuration
"""


import os
from typing import Optional
from dataclasses import dataclass

@dataclass
class LoggingConfig:
    """Logging configuration"""
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    JSON_LOGS: bool = True

@dataclass
class DatabaseConfig:
    """Database configuration"""
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    POOL_SIZE: int = 10
    MAX_RETRIES: int = 3

@dataclass
class APIConfig:
    """API configuration"""
    GROQ_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    PROMETHEUS_PORT: int = 8000
    HEALTH_CHECK_PORT: int = 8001
    METRICS_ENABLED: bool = True

class Settings:
    """Main application settings"""

    def __init__(self) -> None:
        # Environment detection
        self.environment = self._detect_environment()

        # Load configuration based on environment
        self.logging = LoggingConfig(
            LOG_LEVEL=os.getenv('LOG_LEVEL', 'INFO'),
            LOG_FILE=os.getenv('LOG_FILE'),
            JSON_LOGS=os.getenv('JSON_LOGS', 'true').lower() == 'true'
        )

        self.database = DatabaseConfig(
            SUPABASE_URL=os.getenv('SUPABASE_URL', ''),
            SUPABASE_KEY=os.getenv('SUPABASE_KEY', ''),
            POOL_SIZE=int(os.getenv('DB_POOL_SIZE', '10')),
            MAX_RETRIES=int(os.getenv('DB_MAX_RETRIES', '3'))
        )

        self.api = APIConfig(
            GROQ_API_KEY=os.getenv('GROQ_API_KEY', ''),
            OPENAI_API_KEY=os.getenv('OPENAI_API_KEY', ''),
            ANTHROPIC_API_KEY=os.getenv('ANTHROPIC_API_KEY', '')
        )

        self.monitoring = MonitoringConfig(
            PROMETHEUS_PORT=int(os.getenv('PROMETHEUS_PORT', '8000')),
            HEALTH_CHECK_PORT=int(os.getenv('HEALTH_CHECK_PORT', '8001')),
            METRICS_ENABLED=os.getenv('METRICS_ENABLED', 'true').lower() == 'true'
        )

        # Application settings
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        self.host = os.getenv('HOST', '0.0.0.0')
        self.port = int(os.getenv('PORT', '7860'))

    def _detect_environment(self) -> str:
        """Detect the current environment"""
        if os.getenv('HUGGINGFACE_SPACE_ID'):
            return 'huggingface'
        elif os.getenv('DOCKER_CONTAINER'):
            return 'docker'
        elif os.getenv('KUBERNETES_SERVICE_HOST'):
            return 'kubernetes'
        else:
            return 'development'

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment in ['huggingface', 'docker', 'kubernetes']

    @property
    def has_database(self) -> bool:
        """Check if database is configured"""
        return bool(self.database.SUPABASE_URL and self.database.SUPABASE_KEY)

    @property
    def has_api_keys(self) -> bool:
        """Check if API keys are configured"""
        return any([
            self.api.GROQ_API_KEY,
            self.api.OPENAI_API_KEY,
            self.api.ANTHROPIC_API_KEY
        ])

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues"""
        issues = []

        if not self.has_database:
            issues.append("Database not configured (SUPABASE_URL and SUPABASE_KEY required)")

        if not self.has_api_keys:
            issues.append("No API keys configured (GROQ_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY required)")

        return issues
