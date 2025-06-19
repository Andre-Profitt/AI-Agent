"""
ConfigurationService implementation for the AI Agent system.
"""

import os
from typing import Optional
from src.shared.types.config import SystemConfig, AgentConfig, ModelConfig, LoggingConfig, DatabaseConfig, Environment

class ConfigurationService:
    def __init__(self):
        self._config: Optional[SystemConfig] = None

    async def load_configuration(self) -> SystemConfig:
        # For now, just use defaults; in production, load from env/files
        env = os.getenv("AI_AGENT_ENV", "development").lower()
        environment = Environment(env) if env in Environment.__members__.values() else Environment.DEVELOPMENT
        self._config = SystemConfig(
            environment=environment,
            debug_mode=os.getenv("AI_AGENT_DEBUG", "false").lower() == "true"
        )
        return self._config

    def get_config(self) -> SystemConfig:
        if not self._config:
            raise RuntimeError("Configuration not loaded")
        return self._config 