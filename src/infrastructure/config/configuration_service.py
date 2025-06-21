from examples.advanced.multiagent_api_deployment import port
from fix_import_hierarchy import file_path
from fix_security_issues import content
from migrations.env import config
from migrations.env import url
from setup_environment import value

from src.core.monitoring import key
from src.infrastructure.config.configuration_service import agent_config
from src.infrastructure.config.configuration_service import api_config
from src.infrastructure.config.configuration_service import base_config
from src.infrastructure.config.configuration_service import base_file
from src.infrastructure.config.configuration_service import config_file
from src.infrastructure.config.configuration_service import d
from src.infrastructure.config.configuration_service import database_config
from src.infrastructure.config.configuration_service import db_config
from src.infrastructure.config.configuration_service import defaults
from src.infrastructure.config.configuration_service import env_config
from src.infrastructure.config.configuration_service import file_config
from src.infrastructure.config.configuration_service import logging_config
from src.infrastructure.config.configuration_service import model_config
from src.infrastructure.config.configuration_service import secret_config
from src.infrastructure.config.configuration_service import secrets
from src.infrastructure.config.configuration_service import secrets_file
from src.infrastructure.config.configuration_service import system_config
from src.infrastructure.config_cli import env
from src.services.integration_hub import parts
from src.utils.base_tool import ext
from src.utils.logging import debug
from src.utils.structured_logging import merged
from src.workflow.workflow_automation import errors

from src.agents.advanced_agent_fsm import Agent

from src.agents.advanced_agent_fsm import AgentConfig
# TODO: Fix undefined variables: Any, Awaitable, Callable, Dict, Environment, List, Optional, Path, api_config, api_key_required, base_config, base_file, config, config_dir, config_file, content, d, database_config, db_config, db_password, debug, defaults, dicts, e, env_config, errors, ext, f, fallback, file_config, file_path, groq_key, host, json, key, logging, logging_config, merged, model_config, openai_key, os, part, parts, port, primary, result, secret_config, secrets_file, service, system_config, tavily_key, temp, tokens, url, value, watcher, workers
from tests.test_complete_system import agent_config
import secrets

from src.infrastructure.config import AgentConfig
from src.infrastructure.config_cli import env


"""
from typing import Awaitable
from src.infrastructure.agents.concrete_agents import AgentConfig
from src.shared.types import DatabaseConfig
from src.shared.types import LogLevel
from src.shared.types import LoggingConfig
from src.shared.types import ModelConfig
from src.shared.types import SystemConfig
# TODO: Fix undefined variables: Awaitable, Environment, agent_config, aiofiles, api_config, api_key_required, base_config, base_file, config, config_dir, config_file, content, d, database_config, db_config, db_password, debug, defaults, dicts, e, env, env_config, errors, ext, f, fallback, file_config, file_path, groq_key, host, key, logging_config, merged, model_config, openai_key, part, parts, port, primary, result, secret_config, secrets, secrets_file, self, service, system_config, tavily_key, temp, tokens, url, value, watcher, workers, yaml

Enhanced ConfigurationService for the AI Agent system.

Usage Example:
    service = ConfigurationService()
    config = await service.load_configuration()
    logger.info("Value: %s", config.model_config.primary_model)
"""

from typing import Dict
from typing import Any
from typing import List
from typing import Callable

import os
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Awaitable

import aiofiles
import asyncio
import logging

from src.shared.types.config import (
    SystemConfig, AgentConfig, ModelConfig,
    LoggingConfig, DatabaseConfig, Environment,
    LogLevel
)
from src.shared.exceptions import ConfigurationException

logger = logging.getLogger(__name__)

class ConfigurationService:
    """
    Configuration service that loads from multiple sources with validation.
    Priority order (highest to lowest):
    1. Environment variables
    2. Secret management system
    3. Configuration files
    4. Default values
    """
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir or os.getenv("AI_AGENT_CONFIG_DIR", "config"))
        self._config: Optional[SystemConfig] = None
        self._watchers: List[Callable[[SystemConfig], Awaitable[None]]] = []
        self._lock = asyncio.Lock()

    async def load_configuration(self) -> SystemConfig:
        """Load configuration from all sources and merge."""
        async with self._lock:
            defaults = self._get_defaults()
            file_config = await self._load_from_file()
            env_config = self._load_from_env()
            secret_config = await self._load_from_secrets()
            merged = self._deep_merge(
                defaults,
                file_config,
                env_config,
                secret_config
            )
            self._config = self._create_config_objects(merged)
            self._validate_config(self._config)
            await self._notify_watchers(self._config)
            return self._config

    def _get_defaults(self) -> Dict[str, Any]:
        # ... (same as your provided defaults)
        return {
            "environment": Environment.DEVELOPMENT,
            "debug_mode": False,
            "api": {
                "host": "0.0.0.0",
                "port": 7860,
                "workers": 4,
                "cors_enabled": True,
                "allowed_origins": ["*"],
                "api_key_required": False,
                "compression_enabled": True,
                "max_request_size": 10485760,
                "request_timeout": 300
            },
            "model": {
                "primary_model": "llama-3.3-70b-versatile",
                "fallback_model": "llama-3.1-8b-instant",
                "temperature": 0.1,
                "max_tokens": 4096,
                "top_p": 0.95,
                "timeout_seconds": 30,
                "max_retries": 3,
                "retry_delay": 1.0,
                "requests_per_minute": 60,
                "burst_allowance": 10
            },
            "agent": {
                "max_steps": 15,
                "max_stagnation": 3,
                "max_retries": 3,
                "verification_level": "thorough",
                "confidence_threshold": 0.8,
                "cross_validation_sources": 2,
                "enable_caching": True,
                "cache_ttl_seconds": 3600,
                "max_cache_size": 1000,
                "enable_reflection": True,
                "enable_meta_cognition": True,
                "enable_tool_introspection": True,
                "enable_persistent_learning": True,
                "enable_input_validation": True,
                "enable_output_filtering": True,
                "max_input_length": 10000
            },
            "database": {
                "url": "",
                "api_key": "",
                "enable_logging": True,
                "log_table": "interactions",
                "max_connections": 10,
                "connection_timeout": 30
            },
            "logging": {
                "level": "INFO",
                "format": "[%(asctime)s] %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
                "date_format": "%Y-%m-%d %H:%M:%S",
                "enable_file_logging": True,
                "log_file": "agent.log",
                "max_file_size": 10485760,
                "backup_count": 5,
                "enable_json_logging": False,
                "include_correlation_id": True,
                "enable_async_logging": True,
                "log_queue_size": 1000
            },
            "monitoring": {
                "enable_health_checks": True,
                "enable_metrics": True,
                "metrics_port": 9090
            }
        }

    async def _load_from_file(self) -> Dict[str, Any]:
        config = {}
        env = os.getenv("AI_AGENT_ENV", "development").lower()
        for ext in [".yaml", ".yml", ".json"]:
            config_file = self.config_dir / f"{env}{ext}"
            if config_file.exists():
                config = await self._read_config_file(config_file)
                break
        for ext in [".yaml", ".yml", ".json"]:
            base_file = self.config_dir / f"base{ext}"
            if base_file.exists():
                base_config = await self._read_config_file(base_file)
                config = self._deep_merge(base_config, config)
                break
        return config

    async def _read_config_file(self, file_path: Path) -> Dict[str, Any]:
        try:
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
            if file_path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(content)
            elif file_path.suffix == '.json':
                return json.loads(content)
            else:
                raise ConfigurationException(f"Unsupported config file type: {file_path}")
        except Exception as e:
            logger.error("Failed to read config file {}: {}", extra={"file_path": file_path, "str_e_": str(e)})
            raise ConfigurationException(f"Failed to read config file {file_path}: {str(e)}")

    def _load_from_env(self) -> Dict[str, Any]:
        config = {}
        # Standard flat env vars
        if env := os.getenv("AI_AGENT_ENV"):
            config["environment"] = env.lower()
        if debug := os.getenv("AI_AGENT_DEBUG"):
            config["debug_mode"] = debug.lower() == "true"
        # API settings
        api_config = {}
        if host := os.getenv("AI_AGENT_API_HOST"):
            api_config["host"] = host
        if port := os.getenv("AI_AGENT_API_PORT"):
            api_config["port"] = int(port)
        if workers := os.getenv("AI_AGENT_API_WORKERS"):
            api_config["workers"] = int(workers)
        if api_key_required := os.getenv("AI_AGENT_API_KEY_REQUIRED"):
            api_config["api_key_required"] = api_key_required.lower() == "true"
        if api_config:
            config["api"] = api_config
        # Model settings
        model_config = {}
        if primary := os.getenv("AI_AGENT_PRIMARY_MODEL"):
            model_config["primary_model"] = primary
        if fallback := os.getenv("AI_AGENT_FALLBACK_MODEL"):
            model_config["fallback_model"] = fallback
        if temp := os.getenv("AI_AGENT_MODEL_TEMPERATURE"):
            model_config["temperature"] = float(temp)
        if tokens := os.getenv("AI_AGENT_MAX_TOKENS"):
            model_config["max_tokens"] = int(tokens)
        if model_config:
            config["model"] = model_config
        # Database settings
        db_config = {}
        if url := os.getenv("SUPABASE_URL"):
            db_config["url"] = url
        if key := os.getenv("SUPABASE_KEY"):
            db_config["api_key"] = key
        if db_password := os.getenv("SUPABASE_DB_PASSWORD"):
            db_config["password"] = db_password
        if db_config:
            config["database"] = db_config
        # API Keys
        if groq_key := os.getenv("GROQ_API_KEY"):
            config.setdefault("api_keys", {})["groq"] = groq_key
        if openai_key := os.getenv("OPENAI_API_KEY"):
            config.setdefault("api_keys", {})["openai"] = openai_key
        if tavily_key := os.getenv("TAVILY_API_KEY"):
            config.setdefault("api_keys", {})["tavily"] = tavily_key
        # Nested env var support (AI_AGENT_MODEL__TEMPERATURE)
        config = self._load_nested_env_vars(config)
        return config

    def _load_nested_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Support nested env vars like AI_AGENT_MODEL__TEMPERATURE"""
        for key, value in os.environ.items():
            if key.startswith("AI_AGENT_") and "__" in key:
                parts = key[len("AI_AGENT_"):].lower().split("__")
                d = config
                for part in parts[:-1]:
                    if part not in d or not isinstance(d[part], dict):
                        d[part] = {}
                    d = d[part]
                d[parts[-1]] = value
        return config

    async def _load_from_secrets(self) -> Dict[str, Any]:
        config = {}
        secrets_file = self.config_dir / ".secrets.json"
        if secrets_file.exists() and os.getenv("AI_AGENT_ENV") == "development":
            try:
                async with aiofiles.open(secrets_file, 'r') as f:
                    secrets = json.loads(await f.read())
                    config["api_keys"] = secrets.get("api_keys", {})
                    config["database"] = {**config.get("database", {}), **secrets.get("database", {})}
            except Exception as e:
                logger.warning("Failed to load secrets: {}", extra={"e": e})
        return config

    def _deep_merge(self, *dicts: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for d in dicts:
            for key, value in d.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._deep_merge(result[key], value)
                else:
                    result[key] = value
        return result

    def _create_config_objects(self, config: Dict[str, Any]) -> SystemConfig:
        # ... (same as your provided logic)
        model_config = ModelConfig(
            primary_model=config["model"]["primary_model"],
            fallback_model=config["model"]["fallback_model"],
            temperature=config["model"]["temperature"],
            max_tokens=config["model"]["max_tokens"],
            top_p=config["model"]["top_p"],
            timeout_seconds=config["model"]["timeout_seconds"],
            max_retries=config["model"]["max_retries"],
            retry_delay=config["model"]["retry_delay"],
            requests_per_minute=config["model"]["requests_per_minute"],
            burst_allowance=config["model"]["burst_allowance"]
        )
        agent_config = AgentConfig(
            max_steps=config["agent"]["max_steps"],
            max_stagnation=config["agent"]["max_stagnation"],
            max_retries=config["agent"]["max_retries"],
            verification_level=config["agent"]["verification_level"],
            confidence_threshold=config["agent"]["confidence_threshold"],
            cross_validation_sources=config["agent"]["cross_validation_sources"],
            enable_caching=config["agent"]["enable_caching"],
            cache_ttl_seconds=config["agent"]["cache_ttl_seconds"],
            max_cache_size=config["agent"]["max_cache_size"],
            enable_reflection=config["agent"]["enable_reflection"],
            enable_meta_cognition=config["agent"]["enable_meta_cognition"],
            enable_tool_introspection=config["agent"]["enable_tool_introspection"],
            enable_persistent_learning=config["agent"]["enable_persistent_learning"],
            enable_input_validation=config["agent"]["enable_input_validation"],
            enable_output_filtering=config["agent"]["enable_output_filtering"],
            max_input_length=config["agent"]["max_input_length"]
        )
        database_config = DatabaseConfig(
            url=config["database"]["url"],
            api_key=config["database"]["api_key"],
            enable_logging=config["database"]["enable_logging"],
            log_table=config["database"]["log_table"],
            max_connections=config["database"]["max_connections"],
            connection_timeout=config["database"]["connection_timeout"]
        )
        logging_config = LoggingConfig(
            level=LogLevel(config["logging"]["level"]),
            format=config["logging"]["format"],
            date_format=config["logging"]["date_format"],
            enable_file_logging=config["logging"]["enable_file_logging"],
            log_file=config["logging"]["log_file"],
            max_file_size=config["logging"]["max_file_size"],
            backup_count=config["logging"]["backup_count"],
            enable_json_logging=config["logging"]["enable_json_logging"],
            include_correlation_id=config["logging"]["include_correlation_id"],
            enable_async_logging=config["logging"]["enable_async_logging"],
            log_queue_size=config["logging"]["log_queue_size"]
        )
        system_config = SystemConfig(
            environment=Environment(config["environment"]),
            debug_mode=config["debug_mode"],
            api_host=config["api"]["host"],
            api_port=config["api"]["port"],
            api_workers=config["api"]["workers"],
            enable_cors=config["api"]["cors_enabled"],
            allowed_origins=config["api"]["allowed_origins"],
            api_key_required=config["api"]["api_key_required"],
            enable_compression=config["api"]["compression_enabled"],
            max_request_size=config["api"]["max_request_size"],
            request_timeout=config["api"]["request_timeout"],
            enable_health_checks=config["monitoring"]["enable_health_checks"],
            enable_metrics=config["monitoring"]["enable_metrics"],
            metrics_port=config["monitoring"]["metrics_port"]
        )
        system_config.model_config = model_config
        system_config.agent_config = agent_config
        system_config.database_config = database_config
        system_config.logging_config = logging_config
        system_config.api_keys = config.get("api_keys", {})
        return system_config

    def _validate_config(self, config: SystemConfig) -> None:
        errors = []
        if config.is_production():
            if not config.database_config.url:
                errors.append("Database URL is required in production")
            if not config.api_key_required:
                errors.append("API key authentication should be enabled in production")
            if config.debug_mode:
                errors.append("Debug mode should be disabled in production")
            if not hasattr(config, 'api_keys') or not config.api_keys.get('groq'):
                errors.append("GROQ API key is required")
        if not 1 <= config.api_port <= 65535:
            errors.append(f"Invalid API port: {config.api_port}")
        if config.enable_metrics and not 1 <= config.metrics_port <= 65535:
            errors.append(f"Invalid metrics port: {config.metrics_port}")
        if config.model_config.temperature < 0 or config.model_config.temperature > 2:
            errors.append(f"Invalid temperature: {config.model_config.temperature}")
        if errors:
            logger.error("Configuration validation failed: {}", extra={"_____join_errors_": ', '.join(errors)})
            raise ConfigurationException(f"Configuration validation failed: {', '.join(errors)}")

    async def _notify_watchers(self, config: SystemConfig) -> None:
        for watcher in self._watchers:
            try:
                await watcher(config)
            except Exception as e:
                logger.error("Configuration watcher failed: {}", extra={"str_e_": str(e)})

    def add_watcher(self, watcher: Callable[[SystemConfig], Awaitable[None]]) -> None:
        self._watchers.append(watcher)

    def get_config(self) -> SystemConfig:
        if not self._config:
            raise RuntimeError("Configuration not loaded")
        return self._config

    async def reload(self) -> SystemConfig:
        return await self.load_configuration()

    def get_api_key(self, service: str) -> Optional[str]:
        if not self._config:
            raise RuntimeError("Configuration not loaded")
        return getattr(self._config, 'api_keys', {}).get(service)