from setup_environment import value

from src.config.integrations import api_key
from src.config.integrations import enabled
from src.config.integrations import service_map
from src.config.integrations import validation_results
from src.core.langgraph_resilience_patterns import circuit_breaker
from src.core.monitoring import key
from src.utils.structured_logging import get_structured_logger

from src.tools.base_tool import Tool
from dataclasses import dataclass
from src.services.circuit_breaker import CircuitBreakerConfig
# TODO: Fix undefined variables: Any, Dict, List, Optional, api_key, dataclass, default, e, enabled, key, os, prefix, service, service_map, validation_results, value
from src.infrastructure.resilience.circuit_breaker import circuit_breaker

# TODO: Fix undefined variables: api_key, circuit_breaker, default, e, enabled, get_structured_logger, key, prefix, self, service, service_map, validation_results, value

"""
Integration configuration for external services with full circuit breaker protection
"""

from typing import Optional
from typing import Any
from typing import List

import os
import asyncio
import logging
from typing import Dict, Any, Optional, List

from src.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    circuit_breaker,
    CircuitBreakerOpenError
)
from src.utils.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

@dataclass
class SupabaseConfig:
    """Supabase configuration with validation"""
    url: str = ""
    key: str = ""
    service_key: str = ""
    db_password: str = ""
    
    def is_valid(self) -> bool:
        """Check if configuration is valid"""
        return bool(
            self.url and 
            self.key and 
            (self.url.startswith("http://") or self.url.startswith("https://"))
        )
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for connection"""
        return {
            "url": self.url,
            "key": self.key,
            "service_key": self.service_key,
            "db_password": self.db_password
        }

@dataclass
class APIConfig:
    """API configuration for various services"""
    groq_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    elevenlabs_api_key: str = ""
    
    def is_valid(self, service: str) -> bool:
        """Check if specific service is configured"""
        service_map = {
            "groq": self.groq_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "elevenlabs": self.elevenlabs_api_key
        }
        return bool(service_map.get(service, ""))

@dataclass
class ToolConfig:
    """Tool configuration flags"""
    enable_file_tools: bool = True
    enable_web_tools: bool = True
    enable_code_tools: bool = True
    enable_media_tools: bool = True
    enable_communication_tools: bool = True
    
    def get_enabled_tools(self) -> List[str]:
        """Get list of enabled tools"""
        enabled = []
        if self.enable_file_tools:
            enabled.append("file")
        if self.enable_web_tools:
            enabled.append("web")
        if self.enable_code_tools:
            enabled.append("code")
        if self.enable_media_tools:
            enabled.append("media")
        if self.enable_communication_tools:
            enabled.append("communication")
        return enabled

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    enable_prometheus: bool = True
    enable_health_checks: bool = True
    enable_logging: bool = True
    enable_tracing: bool = False
    log_level: str = "INFO"
    metrics_port: int = 8000
    
    def is_monitoring_enabled(self) -> bool:
        """Check if any monitoring is enabled"""
        return any([
            self.enable_prometheus,
            self.enable_health_checks,
            self.enable_logging,
            self.enable_tracing
        ])

class IntegrationConfig:
    """Configuration for external integrations with full protection"""
    
    def __init__(self):
        """Initialize configuration with protection"""
        self._circuit_breaker = None
        self._config_cache: Dict[str, Any] = {}
        self._load_lock = asyncio.Lock()
        self._initialized = False
        
        # Initialize configuration objects
        self.supabase = SupabaseConfig()
        self.api = APIConfig()
        self.tools = ToolConfig()
        self.monitoring = MonitoringConfig()
        
        # Load configuration with protection
        self._initialize_config()
    
    def _initialize_config(self) -> None:
        """Initialize configuration synchronously with error handling"""
        try:
            logger.info("Loading configuration from environment")
            
            # Load Supabase configuration
            self.supabase.url = self._safe_get_env("SUPABASE_URL", "")
            self.supabase.key = self._safe_get_env("SUPABASE_KEY", "")
            self.supabase.service_key = self._safe_get_env("SUPABASE_SERVICE_KEY", "")
            self.supabase.db_password = self._safe_get_env("SUPABASE_DB_PASSWORD", "")
            
            # Load API keys
            self.api.groq_api_key = self._safe_get_env("GROQ_API_KEY", "")
            self.api.openai_api_key = self._safe_get_env("OPENAI_API_KEY", "")
            self.api.anthropic_api_key = self._safe_get_env("ANTHROPIC_API_KEY", "")
            self.api.elevenlabs_api_key = self._safe_get_env("ELEVENLABS_API_KEY", "")
            
            # Load tool flags
            self.tools.enable_file_tools = self._safe_get_bool("ENABLE_FILE_TOOLS", True)
            self.tools.enable_web_tools = self._safe_get_bool("ENABLE_WEB_TOOLS", True)
            self.tools.enable_code_tools = self._safe_get_bool("ENABLE_CODE_TOOLS", True)
            self.tools.enable_media_tools = self._safe_get_bool("ENABLE_MEDIA_TOOLS", True)
            self.tools.enable_communication_tools = self._safe_get_bool("ENABLE_COMMUNICATION_TOOLS", True)
            
            # Load monitoring flags
            self.monitoring.enable_prometheus = self._safe_get_bool("ENABLE_PROMETHEUS", True)
            self.monitoring.enable_health_checks = self._safe_get_bool("ENABLE_HEALTH_CHECKS", True)
            self.monitoring.enable_logging = self._safe_get_bool("ENABLE_LOGGING", True)
            self.monitoring.enable_tracing = self._safe_get_bool("ENABLE_TRACING", False)
            self.monitoring.log_level = self._safe_get_env("LOG_LEVEL", "INFO")
            self.monitoring.metrics_port = self._safe_get_int("METRICS_PORT", 8000)
            
            self._initialized = True
            logger.info("Configuration loaded successfully", extra={
                "supabase_configured": self.supabase.is_valid(),
                "apis_configured": {
                    "groq": bool(self.api.groq_api_key),
                    "openai": bool(self.api.openai_api_key),
                    "anthropic": bool(self.api.anthropic_api_key)
                },
                "tools_enabled": self.tools.get_enabled_tools(),
                "monitoring_enabled": self.monitoring.is_monitoring_enabled()
            })
            
        except Exception as e:
            logger.error("Failed to load configuration", error=e)
            self._set_safe_defaults()
            raise
    
    def _safe_get_env(self, key: str, default: str = "") -> str:
        """Safely get environment variable with validation and caching"""
        # Check cache first
        if key in self._config_cache:
            return self._config_cache[key]
        
        try:
            value = os.environ.get(key, default)
            
            # Validate URLs
            if key.endswith("_URL") and value and value != default:
                if not any(value.startswith(prefix) for prefix in ["http://", "https://", "ws://", "wss://"]):
                    logger.warning("Invalid URL format", extra={
                        "key": key,
                        "value_preview": value[:20] + "..." if len(value) > 20 else value
                    })
                    value = default
            
            # Validate API keys (basic check)
            if key.endswith("_KEY") and value and value != default:
                if len(value) < 10:  # Most API keys are longer
                    logger.warning("Suspicious API key length", extra={"key": key, "length": len(value)})
            
            # Cache the value
            self._config_cache[key] = value
            return value
            
        except Exception as e:
            logger.error("Environment variable access failed", extra={"key": key, "error": str(e)})
            return default
    
    def _safe_get_bool(self, key: str, default: bool = False) -> bool:
        """Safely get boolean environment variable"""
        try:
            value = self._safe_get_env(key, str(default))
            return value.lower() in ("true", "1", "yes", "on", "enabled")
        except Exception as e:
            logger.error("Boolean conversion failed", extra={"key": key, "error": str(e)})
            return default
    
    def _safe_get_int(self, key: str, default: int = 0) -> int:
        """Safely get integer environment variable"""
        try:
            value = self._safe_get_env(key, str(default))
            return int(value)
        except (ValueError, TypeError) as e:
            logger.error("Integer conversion failed", extra={"key": key, "error": str(e)})
            return default
    
    def _set_safe_defaults(self) -> None:
        """Set safe default values when configuration fails"""
        logger.info("Setting safe default configuration values")
        
        # Disable external integrations
        self.tools.enable_file_tools = False
        self.tools.enable_web_tools = False
        self.tools.enable_code_tools = False
        self.tools.enable_media_tools = False
        self.tools.enable_communication_tools = False
        
        # Keep monitoring enabled for debugging
        self.monitoring.enable_logging = True
        self.monitoring.log_level = "DEBUG"
    
    @circuit_breaker("config_check", CircuitBreakerConfig(
        failure_threshold=5, 
        recovery_timeout=15,
        success_threshold=2
    ))
    async def is_configured_safe(self) -> bool:
        """Check if configuration is valid with circuit breaker protection"""
        try:
            return self.supabase.is_valid()
        except Exception as e:
            logger.error("Configuration check failed", error=e)
            return False
    
    def is_configured(self) -> bool:
        """Synchronous configuration check (DEPRECATED - use is_configured_safe)"""
        logger.warning("Using deprecated is_configured() method - switch to is_configured_safe()")
        return self.supabase.is_valid()
    
    @circuit_breaker("config_reload", CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30
    ))
    async def reload_config(self) -> bool:
        """Reload configuration from environment with protection"""
        async with self._load_lock:
            try:
                logger.info("Reloading configuration")
                
                # Clear cache
                self._config_cache.clear()
                
                # Reload
                self._initialize_config()
                
                logger.info("Configuration reloaded successfully")
                return True
                
            except Exception as e:
                logger.error("Configuration reload failed", error=e)
                self._set_safe_defaults()
                return False
    
    @circuit_breaker("config_validate", CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=10
    ))
    async def validate_config(self) -> Dict[str, bool]:
        """Validate all configuration sections"""
        try:
            validation_results = {
                "supabase": self.supabase.is_valid(),
                "groq_api": self.api.is_valid("groq"),
                "openai_api": self.api.is_valid("openai"),
                "anthropic_api": self.api.is_valid("anthropic"),
                "tools": len(self.tools.get_enabled_tools()) > 0,
                "monitoring": self.monitoring.is_monitoring_enabled()
            }
            
            logger.info("Configuration validation completed", extra=validation_results)
            return validation_results
            
        except Exception as e:
            logger.error("Configuration validation failed", error=e)
            return {}
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging/debugging"""
        return {
            "initialized": self._initialized,
            "supabase_configured": self.supabase.is_valid(),
            "api_keys_configured": {
                "groq": bool(self.api.groq_api_key),
                "openai": bool(self.api.openai_api_key),
                "anthropic": bool(self.api.anthropic_api_key),
                "elevenlabs": bool(self.api.elevenlabs_api_key)
            },
            "tools_enabled": self.tools.get_enabled_tools(),
            "monitoring": {
                "prometheus": self.monitoring.enable_prometheus,
                "health_checks": self.monitoring.enable_health_checks,
                "logging": self.monitoring.enable_logging,
                "tracing": self.monitoring.enable_tracing,
                "log_level": self.monitoring.log_level
            }
        }
    
    async def get_service_config(self, service: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific service"""
        try:
            if service == "supabase":
                return self.supabase.to_dict() if self.supabase.is_valid() else None
            elif service in ["groq", "openai", "anthropic", "elevenlabs"]:
                api_key = getattr(self.api, f"{service}_api_key", "")
                return {"api_key": api_key} if api_key else None
            else:
                logger.warning("Unknown service requested", extra={"service": service})
                return None
                
        except Exception as e:
            logger.error("Failed to get service config", extra={"service": service}, error=e)
            return None

# Global instance with lazy initialization
_integration_config: Optional[IntegrationConfig] = None
_config_lock = asyncio.Lock()

async def get_integration_config() -> IntegrationConfig:
    """Get or create integration config instance asynchronously"""
    global _integration_config
    
    if _integration_config is None:
        async with _config_lock:
            if _integration_config is None:
                _integration_config = IntegrationConfig()
    
    return _integration_config

def get_integration_config_sync() -> IntegrationConfig:
    """Get or create integration config instance synchronously"""
    global _integration_config
    
    if _integration_config is None:
        _integration_config = IntegrationConfig()
    
    return _integration_config

# For backward compatibility
integration_config = get_integration_config_sync()

# Export main items
__all__ = [
    'IntegrationConfig',
    'SupabaseConfig',
    'APIConfig',
    'ToolConfig', 
    'MonitoringConfig',
    'get_integration_config',
    'get_integration_config_sync',
    'integration_config'
] 