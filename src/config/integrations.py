"""
Integration configuration for external services
"""

import os
import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from src.infrastructure.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    circuit_breaker
)

logger = logging.getLogger(__name__)

@dataclass
class IntegrationConfig:
    """Configuration for external integrations"""
    
    # Supabase configuration
    supabase_url: str = ""
    supabase_key: str = ""
    
    # API configurations
    groq_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    
    # Tool configurations
    enable_file_tools: bool = True
    enable_web_tools: bool = True
    enable_code_tools: bool = True
    enable_media_tools: bool = True
    
    # Monitoring configurations
    enable_prometheus: bool = True
    enable_health_checks: bool = True
    enable_logging: bool = True
    
    def __init__(self):
        self._circuit_breaker = None
        # Load configuration synchronously for initialization
        self._load_config_sync()
    
    def _load_config_sync(self):
        """Load configuration synchronously during initialization"""
        try:
            self.supabase_url = self._safe_get_env("SUPABASE_URL", "")
            self.supabase_key = self._safe_get_env("SUPABASE_KEY", "")
            self.groq_api_key = self._safe_get_env("GROQ_API_KEY", "")
            self.openai_api_key = self._safe_get_env("OPENAI_API_KEY", "")
            self.anthropic_api_key = self._safe_get_env("ANTHROPIC_API_KEY", "")
            
            # Tool flags
            self.enable_file_tools = self._safe_get_env("ENABLE_FILE_TOOLS", "true").lower() == "true"
            self.enable_web_tools = self._safe_get_env("ENABLE_WEB_TOOLS", "true").lower() == "true"
            self.enable_code_tools = self._safe_get_env("ENABLE_CODE_TOOLS", "true").lower() == "true"
            self.enable_media_tools = self._safe_get_env("ENABLE_MEDIA_TOOLS", "true").lower() == "true"
            
            # Monitoring flags
            self.enable_prometheus = self._safe_get_env("ENABLE_PROMETHEUS", "true").lower() == "true"
            self.enable_health_checks = self._safe_get_env("ENABLE_HEALTH_CHECKS", "true").lower() == "true"
            self.enable_logging = self._safe_get_env("ENABLE_LOGGING", "true").lower() == "true"
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load configuration", extra={"error": str(e)})
            raise
    
    def _safe_get_env(self, key: str, default: str = "") -> str:
        """Safely get environment variable with validation"""
        try:
            value = os.environ.get(key, default)
            if key.endswith("_URL") and value:
                # Validate URL format
                if not value.startswith(("http://", "https://")):
                    logger.warning("Invalid URL format", extra={"key": key, "value": value})
                    return default
            return value
        except Exception as e:
            logger.error("Environment variable access failed", 
                        extra={"key": key, "error": str(e)})
            return default
    
    @circuit_breaker("config_check", CircuitBreakerConfig(failure_threshold=5, recovery_timeout=15))
    async def is_configured_safe(self) -> bool:
        """Check if configuration is valid with circuit breaker protection"""
        try:
            return bool(self.supabase_url and self.supabase_key)
        except Exception as e:
            logger.error("Configuration check failed", extra={"error": str(e)})
            return False
    
    @circuit_breaker("config_validation", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30))
    async def validate_supabase_connection(self) -> bool:
        """Validate Supabase connection with circuit breaker protection"""
        if not self.supabase_url or not self.supabase_key:
            return False
            
        try:
            from supabase import create_client
            client = create_client(self.supabase_url, self.supabase_key)
            
            # Test connection with a simple query
            result = await client.table('_test_connection').select('*').limit(1).execute()
            return True
            
        except Exception as e:
            logger.error("Supabase connection validation failed", extra={"error": str(e)})
            return False
    
    @circuit_breaker("config_validation", CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30))
    async def validate_api_keys(self) -> Dict[str, bool]:
        """Validate API keys with circuit breaker protection"""
        validation_results = {
            'groq': bool(self.groq_api_key),
            'openai': bool(self.openai_api_key),
            'anthropic': bool(self.anthropic_api_key)
        }
        
        # Test API connectivity if keys are present
        if self.groq_api_key:
            try:
                # Simple API test
                validation_results['groq'] = await self._test_groq_api()
            except Exception as e:
                logger.error("Groq API validation failed", extra={"error": str(e)})
                validation_results['groq'] = False
                
        return validation_results
    
    async def _test_groq_api(self) -> bool:
        """Test Groq API connectivity"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    'https://api.groq.com/openai/v1/models',
                    headers={'Authorization': f'Bearer {self.groq_api_key}'},
                    timeout=5
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error("Groq API test failed", extra={"error": str(e)})
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'supabase_url': self.supabase_url,
            'supabase_key': '***' if self.supabase_key else '',
            'groq_api_key': '***' if self.groq_api_key else '',
            'openai_api_key': '***' if self.openai_api_key else '',
            'anthropic_api_key': '***' if self.anthropic_api_key else '',
            'enable_file_tools': self.enable_file_tools,
            'enable_web_tools': self.enable_web_tools,
            'enable_code_tools': self.enable_code_tools,
            'enable_media_tools': self.enable_media_tools,
            'enable_prometheus': self.enable_prometheus,
            'enable_health_checks': self.enable_health_checks,
            'enable_logging': self.enable_logging
        } 