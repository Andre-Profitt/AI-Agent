"""
Source package for the AI Agent application.
This package contains all the core functionality and components.
"""

"""AI Agent System - Core Package"""
__version__ = "1.0.0"

# Core exports
from .config.settings import Settings
from .config.integrations import IntegrationConfig
from .core.monitoring import MetricsCollector
from .core.health_check import HealthChecker
from .services.integration_hub import IntegrationHub
from .database.supabase_manager import SupabaseManager
from .tools.registry import ToolRegistry

__all__ = [
    'Settings',
    'IntegrationConfig', 
    'MetricsCollector',
    'HealthChecker',
    'IntegrationHub',
    'SupabaseManager',
    'ToolRegistry'
] 