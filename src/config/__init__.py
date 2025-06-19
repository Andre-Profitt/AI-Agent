"""
Configuration package for AI Agent
Provides centralized configuration management for all integrations
"""

from .integrations import IntegrationConfig, integration_config

__all__ = ['IntegrationConfig', 'integration_config']
