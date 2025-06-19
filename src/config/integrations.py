"""
Integration configuration for external services
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

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
        # Load from environment
        self.supabase_url = os.getenv('SUPABASE_URL', '')
        self.supabase_key = os.getenv('SUPABASE_KEY', '')
        self.groq_api_key = os.getenv('GROQ_API_KEY', '')
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY', '')
        
        # Tool flags
        self.enable_file_tools = os.getenv('ENABLE_FILE_TOOLS', 'true').lower() == 'true'
        self.enable_web_tools = os.getenv('ENABLE_WEB_TOOLS', 'true').lower() == 'true'
        self.enable_code_tools = os.getenv('ENABLE_CODE_TOOLS', 'true').lower() == 'true'
        self.enable_media_tools = os.getenv('ENABLE_MEDIA_TOOLS', 'true').lower() == 'true'
        
        # Monitoring flags
        self.enable_prometheus = os.getenv('ENABLE_PROMETHEUS', 'true').lower() == 'true'
        self.enable_health_checks = os.getenv('ENABLE_HEALTH_CHECKS', 'true').lower() == 'true'
        self.enable_logging = os.getenv('ENABLE_LOGGING', 'true').lower() == 'true'
    
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