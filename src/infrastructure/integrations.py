"""
Centralized Integration Configuration
Manages all integration settings for Supabase, LangChain, CrewAI, LlamaIndex, and GAIA
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class SupabaseConfig:
    """Supabase integration configuration"""
    url: str = ""
    key: str = ""
    service_key: str = ""
    db_password: str = ""
    collection_name: str = "gaia_knowledge"
    enable_realtime: bool = True
    connection_pool_size: int = 10
    batch_size: int = 100
    
    def is_configured(self) -> bool:
        return bool(self.url and self.key)
    
    def get_connection_string(self) -> str:
        if not self.is_configured():
            return ""
        return f"postgresql://postgres:{self.db_password}@{self.url.replace('https://', '')}:5432/postgres"

@dataclass
class LangChainConfig:
    """LangChain integration configuration"""
    enable_memory: bool = True
    memory_type: str = "conversation_buffer"
    max_memory_tokens: int = 2000
    enable_callbacks: bool = True
    enable_parallel_execution: bool = True
    max_parallel_tools: int = 3
    tracing_enabled: bool = False
    langsmith_project: str = ""
    langsmith_api_key: str = ""
    
    def is_tracing_configured(self) -> bool:
        return bool(self.langsmith_api_key and self.langsmith_project)

@dataclass
class CrewAIConfig:
    """CrewAI integration configuration"""
    enable_multi_agent: bool = True
    max_agents: int = 5
    agent_timeout: int = 300
    enable_task_factory: bool = True
    enable_executor: bool = True
    default_crew_size: int = 3
    agent_roles: List[str] = field(default_factory=lambda: [
        "Researcher", "Executor", "Synthesizer"
    ])

@dataclass
class LlamaIndexConfig:
    """LlamaIndex integration configuration"""
    enable_hierarchical_indexing: bool = True
    enable_multi_modal: bool = True
    enable_incremental_updates: bool = True
    enable_caching: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_top_k: int = 5
    similarity_cutoff: float = 0.7
    storage_path: str = "./knowledge_cache"
    enable_hybrid_search: bool = True

@dataclass
class GAIAConfig:
    """GAIA benchmark specific configuration"""
    enable_gaia_tools: bool = True
    enable_gaia_metrics: bool = True
    enable_gaia_caching: bool = True
    gaia_test_patterns: List[str] = field(default_factory=lambda: [
        "video_analysis", "text_processing", "mathematical", 
        "factual", "creative", "multimodal"
    ])
    gaia_timeout: int = 600
    gaia_retry_attempts: int = 3

class IntegrationConfig:
    """Centralized configuration for all integrations"""
    
    def __init__(self):
        self.supabase = SupabaseConfig()
        self.langchain = LangChainConfig()
        self.crewai = CrewAIConfig()
        self.llamaindex = LlamaIndexConfig()
        self.gaia = GAIAConfig()
        self._load_from_environment()
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        
        # Supabase
        self.supabase.url = os.getenv("SUPABASE_URL", "")
        self.supabase.key = os.getenv("SUPABASE_KEY", "")
        self.supabase.service_key = os.getenv("SUPABASE_SERVICE_KEY", "")
        self.supabase.db_password = os.getenv("SUPABASE_DB_PASSWORD", "")
        
        # LangChain
        self.langchain.tracing_enabled = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
        self.langchain.langsmith_project = os.getenv("LANGSMITH_PROJECT", "")
        self.langchain.langsmith_api_key = os.getenv("LANGSMITH_API_KEY", "")
        
        # CrewAI
        self.crewai.enable_multi_agent = os.getenv("CREWAI_ENABLED", "true").lower() == "true"
        self.crewai.max_agents = int(os.getenv("CREWAI_MAX_AGENTS", "5"))
        
        # LlamaIndex
        self.llamaindex.storage_path = os.getenv("LLAMAINDEX_STORAGE_PATH", "./knowledge_cache")
        self.llamaindex.chunk_size = int(os.getenv("LLAMAINDEX_CHUNK_SIZE", "512"))
        
        # GAIA
        self.gaia.enable_gaia_tools = os.getenv("GAIA_TOOLS_ENABLED", "true").lower() == "true"
        self.gaia.gaia_timeout = int(os.getenv("GAIA_TIMEOUT", "600"))
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values"""
        try:
            for section, values in updates.items():
                if hasattr(self, section):
                    section_config = getattr(self, section)
                    for key, value in values.items():
                        if hasattr(section_config, key):
                            setattr(section_config, key, value)
            
            logger.info("Configuration updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False
    
    async def validate(self) -> tuple[bool, List[str]]:
        """Validate configuration and return issues"""
        issues = []
        
        # Validate Supabase if enabled
        if await self.supabase.is_configured_safe():
            if not self.supabase.url.startswith("https://"):
                issues.append("Supabase URL must start with https://")
        
        # Validate LangChain tracing
        if self.langchain.tracing_enabled and not self.langchain.is_tracing_configured():
            issues.append("LangSmith tracing enabled but not configured")
        
        # Validate CrewAI
        if self.crewai.max_agents < 1:
            issues.append("CrewAI max_agents must be at least 1")
        
        # Validate LlamaIndex
        if self.llamaindex.chunk_size < 100:
            issues.append("LlamaIndex chunk_size must be at least 100")
        
        return len(issues) == 0, issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "supabase": {
                "url": self.supabase.url,
                "key": "***" if self.supabase.key else "",
                "collection_name": self.supabase.collection_name,
                "enable_realtime": self.supabase.enable_realtime
            },
            "langchain": {
                "enable_memory": self.langchain.enable_memory,
                "enable_callbacks": self.langchain.enable_callbacks,
                "tracing_enabled": self.langchain.tracing_enabled
            },
            "crewai": {
                "enable_multi_agent": self.crewai.enable_multi_agent,
                "max_agents": self.crewai.max_agents,
                "agent_roles": self.crewai.agent_roles
            },
            "llamaindex": {
                "enable_hierarchical_indexing": self.llamaindex.enable_hierarchical_indexing,
                "enable_multi_modal": self.llamaindex.enable_multi_modal,
                "storage_path": self.llamaindex.storage_path
            },
            "gaia": {
                "enable_gaia_tools": self.gaia.enable_gaia_tools,
                "enable_gaia_metrics": self.gaia.enable_gaia_metrics,
                "gaia_timeout": self.gaia.gaia_timeout
            }
        }
    
    def save_to_file(self, file_path: str) -> bool:
        """Save configuration to file"""
        try:
            config_dict = self.to_dict()
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Configuration saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def load_from_file(self, file_path: str) -> bool:
        """Load configuration from file"""
        try:
            if not Path(file_path).exists():
                logger.warning(f"Configuration file {file_path} not found")
                return False
            
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            
            # Update configuration with loaded values
            self.update_config(config_dict)
            logger.info(f"Configuration loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False

# Global integration configuration instance
integration_config = IntegrationConfig()

# Validate on import
is_valid, issues = integration_config.validate()
if not is_valid:
    for issue in issues:
        logger.warning(f"Integration config issue: {issue}") 