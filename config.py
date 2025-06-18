"""
Centralized Configuration for AI Agent
=====================================

This module contains all configuration settings, model names, API endpoints,
and constants used throughout the AI Agent application.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class Environment(str, Enum):
    """Application environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    HUGGINGFACE_SPACE = "huggingface_space"


@dataclass
class ModelConfig:
    """Configuration for language models"""
    # Reasoning models - for complex logical thinking
    REASONING_MODELS: Dict[str, str] = None
    # Function calling models - for tool use  
    FUNCTION_CALLING_MODELS: Dict[str, str] = None
    # Text generation models - for final answers
    TEXT_GENERATION_MODELS: Dict[str, str] = None
    # Vision models - for image analysis
    VISION_MODELS: Dict[str, str] = None
    
    def __post_init__(self):
        self.REASONING_MODELS = {
            "primary": "llama-3.3-70b-versatile",
            "fast": "llama-3.1-8b-instant",
            "deep": "deepseek-r1-distill-llama-70b"
        }
        
        self.FUNCTION_CALLING_MODELS = {
            "primary": "llama-3.3-70b-versatile",
            "fast": "llama-3.1-8b-instant",
            "versatile": "llama3-groq-70b-8192-tool-use-preview"
        }
        
        self.TEXT_GENERATION_MODELS = {
            "primary": "llama-3.3-70b-versatile",
            "fast": "llama-3.1-8b-instant",
            "creative": "gemma2-9b-it"
        }
        
        self.VISION_MODELS = {
            "primary": "meta-llama/llama-4-scout-17b-16e-instruct",
            "fallback": "llava-v1.6-mistral-7b-hf"
        }


@dataclass
class APIConfig:
    """Configuration for external APIs"""
    # Groq API
    GROQ_API_KEY: Optional[str] = None
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_TPM_LIMIT: int = 6000  # Tokens per minute
    
    # OpenAI API
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    
    # Tavily API
    TAVILY_API_KEY: Optional[str] = None
    TAVILY_BASE_URL: str = "https://api.tavily.com/v1"
    
    # Supabase
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None
    
    # GAIA Benchmark
    GAIA_API_URL: str = "https://agents-course-unit4-scoring.hf.space"
    
    def __post_init__(self):
        # Load from environment
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
        self.SUPABASE_URL = os.getenv("SUPABASE_URL")
        self.SUPABASE_KEY = os.getenv("SUPABASE_KEY")


@dataclass
class PerformanceConfig:
    """Performance and rate limiting configuration"""
    # Parallel processing
    MAX_PARALLEL_WORKERS: int = 8
    
    # Rate limiting
    API_RATE_LIMIT_BUFFER: int = 5  # Extra seconds between API calls
    REQUEST_SPACING: float = 0.5  # Minimum seconds between requests
    MAX_REQUESTS_PER_MINUTE: int = 60
    
    # Caching
    CACHE_MAX_SIZE: int = 1000
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    
    # Timeouts
    DEFAULT_TIMEOUT: int = 30
    LONG_RUNNING_TIMEOUT: int = 300  # 5 minutes
    
    # FSM Configuration
    MAX_STAGNATION: int = 3
    MAX_RETRIES: int = 3
    MAX_REASONING_STEPS: int = 15


@dataclass 
class LoggingConfig:
    """Logging configuration"""
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "[%(asctime)s] %(levelname)-8s [%(name)s:%(lineno)d] [%(correlation_id)s] %(message)s"
    LOG_FILE: str = "agent_fsm.log"
    LOG_MAX_BYTES: int = 10485760  # 10MB
    LOG_BACKUP_COUNT: int = 5
    
    # Feature flags
    ENABLE_SUPABASE_LOGGING: bool = True
    ENABLE_CORRELATION_IDS: bool = True


@dataclass
class ToolConfig:
    """Tool-specific configuration"""
    # Video analysis
    VIDEO_DOWNLOAD_TIMEOUT: int = 60
    MAX_VIDEO_SIZE_MB: int = 500
    
    # Web search
    SEARCH_MAX_RESULTS: int = 3
    WIKIPEDIA_MAX_PARAGRAPHS: int = 5
    
    # Code execution
    PYTHON_EXEC_TIMEOUT: int = 30
    PYTHON_MAX_OUTPUT_LENGTH: int = 10000
    
    # File operations
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_FILE_EXTENSIONS: List[str] = None
    
    def __post_init__(self):
        self.ALLOWED_FILE_EXTENSIONS = [
            ".txt", ".md", ".pdf", ".docx", ".xlsx", ".csv",
            ".json", ".py", ".js", ".html", ".xml"
        ]


class Config:
    """Main configuration class combining all settings"""
    
    def __init__(self):
        # Detect environment
        self.environment = self._detect_environment()
        
        # Initialize sub-configurations
        self.models = ModelConfig()
        self.api = APIConfig()
        self.performance = PerformanceConfig()
        self.logging = LoggingConfig()
        self.tools = ToolConfig()
        
        # Apply environment-specific overrides
        self._apply_environment_overrides()
    
    def _detect_environment(self) -> Environment:
        """Detect the current environment"""
        if os.getenv("SPACE_ID"):
            return Environment.HUGGINGFACE_SPACE
        elif os.getenv("ENV") == "production":
            return Environment.PRODUCTION
        elif os.getenv("ENV") == "staging":
            return Environment.STAGING
        else:
            return Environment.DEVELOPMENT
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides"""
        if self.environment == Environment.HUGGINGFACE_SPACE:
            # Reduce workers for Hugging Face Spaces
            self.performance.MAX_PARALLEL_WORKERS = 4
            # Use faster models
            self.models.REASONING_MODELS["primary"] = self.models.REASONING_MODELS["fast"]
        
        elif self.environment == Environment.DEVELOPMENT:
            # More verbose logging in development
            self.logging.LOG_LEVEL = "DEBUG"
            # Disable some production features
            self.logging.ENABLE_SUPABASE_LOGGING = False
    
    def get_model(self, task_type: str, preference: str = "primary") -> str:
        """Get the appropriate model for a task type"""
        model_map = {
            "reasoning": self.models.REASONING_MODELS,
            "function_calling": self.models.FUNCTION_CALLING_MODELS,
            "text_generation": self.models.TEXT_GENERATION_MODELS,
            "vision": self.models.VISION_MODELS
        }
        
        models = model_map.get(task_type, self.models.REASONING_MODELS)
        return models.get(preference, models["primary"])
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment in [Environment.PRODUCTION, Environment.HUGGINGFACE_SPACE]
    
    def validate(self) -> List[str]:
        """Validate configuration and return any issues"""
        issues = []
        
        # Check required API keys
        if not self.api.GROQ_API_KEY and self.is_production():
            issues.append("GROQ_API_KEY is required in production")
        
        if not self.api.TAVILY_API_KEY:
            issues.append("TAVILY_API_KEY is missing - web search will be limited")
        
        # Check performance settings
        if self.performance.MAX_PARALLEL_WORKERS > 20:
            issues.append("MAX_PARALLEL_WORKERS too high - may cause rate limiting")
        
        return issues


# Global configuration instance
config = Config()

# Export commonly used values for backward compatibility
GROQ_API_KEY = config.api.GROQ_API_KEY
TAVILY_API_KEY = config.api.TAVILY_API_KEY
MAX_PARALLEL_WORKERS = config.performance.MAX_PARALLEL_WORKERS
DEFAULT_TIMEOUT = config.performance.DEFAULT_TIMEOUT 