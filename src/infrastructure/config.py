import os

from src.agents.advanced_agent_fsm import Agent
# TODO: Fix undefined variables: self

class Config:
    """Configuration for GAIA Agent in Hugging Face Space"""

    # API Keys (all from Space secrets)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    brave_api_key: str = os.getenv("BRAVE_API_KEY", "")
    youtube_api_key: str = os.getenv("YOUTUBE_API_KEY", "")
    hf_token: str = os.getenv("HF_TOKEN", "")
    huggingface_api_token: str = os.getenv("HUGGINGFACE_API_TOKEN", "")

    # Supabase Configuration
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_KEY", "")
    supabase_service_key: str = os.getenv("SUPABASE_SERVICE_KEY", "")
    supabase_db_password: str = os.getenv("SUPABASE_DB_PASSWORD", "")

    # LangSmith Tracing (for debugging)
    langsmith_tracing: str = os.getenv("LANGSMITH_TRACING", "")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "")
    langsmith_endpoint: str = os.getenv("LANGSMITH_ENDPOINT", "")
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")

    # Model preferences based on available APIs
    @property
    def primary_model(self) -> str:
        """Choose best available model"""
        if self.anthropic_api_key:
            return "claude-3-opus-20240229"
        elif self.groq_api_key:
            return "llama-3.3-70b-versatile"
        elif self.google_api_key:
            return "gemini-pro"
        return "llama-3.1-8b-instant"  # fallback

    @property
    def search_provider(self) -> str:
        """Choose best available search"""
        if self.tavily_api_key:
            return "tavily"
        elif self.brave_api_key:
            return "brave"
        elif self.google_api_key:
            return "google"
        return "duckduckgo"  # free fallback

    @property
    def is_tracing_enabled(self) -> bool:
        """Check if LangSmith tracing is configured"""
        return bool(self.langsmith_api_key and self.langsmith_project)

    @property
    def has_database(self) -> bool:
        """Check if Supabase is configured"""
        return bool(self.supabase_url and self.supabase_key)

    def validate(self) -> tuple[bool, list[str]]:
        """Validate configuration and return issues"""
        issues = []

        # Check for at least one LLM API key
        if not any([self.groq_api_key, self.anthropic_api_key, self.google_api_key]):
            issues.append("No LLM API key found (Groq, Anthropic, or Google)")

        # Check for search capability
        if not any([self.tavily_api_key, self.brave_api_key, self.google_api_key]):
            issues.append("No search API configured (Tavily, Brave, or Google)")

        return len(issues) == 0, issues

# Global instance
config = Config()

# Validate on import
is_valid, issues = config.validate()
if not is_valid:
    import logging
    logger = logging.getLogger(__name__)
    for issue in issues:
        logger.warning("Config issue: {}", extra={"issue": issue})