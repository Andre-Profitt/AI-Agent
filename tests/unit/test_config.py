"""
Unit tests for configuration module
"""

import os
import pytest
from src.config import Config, Environment, ModelConfig, APIConfig, PerformanceConfig


class TestConfig:
    """Test configuration functionality"""
    
    def test_environment_detection(self):
        """Test environment detection logic"""
        # Save original env vars
        original_space_id = os.getenv("SPACE_ID")
        original_env = os.getenv("ENV")
        
        try:
            # Test Hugging Face Space detection
            os.environ["SPACE_ID"] = "test-space"
            config = Config()
            assert config.environment == Environment.HUGGINGFACE_SPACE
            
            # Test production environment
            os.environ.pop("SPACE_ID", None)
            os.environ["ENV"] = "production"
            config = Config()
            assert config.environment == Environment.PRODUCTION
            
            # Test development environment (default)
            os.environ.pop("ENV", None)
            config = Config()
            assert config.environment == Environment.DEVELOPMENT
            
        finally:
            # Restore original env vars
            if original_space_id:
                os.environ["SPACE_ID"] = original_space_id
            else:
                os.environ.pop("SPACE_ID", None)
            
            if original_env:
                os.environ["ENV"] = original_env
            else:
                os.environ.pop("ENV", None)
    
    def test_model_config(self):
        """Test model configuration"""
        model_config = ModelConfig()
        
        # Test reasoning models
        assert "primary" in model_config.REASONING_MODELS
        assert "fast" in model_config.REASONING_MODELS
        
        # Test function calling models
        assert "primary" in model_config.FUNCTION_CALLING_MODELS
        
        # Test text generation models
        assert "primary" in model_config.TEXT_GENERATION_MODELS
        
        # Test vision models
        assert "primary" in model_config.VISION_MODELS
    
    def test_api_config_loading(self):
        """Test API configuration loading from environment"""
        # Save original values
        original_groq = os.getenv("GROQ_API_KEY")
        
        try:
            # Set test API key
            os.environ["GROQ_API_KEY"] = "test-key"
            
            api_config = APIConfig()
            assert api_config.GROQ_API_KEY == "test-key"
            assert api_config.GROQ_BASE_URL == "https://api.groq.com/openai/v1"
            assert api_config.GROQ_TPM_LIMIT == 6000
            
        finally:
            # Restore original
            if original_groq:
                os.environ["GROQ_API_KEY"] = original_groq
            else:
                os.environ.pop("GROQ_API_KEY", None)
    
    def test_performance_config(self):
        """Test performance configuration"""
        perf_config = PerformanceConfig()
        
        assert perf_config.MAX_PARALLEL_WORKERS == 8
        assert perf_config.REQUEST_SPACING == 0.5
        assert perf_config.CACHE_MAX_SIZE == 1000
        assert perf_config.CACHE_TTL_SECONDS == 3600
        assert perf_config.MAX_REASONING_STEPS == 15
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = Config()
        
        # In development, should not require GROQ_API_KEY
        os.environ.pop("GROQ_API_KEY", None)
        os.environ.pop("ENV", None)
        config = Config()
        issues = config.validate()
        # Should not have critical issues in development
        assert not any("GROQ_API_KEY is required" in issue for issue in issues)
        
        # In production, should require GROQ_API_KEY
        os.environ["ENV"] = "production"
        config = Config()
        issues = config.validate()
        if not os.getenv("GROQ_API_KEY"):
            assert any("GROQ_API_KEY is required" in issue for issue in issues)
    
    def test_get_model(self):
        """Test model selection"""
        config = Config()
        
        # Test reasoning model selection
        model = config.get_model("reasoning", "primary")
        assert model == config.models.REASONING_MODELS["primary"]
        
        # Test fast model selection
        model = config.get_model("reasoning", "fast")
        assert model == config.models.REASONING_MODELS["fast"]
        
        # Test fallback to primary
        model = config.get_model("reasoning", "nonexistent")
        assert model == config.models.REASONING_MODELS["primary"]
    
    def test_environment_overrides(self):
        """Test environment-specific overrides"""
        os.environ["SPACE_ID"] = "test-space"
        config = Config()
        
        # In Hugging Face Space, workers should be reduced
        assert config.performance.MAX_PARALLEL_WORKERS == 4
        
        # Cleanup
        os.environ.pop("SPACE_ID", None) 