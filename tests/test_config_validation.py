from migrations.env import config

from src.database_extended import failure_count

from src.infrastructure.integrations import IntegrationConfig
from unittest.mock import patch
# TODO: Fix undefined variables: config, failure_count, result
"""
Test configuration validation and circuit breaker protection
"""

import pytest

from src.config.integrations import IntegrationConfig
from src.infrastructure.resilience.circuit_breaker import CircuitBreakerOpenError

class TestConfigValidation:
    """Test configuration circuit breaker protection"""

    @pytest.mark.asyncio
    async def test_env_var_protection(self):
        """Verify environment variable access is protected"""
        config = IntegrationConfig()

        # Test safe environment variable access
        with patch('os.environ.get', return_value="test_value"):
            result = config._safe_get_env("TEST_KEY", "default")
            assert result == "test_value"

    @pytest.mark.asyncio
    async def test_config_circuit_breaker(self):
        """Test configuration circuit breaker triggers correctly"""
        config = IntegrationConfig()

        # Mock to force failures
        with patch.object(config, '_safe_get_env', side_effect=Exception("Config error")):
            # Should fail 3 times then open circuit
            for i in range(3):
                with pytest.raises(Exception):
                    await config._load_config()

            # Circuit should now be open
            with pytest.raises(CircuitBreakerOpenError):
                await config._load_config()

    @pytest.mark.asyncio
    async def test_is_configured_safe(self):
        """Test safe configuration checking"""
        config = IntegrationConfig()

        # Test with valid configuration
        with patch.object(config, 'supabase_url', "https://test.supabase.co"):
            with patch.object(config, 'supabase_key', "test_key"):
                result = await config.is_configured_safe()
                assert result is True

        # Test with invalid configuration
        with patch.object(config, 'supabase_url', ""):
            with patch.object(config, 'supabase_key', ""):
                result = await config.is_configured_safe()
                assert result is False

    @pytest.mark.asyncio
    async def test_url_validation(self):
        """Test URL format validation"""
        config = IntegrationConfig()

        # Test valid URL
        with patch('os.environ.get', return_value="https://test.supabase.co"):
            result = config._safe_get_env("SUPABASE_URL", "")
            assert result == "https://test.supabase.co"

        # Test invalid URL
        with patch('os.environ.get', return_value="invalid-url"):
            result = config._safe_get_env("SUPABASE_URL", "")
            assert result == "invalid-url"  # Should return the value but log warning

    @pytest.mark.asyncio
    async def test_config_error_handling(self):
        """Test configuration error handling"""
        config = IntegrationConfig()

        # Test with missing environment variables
        with patch('os.environ.get', return_value=None):
            result = config._safe_get_env("MISSING_KEY", "default_value")
            assert result == "default_value"

    @pytest.mark.asyncio
    async def test_config_recovery(self):
        """Test configuration recovery after failures"""
        config = IntegrationConfig()

        # Simulate failures then recovery
        failure_count = 0

        def mock_safe_get_env(self, key, default):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:
                raise Exception("Config error")
            return "recovered_value"

        with patch.object(config, '_safe_get_env', mock_safe_get_env):
            # First 3 calls should fail
            for i in range(3):
                with pytest.raises(Exception):
                    await config._load_config()

            # 4th call should succeed
            await config._load_config()
            assert failure_count == 4
