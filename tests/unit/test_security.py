"""
Security Tests
Tests for authentication, authorization, and security measures
"""

import pytest
from unittest.mock import Mock, patch
import jwt
from datetime import datetime, timedelta

from src.utils.security_validation import SecurityValidator
from src.config.security import SecurityConfig
from src.api.auth import create_token, verify_token
from src.api.rate_limiter import RateLimiter


class TestSecurityMeasures:
    """Test security implementations"""
    
    @pytest.fixture
    def security_config(self):
        """Create security configuration"""
        return SecurityConfig(
            jwt_secret="test_secret_key_12345",
            jwt_algorithm="HS256",
            token_expire_minutes=30,
            bcrypt_rounds=4,  # Lower for testing
            rate_limit_requests=100,
            rate_limit_window=60
        )
        
    @pytest.fixture
    def validator(self):
        """Create security validator"""
        return SecurityValidator()
        
    def test_sql_injection_detection(self, validator):
        """Test SQL injection detection"""
        # Test malicious inputs
        assert validator.is_sql_injection("'; DROP TABLE users; --")
        assert validator.is_sql_injection("1' OR '1'='1")
        assert validator.is_sql_injection("admin'--")
        
        # Test safe inputs
        assert not validator.is_sql_injection("normal user input")
        assert not validator.is_sql_injection("user@example.com")
        
    def test_xss_detection(self, validator):
        """Test XSS detection"""
        # Test malicious inputs
        assert validator.contains_xss("<script>alert('XSS')</script>")
        assert validator.contains_xss("<img src=x onerror=alert(1)>")
        assert validator.contains_xss("javascript:alert('XSS')")
        
        # Test safe inputs
        assert not validator.contains_xss("Normal text")
        assert not validator.contains_xss("user@example.com")
        
    def test_path_traversal_detection(self, validator):
        """Test path traversal detection"""
        # Test malicious paths
        assert not validator.is_safe_path("../../etc/passwd")
        assert not validator.is_safe_path("/etc/shadow")
        assert not validator.is_safe_path("..\\windows\\system32")
        
        # Test safe paths
        assert validator.is_safe_path("data/file.txt")
        assert validator.is_safe_path("./local/path")
        
    def test_jwt_token_creation(self, security_config):
        """Test JWT token creation"""
        user_id = "test_user_123"
        
        with patch('src.config.security.get_security_config', return_value=security_config):
            token = create_token(user_id)
            
            # Decode token
            payload = jwt.decode(
                token, 
                security_config.jwt_secret, 
                algorithms=[security_config.jwt_algorithm]
            )
            
            assert payload["sub"] == user_id
            assert "exp" in payload
            
    def test_jwt_token_verification(self, security_config):
        """Test JWT token verification"""
        user_id = "test_user_123"
        
        with patch('src.config.security.get_security_config', return_value=security_config):
            # Create token
            token = create_token(user_id)
            
            # Verify token
            verified_user_id = verify_token(token)
            assert verified_user_id == user_id
            
            # Test invalid token
            assert verify_token("invalid.token.here") is None
            
            # Test expired token
            expired_token = jwt.encode(
                {
                    "sub": user_id,
                    "exp": datetime.utcnow() - timedelta(minutes=1)
                },
                security_config.jwt_secret,
                algorithm=security_config.jwt_algorithm
            )
            assert verify_token(expired_token) is None
            
    @pytest.mark.asyncio
    async def test_rate_limiter(self, security_config):
        """Test rate limiting"""
        limiter = RateLimiter(
            max_requests=5,
            window_seconds=60
        )
        
        client_id = "test_client"
        
        # Make 5 requests
        for i in range(5):
            allowed = await limiter.check_rate_limit(client_id)
            assert allowed is True
            
        # 6th request should be blocked
        allowed = await limiter.check_rate_limit(client_id)
        assert allowed is False
        
    def test_password_hashing(self, security_config):
        """Test password hashing"""
        from src.utils.security_validation import hash_password, verify_password
        
        password = "SecurePassword123!"
        
        # Hash password
        hashed = hash_password(password)
        assert hashed != password
        assert hashed.startswith("$2b$")
        
        # Verify correct password
        assert verify_password(password, hashed) is True
        
        # Verify incorrect password
        assert verify_password("wrong_password", hashed) is False
        
    def test_input_sanitization(self, validator):
        """Test input sanitization"""
        # Test various inputs
        assert validator.sanitize_input("<script>alert(1)</script>") == "alert(1)"
        assert validator.sanitize_input("normal text") == "normal text"
        assert validator.sanitize_input("user@example.com") == "user@example.com"
        
    @pytest.mark.asyncio
    async def test_secure_file_operations(self, validator):
        """Test secure file operations"""
        # Test file path validation
        safe_paths = [
            "data/users.json",
            "./config/settings.yaml",
            "logs/app.log"
        ]
        
        unsafe_paths = [
            "../../../etc/passwd",
            "/etc/shadow",
            "C:\\Windows\\System32\\config"
        ]
        
        for path in safe_paths:
            assert validator.is_safe_path(path)
            
        for path in unsafe_paths:
            assert not validator.is_safe_path(path)


class TestSecurityIntegration:
    """Integration tests for security features"""
    
    @pytest.mark.asyncio
    async def test_authenticated_request_flow(self):
        """Test full authentication flow"""
        from src.api.auth import create_token, get_current_user
        from fastapi import HTTPException
        
        user_id = "test_user"
        
        # Create token
        token = create_token(user_id)
        
        # Mock request with token
        mock_credentials = Mock()
        mock_credentials.credentials = token
        
        # Verify user
        with patch('src.api.auth.verify_token', return_value=user_id):
            current_user = await get_current_user(mock_credentials)
            assert current_user == user_id
            
        # Test invalid token
        mock_credentials.credentials = "invalid_token"
        with patch('src.api.auth.verify_token', return_value=None):
            with pytest.raises(HTTPException) as exc:
                await get_current_user(mock_credentials)
            assert exc.value.status_code == 401
