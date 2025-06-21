"""
Centralized security configuration
"""
import os
from typing import Dict, List, Optional
from pydantic import BaseSettings, SecretStr, validator
import secrets

class SecuritySettings(BaseSettings):
    """Security-related settings with validation"""
    
    # JWT Settings
    jwt_secret_key: SecretStr = SecretStr(os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32)))
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 30
    jwt_refresh_expiration_days: int = 7
    
    # Password Policy
    min_password_length: int = 12
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_special_chars: bool = True
    password_history_size: int = 5
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    general_rate_limit_per_minute: int = 60
    general_rate_limit_per_hour: int = 1000
    auth_rate_limit_per_minute: int = 10
    auth_rate_limit_per_hour: int = 100
    
    # Session Security
    session_timeout_minutes: int = 30
    max_concurrent_sessions: int = 5
    session_secure_cookie: bool = True
    session_httponly: bool = True
    session_samesite: str = "strict"
    
    # CORS Settings
    cors_enabled: bool = True
    cors_origins: List[str] = ["http://localhost:3000"]
    cors_allow_credentials: bool = True
    cors_max_age: int = 3600
    
    # Security Headers
    enable_security_headers: bool = True
    hsts_max_age: int = 31536000  # 1 year
    content_security_policy: str = "default-src 'self'"
    
    # Input Validation
    max_input_length: int = 10000
    max_file_size_mb: int = 10
    allowed_file_extensions: List[str] = [".txt", ".json", ".csv", ".md"]
    
    # Audit Logging
    enable_audit_logging: bool = True
    audit_log_retention_days: int = 90
    
    # API Security
    api_key_header: str = "X-API-Key"
    require_api_key: bool = False
    api_key_rotation_days: int = 90
    
    # Database Security
    encrypt_sensitive_data: bool = True
    encryption_key: SecretStr = SecretStr(os.getenv("ENCRYPTION_KEY", secrets.token_urlsafe(32)))
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('jwt_secret_key')
    def validate_jwt_secret(cls, v):
        if len(v.get_secret_value()) < 32:
            raise ValueError("JWT secret key must be at least 32 characters")
        return v
    
    class Config:
        env_prefix = "SECURITY_"
        case_sensitive = False

# Global security settings instance
security_settings = SecuritySettings()

# Security utility functions
def validate_password(password: str) -> tuple[bool, Optional[str]]:
    """Validate password against security policy"""
    settings = security_settings
    
    if len(password) < settings.min_password_length:
        return False, f"Password must be at least {settings.min_password_length} characters"
    
    if settings.require_uppercase and not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    
    if settings.require_lowercase and not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    
    if settings.require_numbers and not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    
    if settings.require_special_chars and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
        return False, "Password must contain at least one special character"
    
    return True, None

def generate_secure_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token"""
    return secrets.token_urlsafe(length)

def hash_sensitive_data(data: str) -> str:
    """Hash sensitive data for storage"""
    import hashlib
    import hmac
    
    key = security_settings.encryption_key.get_secret_value().encode()
    return hmac.new(key, data.encode(), hashlib.sha256).hexdigest()
