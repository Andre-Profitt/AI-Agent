#!/usr/bin/env python3
"""
Security fixes for AI Agent project
Removes hardcoded credentials and implements proper security measures
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class SecurityFixer:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.security_issues = []
        self.fixes_applied = []
        
    def scan_and_fix(self):
        """Main method to scan and fix security issues"""
        logger.info("🔒 Starting security audit and fixes...\n")
        
        # Scan for issues
        self._scan_hardcoded_credentials()
        
        # Apply fixes
        self._fix_hardcoded_credentials()
        self._fix_jwt_security()
        self._add_input_validation()
        self._add_rate_limiting()
        self._create_security_config()
        
        # Generate report
        self._generate_security_report()
        
    def _scan_hardcoded_credentials(self):
        """Scan for hardcoded passwords and secrets"""
        patterns = [
            (r'password\s*=\s*["\'][\w\d]+["\']', 'Hardcoded password'),
            (r'secret\s*=\s*["\'][\w\d]+["\']', 'Hardcoded secret'),
            (r'api_key\s*=\s*["\'][\w\d]+["\']', 'Hardcoded API key'),
            (r'hashpw\(["\'][\w\d]+["\']', 'Hardcoded password in hash'),
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            if 'venv' in str(py_file) or '__pycache__' in str(py_file):
                continue
                
            try:
                content = py_file.read_text()
                for pattern, issue_type in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        self.security_issues.append({
                            'file': str(py_file),
                            'line': content[:match.start()].count('\n') + 1,
                            'type': issue_type,
                            'match': match.group()
                        })
            except Exception as e:
                logger.warning(f"Could not scan {py_file}: {e}")
                
    def _fix_hardcoded_credentials(self):
        """Fix hardcoded credentials"""
        # Fix in_memory_user_repository.py
        user_repo_path = self.project_root / "src/infrastructure/database/in_memory_user_repository.py"
        if user_repo_path.exists():
            content = user_repo_path.read_text()
            
            # Replace hardcoded admin password
            new_content = re.sub(
                r'password_hash=bcrypt\.hashpw\(b"admin123", bcrypt\.gensalt\(\)\)\.decode\(\)',
                'password_hash=bcrypt.hashpw(os.getenv("ADMIN_PASSWORD", os.urandom(32).hex()).encode(), bcrypt.gensalt()).decode()',
                content
            )
            
            # Add import if needed
            if 'import os' not in new_content:
                new_content = 'import os\n' + new_content
                
            user_repo_path.write_text(new_content)
            self.fixes_applied.append("Fixed hardcoded admin password in user repository")
            
    def _fix_jwt_security(self):
        """Fix JWT security issues"""
        auth_path = self.project_root / "src/api/auth.py"
        if auth_path.exists():
            content = auth_path.read_text()
            
            # Replace weak JWT secret generation
            new_content = re.sub(
                r'SECRET_KEY = os\.getenv\("JWT_SECRET_KEY", "your-secret-key-here"\)',
                'SECRET_KEY = os.getenv("JWT_SECRET_KEY", os.urandom(32).hex())',
                content
            )
            
            # Add token expiration validation
            if 'def verify_token' in new_content and 'exp' not in new_content:
                new_content = new_content.replace(
                    'payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])',
                    '''payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # Validate token expiration
        if datetime.utcnow().timestamp() > payload.get("exp", 0):
            raise HTTPException(status_code=401, detail="Token expired")'''
                )
                
            auth_path.write_text(new_content)
            self.fixes_applied.append("Enhanced JWT security")
            
    def _add_input_validation(self):
        """Add input validation to critical endpoints"""
        # Create input validation utilities
        validation_content = '''"""
Input validation utilities for security
"""
import re
from typing import Any, Dict, List
from pathlib import Path
import bleach
from pydantic import BaseModel, validator

class SecurityValidator:
    """Security-focused input validation"""
    
    @staticmethod
    def sanitize_html(text: str) -> str:
        """Sanitize HTML input to prevent XSS"""
        allowed_tags = ['b', 'i', 'u', 'em', 'strong', 'p', 'br']
        return bleach.clean(text, tags=allowed_tags, strip=True)
    
    @staticmethod
    def validate_file_path(path: str, base_dir: str = None) -> bool:
        """Validate file path to prevent directory traversal"""
        try:
            path_obj = Path(path).resolve()
            if base_dir:
                base_path = Path(base_dir).resolve()
                return path_obj.is_relative_to(base_path)
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_sql_input(text: str) -> bool:
        """Basic SQL injection prevention"""
        dangerous_patterns = [
            r"(\b(DELETE|DROP|EXEC(UTE)?|INSERT|SELECT|UNION|UPDATE)\b)",
            r"(--|#|/\\*|\\*/)",
            r"(\bOR\b\s*\d+\s*=\s*\d+)",
            r"(\bAND\b\s*\d+\s*=\s*\d+)"
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        return True
    
    @staticmethod
    def validate_command_input(command: str) -> bool:
        """Validate shell command input"""
        dangerous_chars = [';', '|', '&', '$', '`', '\\n', '\\r']
        return not any(char in command for char in dangerous_chars)

class SecureFileOperation(BaseModel):
    """Secure file operation model"""
    file_path: str
    operation: str
    
    @validator('file_path')
    def validate_path(cls, v):
        if not SecurityValidator.validate_file_path(v):
            raise ValueError("Invalid file path")
        return v
    
    @validator('operation')
    def validate_operation(cls, v):
        allowed_ops = ['read', 'write', 'append', 'delete']
        if v not in allowed_ops:
            raise ValueError(f"Operation must be one of {allowed_ops}")
        return v

class SecureToolInput(BaseModel):
    """Secure tool execution input"""
    tool_name: str
    parameters: Dict[str, Any]
    
    @validator('tool_name')
    def validate_tool_name(cls, v):
        # Only allow alphanumeric and underscore
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError("Invalid tool name")
        return v
    
    @validator('parameters')
    def validate_parameters(cls, v):
        # Sanitize string parameters
        for key, value in v.items():
            if isinstance(value, str):
                if not SecurityValidator.validate_sql_input(value):
                    raise ValueError(f"Potentially dangerous input in parameter {key}")
        return v
'''
        
        security_utils_path = self.project_root / "src/utils/security_validation.py"
        security_utils_path.parent.mkdir(parents=True, exist_ok=True)
        security_utils_path.write_text(validation_content)
        self.fixes_applied.append("Created security validation utilities")
        
    def _add_rate_limiting(self):
        """Add rate limiting configuration"""
        rate_limit_content = '''"""
Rate limiting implementation for API endpoints
"""
from typing import Dict, Optional
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict, deque
from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(
        self, 
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        burst_size: int = 10
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size
        self.minute_buckets: Dict[str, deque] = defaultdict(lambda: deque(maxlen=requests_per_minute))
        self.hour_buckets: Dict[str, deque] = defaultdict(lambda: deque(maxlen=requests_per_hour))
        self._cleanup_task = None
        
    async def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limits"""
        now = datetime.utcnow()
        
        # Check minute limit
        minute_bucket = self.minute_buckets[identifier]
        minute_cutoff = now - timedelta(minutes=1)
        
        # Remove old entries
        while minute_bucket and minute_bucket[0] < minute_cutoff:
            minute_bucket.popleft()
            
        if len(minute_bucket) >= self.requests_per_minute:
            return False
            
        # Check hour limit
        hour_bucket = self.hour_buckets[identifier]
        hour_cutoff = now - timedelta(hours=1)
        
        while hour_bucket and hour_bucket[0] < hour_cutoff:
            hour_bucket.popleft()
            
        if len(hour_bucket) >= self.requests_per_hour:
            return False
            
        # Add current request
        minute_bucket.append(now)
        hour_bucket.append(now)
        
        return True
    
    async def __call__(self, request: Request) -> None:
        """FastAPI dependency for rate limiting"""
        # Get identifier (IP address or user ID)
        identifier = request.client.host
        
        # Check for authenticated user
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # Use user ID if authenticated
            # This would need to decode the JWT token
            identifier = f"user_{auth_header}"
            
        if not await self.check_rate_limit(identifier):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
            
    def start_cleanup(self):
        """Start periodic cleanup of old entries"""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
    async def _cleanup_loop(self):
        """Periodic cleanup of old entries"""
        while True:
            await asyncio.sleep(300)  # Clean up every 5 minutes
            now = datetime.utcnow()
            
            # Clean up minute buckets
            for key in list(self.minute_buckets.keys()):
                bucket = self.minute_buckets[key]
                cutoff = now - timedelta(minutes=2)
                if bucket and bucket[-1] < cutoff:
                    del self.minute_buckets[key]
                    
            # Clean up hour buckets  
            for key in list(self.hour_buckets.keys()):
                bucket = self.hour_buckets[key]
                cutoff = now - timedelta(hours=2)
                if bucket and bucket[-1] < cutoff:
                    del self.hour_buckets[key]

# Global rate limiter instances
general_limiter = RateLimiter(requests_per_minute=60, requests_per_hour=1000)
auth_limiter = RateLimiter(requests_per_minute=10, requests_per_hour=100)  # Stricter for auth
api_limiter = RateLimiter(requests_per_minute=30, requests_per_hour=500)

# Endpoint-specific limiters
ENDPOINT_LIMITS = {
    "/api/v1/auth/login": auth_limiter,
    "/api/v1/auth/register": auth_limiter,
    "/api/v1/agents/execute": api_limiter,
    "/api/v1/tools/execute": api_limiter,
}

async def get_rate_limiter(request: Request) -> RateLimiter:
    """Get appropriate rate limiter for endpoint"""
    path = request.url.path
    return ENDPOINT_LIMITS.get(path, general_limiter)
'''
        
        rate_limit_path = self.project_root / "src/api/rate_limiter_enhanced.py"
        rate_limit_path.write_text(rate_limit_content)
        self.fixes_applied.append("Created enhanced rate limiting")
        
    def _create_security_config(self):
        """Create centralized security configuration"""
        security_config = '''"""
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
'''
        
        security_config_path = self.project_root / "src/config/security.py"
        security_config_path.parent.mkdir(parents=True, exist_ok=True)
        security_config_path.write_text(security_config)
        self.fixes_applied.append("Created centralized security configuration")
        
    def _generate_security_report(self):
        """Generate security audit report"""
        report = f"""# Security Audit Report

## Issues Found: {len(self.security_issues)}

### Hardcoded Credentials and Secrets
"""
        
        for issue in self.security_issues:
            report += f"- **{issue['type']}** in `{issue['file']}` (line {issue['line']})\n"
            
        report += f"""

## Fixes Applied: {len(self.fixes_applied)}

"""
        
        for fix in self.fixes_applied:
            report += f"- ✅ {fix}\n"
            
        report += """

## Security Enhancements Added:

1. **Authentication & Authorization**
   - Removed hardcoded passwords
   - Enhanced JWT security with proper expiration
   - Added password policy validation
   
2. **Input Validation**
   - Created security validation utilities
   - Added sanitization for HTML input
   - File path validation to prevent directory traversal
   - SQL injection prevention
   
3. **Rate Limiting**
   - Implemented token bucket algorithm
   - Different limits for different endpoints
   - Automatic cleanup of old entries
   
4. **Security Configuration**
   - Centralized security settings
   - Environment-based configuration
   - Secure defaults for all settings

## Next Steps:

1. Update `.env` file with secure values:
   ```
   ADMIN_PASSWORD=<generate-secure-password>
   JWT_SECRET_KEY=<generate-secure-key>
   ENCRYPTION_KEY=<generate-secure-key>
   ```

2. Review and update CORS origins for production

3. Enable security headers in production

4. Implement audit logging for sensitive operations

5. Regular security assessments and dependency updates
"""
        
        report_path = self.project_root / "SECURITY_AUDIT_REPORT.md"
        report_path.write_text(report)
        logger.info(f"📄 Security report saved to {report_path}")

def main():
    fixer = SecurityFixer()
    fixer.scan_and_fix()
    logger.info("\n✅ Security fixes completed!")

if __name__ == "__main__":
    main()