"""
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
            r"((DELETE|DROP|EXEC(UTE)?|INSERT|SELECT|UNION|UPDATE))",
            r"(--|#|/\*|\*/)",
            r"(OR\s*\d+\s*=\s*\d+)",
            r"(AND\s*\d+\s*=\s*\d+)"
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        return True
    
    @staticmethod
    def validate_command_input(command: str) -> bool:
        """Validate shell command input"""
        dangerous_chars = [';', '|', '&', '$', '`', '\n', '\r']
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
