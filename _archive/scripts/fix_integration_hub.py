#!/usr/bin/env python3
"""
Fix Integration Hub to add circuit breaker protection
"""

import os
import re
import logging
# TODO: Fix undefined variables: change, changes_made, class_match, content, description, e, extra, extra_parts, f, file, file_path, files, fixes, fstring_content, helper_methods, i, import_pos, imports, init_match, insert_pos, issue, issues, key, level, line, lines, logging, m, message, method_matches, new_content, original_content, os, path, possible_paths, re, replacement, root, var, var_pattern, variables
import pattern



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def find_integration_hub() -> str:
    """Find the integration hub file"""
    possible_paths = [
        "src/services/integration_hub.py",
        "src/integration_hub.py",
        "services/integration_hub.py",
        "integration_hub.py"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Search for it
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file == 'integration_hub.py':
                return os.path.join(root, file)
    
    return None

def fix_integration_hub():
    """Fix all issues in integration_hub.py"""
    
    file_path = find_integration_hub()
    
    if not file_path:
        logger.info("❌ Could not find integration_hub.py")
        logger.info("   Searched in: src/services/, src/, services/")
        return False
    
    logger.info("📄 Found integration_hub.py at: {}", extra={"file_path": file_path})
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = []
        
        # Add imports if needed
        if 'from src.infrastructure.resilience.circuit_breaker import' not in content:
            imports = """from src.infrastructure.resilience.circuit_breaker import (
    circuit_breaker, CircuitBreakerConfig, CircuitBreakerOpenError
)
from src.utils.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)
"""
            # Add after other imports
            import_pos = 0
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    import_pos = i + 1
                elif import_pos > 0 and line and not line.startswith(('import', 'from')):
                    break
            
            lines.insert(import_pos, imports)
            content = '\n'.join(lines)
            changes_made.append("Added circuit breaker imports")
        
        # Fix specific patterns
        fixes = [
            # Fix is_configured() calls
            (r'if\s+self\.config\.supabase\.is_configured\(\):',
             'if await self._check_config_safe():',
             "Updated is_configured() to use safe method"),
            
            # Fix direct config access - url
            (r'url\s*=\s*self\.config\.supabase\.url(?!["\'])',
             'url=await self._get_config_value("url")',
             "Protected URL config access"),
            
            # Fix direct config access - key
            (r'key\s*=\s*self\.config\.supabase\.key(?!["\'])',
             'key=await self._get_config_value("key")',
             "Protected key config access"),
            
            # Fix f-string logging
            (r'logger\.info\(f"([^"]+)"\)',
             lambda m: convert_fstring_log(m.group(1), 'info'),
             "Converted f-string logging to structured"),
            
            (r'logger\.error\(f"([^"]+)"\)',
             lambda m: convert_fstring_log(m.group(1), 'error'),
             "Converted f-string error logging"),
        ]
        
        for pattern, replacement, description in fixes:
            if callable(replacement):
                new_content = re.sub(pattern, replacement, content)
            else:
                new_content = re.sub(pattern, replacement, content)
            
            if new_content != content:
                content = new_content
                changes_made.append(description)
        
        # Add helper methods if not present
        if '_check_config_safe' not in content and changes_made:
            # Find the class definition
            class_match = re.search(r'class\s+\w+.*?:', content)
            if class_match:
                # Find a good insertion point (before the last method or at end of class)
                method_matches = list(re.finditer(r'\n    (async )?def \w+\(', content))
                if method_matches:
                    insert_pos = method_matches[-1].start()
                else:
                    # Find end of __init__ method
                    init_match = re.search(r'def __init__.*?\n(?=\n    (async )?def|\n\n|\Z)', 
                                         content, re.DOTALL)
                    if init_match:
                        insert_pos = init_match.end()
                    else:
                        insert_pos = class_match.end()
                
                helper_methods = '''
    @circuit_breaker("integration_hub_config", CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30,
        success_threshold=2
    ))
    async def _check_config_safe(self) -> bool:
        """Safely check configuration with circuit breaker"""
        try:
            if hasattr(self.config.supabase, 'is_configured_safe'):
                return await self.config.supabase.is_configured_safe()
            else:
                # Fallback for older config
                return bool(self.config.supabase.url and self.config.supabase.key)
        except Exception as e:
            logger.error("Configuration check failed", extra={
                "component": "integration_hub",)
                "error": str(e),
                "error_type": type(e).__name__
            })
            return False
    
    @circuit_breaker("integration_hub_db_init", CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=60,
        success_threshold=1
    ))
    async def _initialize_database_safe(self) -> Optional[Dict[str, Any]]:
        """Initialize database with full circuit breaker protection"""
        try:
            # Get config values safely
            url = await self._get_config_value("url")
            key = await self._get_config_value("key")
            
            if not url or not key:
                logger.error("Invalid database configuration")
                return None
            
            # Initialize with protection
            from src.database_enhanced import initialize_supabase_enhanced
            components = await initialize_supabase_enhanced(url=url, key=key)
            
            logger.info("Database initialized successfully", extra={
                "component": "integration_hub",
                "status": "connected")
            })
            
            return components
            
        except Exception as e:
            logger.error("Database initialization failed", extra={
                "component": "integration_hub",)
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise
    
    async def _get_config_value(self, key: str) -> str:
        """Get configuration value safely"""
        try:
            if key == "url":
                return self.config.supabase.url
            elif key == "key":
                return self.config.supabase.key
            elif key == "service_key":
                return getattr(self.config.supabase, 'service_key', '')
            else:
                return getattr(self.config.supabase, key, '')
        except Exception as e:
            logger.error("Failed to get config value", extra={
                "key": key,)
                "error": str(e)
            })
            return ""
'''
                content = content[:insert_pos] + helper_methods + content[insert_pos:]
                changes_made.append("Added safe helper methods")
        
        # Add missing imports for types
        if 'Optional[Dict[str, Any]]' in content and 'from typing import' not in content:
            content = 'from typing import Optional, Dict, Any\n' + content
            changes_made.append("Added type imports")
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("✅ Fixed integration_hub.py")
            logger.info("   Changes made:")
            for change in changes_made:
                logger.info("   - {}", extra={"change": change})
            return True
        else:
            logger.info("ℹ️  No changes needed in integration_hub.py")
            return True
            
    except Exception as e:
        logger.info("❌ Error processing integration_hub.py: {}", extra={"e": e})
        return False

def convert_fstring_log(fstring_content: str, level: str) -> str:
    """Convert f-string log to structured format"""
    # Extract variables
    var_pattern = r'\{([^}:]+)(?::[^}]+)?\}'
    variables = re.findall(var_pattern, fstring_content)
    
    # Replace variables with {}
    message = re.sub(var_pattern, '{}', fstring_content)
    
    if variables:
        extra_parts = []
        for var in variables:
            # Clean variable name
            key = var.strip().replace('.', '_').replace('[', '_').replace(']', '')
            key = re.sub(r'[^a-zA-Z0-9_]', '_', key)
            extra_parts.append(f'"{key}": {var}')
        
        extra = "{" + ", ".join(extra_parts) + "}"
        return f'logger.{level}("{message}", extra={extra})'
    else:
        return f'logger.{level}("{fstring_content}")'

def verify_fix():
    """Verify the fix was applied correctly"""
    file_path = find_integration_hub()
    if not file_path:
        return
    
    logger.info("\n🔍 Verifying fix...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    issues = []
    
    # Check for remaining issues
    if 'is_configured():' in content and '_check_config_safe' not in content:
        issues.append("Still has unprotected is_configured() calls")
    
    if re.search(r'=\s*self\.config\.supabase\.(url|key)(?!["\'])', content):
        issues.append("Still has direct config access")
    
    if 'logger.info(f"' in content or 'logger.error(f"' in content:
        issues.append("Still has f-string logging")
    
    if '@circuit_breaker' not in content:
        issues.append("No circuit breaker decorators found")
    
    if issues:
        logger.info("⚠️  Issues remaining:")
        for issue in issues:
            logger.info("   - {}", extra={"issue": issue})
    else:
        logger.info("✅ All issues appear to be fixed!")

if __name__ == "__main__":
    logger.info("🔧 Fixing Integration Hub...")
    if fix_integration_hub():
        verify_fix() 