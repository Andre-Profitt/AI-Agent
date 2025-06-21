#!/usr/bin/env python3
from typing import List
# TODO: Fix undefined variables: List, Path, class_match, content, e, f, file, file_path, files_to_check, files_to_fix, fixes_applied, found_files, helper_method, init_end, insert_pos, issue, issues, logging, original_content, os, patterns, re, replacement, replacements
import pattern

# TODO: Fix undefined variables: class_match, content, e, f, file, file_path, files_to_check, files_to_fix, fixes_applied, found_files, helper_method, init_end, insert_pos, issue, issues, original_content, pattern, patterns, replacement, replacements

"""
Script to fix all is_configured() calls to use is_configured_safe()
"""

import os
import re
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

def find_files_with_config_checks() -> List[Path]:
    """Find all files that use is_configured()"""
    files_to_check = []

    # Search patterns
    patterns = [
        r'\.is_configured\(\)',
        r'config\.supabase\.',
        r'self\.config\.'
    ]

    # Search in src directory
    for file_path in Path('src').rglob('*.py'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                for pattern in patterns:
                    if re.search(pattern, content):
                        files_to_check.append(file_path)
                        break
        except Exception as e:
            logger.info("Error reading {}: {}", extra={"file_path": file_path, "e": e})

    return list(set(files_to_check))  # Remove duplicates

def fix_config_checks():
    """Fix all is_configured() calls to use is_configured_safe()"""

    # Known files that need updating
    files_to_fix = [
        "src/integration_hub.py",
        "src/services/integration_hub.py",
        "src/integration_manager.py",
        "src/config_cli.py",
        "src/database_enhanced.py",
        "src/health_check.py",
        "src/llamaindex_enhanced.py"
    ]

    # Add any additional files found
    found_files = find_files_with_config_checks()
    for file in found_files:
        if str(file) not in files_to_fix:
            files_to_fix.append(str(file))

    fixes_applied = 0

    for file_path in files_to_fix:
        if not os.path.exists(file_path):
            logger.info("⚠️  Skipping {} - not found", extra={"file_path": file_path})
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # Pattern replacements
            replacements = [
                # Simple is_configured() calls
                (r'if\s+self\.config\.supabase\.is_configured\(\):',
                 'if await self.config.supabase.is_configured_safe():'),

                (r'if\s+(\w+)\.supabase\.is_configured\(\):',
                 r'if await \1.supabase.is_configured_safe():'),

                # Direct config access - url
                (r'url\s*=\s*self\.config\.supabase\.url',
                 'url=await self._get_safe_config_value("supabase_url")'),

                # Direct config access - key
                (r'key\s*=\s*self\.config\.supabase\.key',
                 'key=await self._get_safe_config_value("supabase_key")'),

                # Other config access patterns
                (r'self\.config\.(\w+)\.(\w+)',
                 r'await self._get_safe_config_value("\1_\2")'),
            ]

            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)

            # Add import if needed
            if 'is_configured_safe' in content and 'from typing import' not in content:
                content = 'from typing import Optional, Dict, Any\n' + content

            # Add helper method if needed and not present
            if '_get_safe_config_value' in content and 'def _get_safe_config_value' not in content:
                # Find the class definition
                class_match = re.search(r'class\s+(\w+).*?:', content)
                if class_match:
                    # Find the end of __init__ or first method
                    init_end = re.search(r'def\s+__init__.*?\n(?=\s{0,4}def|\s{0,4}async\s+def|\s{0,4}@)',
                                       content, re.DOTALL)
                    if init_end:
                        insert_pos = init_end.end()
                    else:
                        insert_pos = class_match.end()

                    helper_method = '''
    async def _get_safe_config_value(self, key: str) -> str:
        """Safely get configuration value with error handling"""
        try:
            parts = key.split('_')
            if len(parts) == 2:
                service, attr = parts
                config_obj = getattr(self.config, service, None)
                if config_obj:
                    return getattr(config_obj, attr, "")

            # Direct attribute access
            return getattr(self.config, key, "")
        except Exception as e:
            logger.error("Config access failed", extra={"key": key, "error": str(e)})
            return ""
'''
                    content = content[:insert_pos] + helper_method + content[insert_pos:]

            # Only write if changes were made
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixes_applied += 1
                logger.info("✅ Fixed config checks in {}", extra={"file_path": file_path})
            else:
                logger.info("ℹ️  No changes needed in {}", extra={"file_path": file_path})

        except Exception as e:
            logger.info("❌ Error processing {}: {}", extra={"file_path": file_path, "e": e})

    logger.info("\n📊 Summary: Applied fixes to {} files", extra={"fixes_applied": fixes_applied})

def verify_fixes():
    """Verify that fixes were applied correctly"""
    logger.info("\n🔍 Verifying fixes...")

    issues = []

    for file_path in Path('src').rglob('*.py'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for remaining issues
            if 'is_configured()' in content and 'is_configured_safe' not in content:
                issues.append(f"{file_path}: Still has is_configured() calls")

            if re.search(r'=\s*self\.config\.\w+\.\w+(?!\()', content):
                issues.append(f"{file_path}: Still has direct config access")

        except Exception:
            pass

    if issues:
        logger.info("⚠️  Found remaining issues:")
        for issue in issues:
            logger.info("   - {}", extra={"issue": issue})
    else:
        logger.info("✅ All config checks appear to be fixed!")

if __name__ == "__main__":
    logger.info("🔧 Fixing configuration checks...")
    fix_config_checks()
    verify_fixes()
