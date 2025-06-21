#!/usr/bin/env python3
from typing import List
# TODO: Fix undefined variables: List, Path, Tuple, content, count, e, extra_dict, extra_parts, f, file_path, files_modified, fixes_in_file, format_spec, fstring_content, issue, key, level, logging, match, message, new_content, original_content, re, remaining_issues, replacement, specific_patterns, src_files, total_fixed, var_name, variables
import pattern

# TODO: Fix undefined variables: content, count, e, extra_dict, extra_parts, f, file_path, files_modified, fixes_in_file, format_spec, fstring_content, issue, key, level, match, message, new_content, original_content, pattern, remaining_issues, replacement, specific_patterns, src_files, total_fixed, var_name, variables

"""
Script to convert all f-string logging to structured logging
"""

from typing import Tuple

import re
from pathlib import Path

import logging

logger = logging.getLogger(__name__)

def extract_fstring_variables(self, fstring_content: str) -> List[Tuple[str, str]]:
    """Extract variables and their format specs from f-string"""
    variables = []

    # Pattern to match {variable} or {variable:format} or {variable.attribute}
    pattern = r'\{([^{}:]+)(?::([^}]+))?\}'

    for match in re.finditer(pattern, fstring_content):
        var_name = match.group(1).strip()
        format_spec = match.group(2) if match.group(2) else None
        variables.append((var_name, format_spec))

    return variables

def convert_fstring_to_structured(self, match) -> str:
    """Convert f-string log call to structured logging"""
    level = match.group(1)
    quote_char = match.group(2)
    fstring_content = match.group(3)

    # Extract variables from f-string
    variables = extract_fstring_variables(fstring_content)

    if not variables:
        # No variables, just return as regular string
        return f'logger.{level}("{fstring_content}")'

    # Replace variable placeholders with {}
    message = fstring_content
    for var_name, format_spec in variables:
        if format_spec:
            pattern = r'\{' + re.escape(var_name) + r':' + re.escape(format_spec) + r'\}'
        else:
            pattern = r'\{' + re.escape(var_name) + r'\}'
        message = re.sub(pattern, '{}', message)

    # Build extra dict
    extra_parts = []
    for var_name, format_spec in variables:
        # Clean up variable name for key (remove dots, brackets)
        key = var_name.replace('.', '_').replace('[', '_').replace(']', '').replace('()', '')
        key = re.sub(r'[^a-zA-Z0-9_]', '_', key)

        # Handle format specifications
        if format_spec:
            if format_spec.startswith('.') and format_spec[1:].isdigit():
                # Float precision, e.g., {value:.2f}
                extra_parts.append(f'"{key}": round({var_name}, {format_spec[1:]})')
            else:
                # Other format specs, just use the variable
                extra_parts.append(f'"{key}": {var_name}')
        else:
            extra_parts.append(f'"{key}": {var_name}')

    extra_dict = "{" + ", ".join(extra_parts) + "}"

    return f'logger.{level}("{message}", extra={extra_dict})'

def fix_fstring_logging():
    """Convert all f-string logging to structured logging"""

    # Find all Python files
    src_files = list(Path('src').rglob('*.py'))

    total_fixed = 0
    files_modified = 0

    for file_path in src_files:
        if 'structured_logging.py' in str(file_path):
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            fixes_in_file = 0

            # Pattern to match logger.{level}(f"...") or logger.{level}(f'...')
            # Captures: (level)(quote)(content)
            pattern = r'logger\.(\w+)\(f(["\'])((?:[^\2\\]|\\.)*?)\2\)'

            def replacement_func(self, match):
                nonlocal fixes_in_file
                fixes_in_file += 1
                return convert_fstring_to_structured(match)

            content = re.sub(pattern, replacement_func, content)

            # Also fix common specific patterns
            specific_patterns = [
                # logger.info(f"Some {var} text")
                (r'logger\.info\(f"Circuit breaker closed for \{tool_name\}"\)',
                 'logger.info("Circuit breaker closed", extra={"tool_name": tool_name})'),

                (r'logger\.info\(f"Set rate limit for \{tool_name\}: \{calls_per_minute\} calls/minute"\)',
                 'logger.info("Set rate limit", extra={"tool_name": tool_name, "calls_per_minute": calls_per_minute})'),

                (r'logger\.error\(f"([^"]+)\{(\w+)\}([^"]+)"\)',
                 r'logger.error("\1{}\3", extra={"\2": \2})'),

                # String concatenation patterns
                (r'logger\.(\w+)\("([^"]+)" \+ str\((\w+)\)\)',
                 r'logger.\1("\2", extra={"value": \3})'),

                (r'logger\.(\w+)\("([^"]+)" \+ (\w+)\)',
                 r'logger.\1("\2", extra={"value": \3})'),
            ]

            for pattern, replacement in specific_patterns:
                new_content = re.sub(pattern, replacement, content)
                if new_content != content:
                    fixes_in_file += 1
                    content = new_content

            # Write back if changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                files_modified += 1
                total_fixed += fixes_in_file
                logger.info("✅ Fixed {} f-string logs in {}", extra={"fixes_in_file": fixes_in_file, "file_path": file_path})

        except Exception as e:
            logger.info("❌ Error processing {}: {}", extra={"file_path": file_path, "e": e})

    logger.info("\n📊 Summary: Fixed {} f-string logs across {} files", extra={"total_fixed": total_fixed, "files_modified": files_modified})

def verify_fstring_fixes():
    """Verify no f-string logging remains"""
    logger.info("\n🔍 Verifying f-string logging fixes...")

    remaining_issues = []

    for file_path in Path('src').rglob('*.py'):
        if 'structured_logging.py' in str(file_path):
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for remaining f-string logs
            if re.search(r'logger\.\w+\(f["\']', content):
                # Count occurrences
                count = len(re.findall(r'logger\.\w+\(f["\']', content))
                remaining_issues.append(f"{file_path}: {count} f-string log(s) remaining")

        except Exception:
            pass

    if remaining_issues:
        logger.info("⚠️  Found remaining f-string logs:")
        for issue in remaining_issues:
            logger.info("   - {}", extra={"issue": issue})
    else:
        logger.info("✅ All f-string logging has been fixed!")

if __name__ == "__main__":
    logger.info("📝 Fixing f-string logging...")
    fix_fstring_logging()
    verify_fstring_fixes()
