#!/usr/bin/env python3
from typing import List
# TODO: Fix undefined variables: List, Path, all_files_with_prints, content, count, dir_path, e, extra_dict, extra_parts, f, file_path, files, files_to_process, files_with_prints, fixed, fstring_content, has_logger, has_logging_import, i, import_line, insert_idx, issue, key, line, lines, logger_line, logging, m, message, new_content, os, part, patterns, pf, prints_after, prints_before, priority_files, production_dirs, re, remaining_prints, replacement, test_prints, total_fixed, var, var_pattern, variables
import pattern

# TODO: Fix undefined variables: all_files_with_prints, content, count, dir_path, e, extra_dict, extra_parts, f, file_path, files, files_to_process, files_with_prints, fixed, fstring_content, has_logger, has_logging_import, i, import_line, insert_idx, issue, key, line, lines, logger_line, m, message, new_content, part, pattern, patterns, pf, prints_after, prints_before, priority_files, production_dirs, remaining_prints, replacement, test_prints, total_fixed, var, var_pattern, variables

"""
Script to remove all print statements and replace with proper logging
"""

import logging
logger = logging.getLogger(__name__)

import os
import re

from pathlib import Path

def get_files_with_prints() -> List[Path]:
    """Find all files containing print statements"""
    files_with_prints = []

    # Check all Python files
    for file_path in Path('.').rglob('*.py'):
        # Skip virtual environments and build directories
        if any(part in str(file_path) for part in ['venv', 'env', '.env', 'build', 'dist', '__pycache__']):
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Quick check for print statements
            if 'print(' in content:
                files_with_prints.append(file_path)

        except Exception as e:
            logger.info("Error reading {}: {}", extra={"file_path": file_path, "e": e})

    return files_with_prints

def convert_print_to_logger(self, content: str) -> str:
    """Convert print statements to logger calls"""

    # Add imports if not present
    has_logging_import = 'import logging' in content
    has_logger = 'logger = ' in content or 'logger=' in content

    if not has_logging_import:
        # Add at the top after other imports or at beginning
        import_line = "import logging\n"

        # Find the right place to insert
        lines = content.split('\n')
        insert_idx = 0

        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_idx = i + 1
            elif line and not line.startswith('#') and insert_idx > 0:
                break

        lines.insert(insert_idx, import_line)
        content = '\n'.join(lines)

    if not has_logger:
        # Add logger after imports
        logger_line = "logger = logging.getLogger(__name__)\n"

        lines = content.split('\n')
        insert_idx = 0

        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_idx = i + 1
            elif line and not line.startswith('#') and not line.startswith('import') and insert_idx > 0:
                break

        lines.insert(insert_idx + 1, logger_line)
        content = '\n'.join(lines)

    # Convert different print patterns
    patterns = [
        # logger.info("simple string")
        (r'print\s*\(\s*"([^"]+)"\s*\)',
         r'logger.info("\1")'),

        # logger.info("simple string")
        (r"print\s*\(\s*'([^']+)'\s*\)",
         r'logger.info("\1")'),

        # logger.info("string with {}", extra={"variable": variable})
        (r'print\s*\(\s*f"([^"]+)"\s*\)',
         lambda m: convert_fstring_print(m.group(1))),

        # logger.info("string with {}", extra={"variable": variable})
        (r"print\s*\(\s*f'([^']+)'\s*\)",
         lambda m: convert_fstring_print(m.group(1))),

        # logger.info("Value", extra={"value": variable})
        (r'print\s*\(\s*([a-zA-Z_]\w*)\s*\)',
         r'logger.info("Value", extra={"value": \1})'),

        # logger.info("string", extra={"data": variable})
        (r'print\s*\(\s*"([^"]+)"\s*,\s*([^)]+)\s*\)',
         r'logger.info("\1", extra={"data": \2})'),

        # logger.info("") - empty print
        (r'print\s*\(\s*\)',
         r'logger.info("")'),
    ]

    for pattern, replacement in patterns:
        if callable(replacement):
            content = re.sub(pattern, replacement, content)
        else:
            content = re.sub(pattern, replacement, content)

    return content

def convert_fstring_print(self, fstring_content: str) -> str:
    """Convert f-string print to structured logger call"""
    # Extract variables from f-string
    var_pattern = r'\{([^{}:]+)(?::[^}]+)?\}'
    variables = re.findall(var_pattern, fstring_content)

    # Replace {var} with {} in message
    message = re.sub(var_pattern, '{}', fstring_content)

    if variables:
        # Build extra dict
        extra_parts = []
        for var in variables:
            # Clean variable name for key
            key = var.replace('.', '_').replace('[', '_').replace(']', '')
            key = re.sub(r'[^a-zA-Z0-9_]', '_', key)
            extra_parts.append(f'"{key}": {var}')

        extra_dict = "{" + ", ".join(extra_parts) + "}"
        return f'logger.info("{message}", extra={extra_dict})'
    else:
        return f'logger.info("{fstring_content}")'

def remove_print_statements():
    """Remove all print statements and replace with logging"""

    # Priority files that definitely need fixing
    priority_files = [
        "scripts/setup_supabase.py",
        "tests/test_integration_fixes.py",
        "tests/gaia_testing_framework.py",
        "src/integration_hub_examples.py",
        "ai_codebase_analyzer.py",
        "demo_hybrid_architecture.py"
    ]

    # Find all files with prints
    all_files_with_prints = get_files_with_prints()

    # Combine lists, prioritizing known files
    files_to_process = []
    for pf in priority_files:
        if Path(pf).exists():
            files_to_process.append(Path(pf))

    for f in all_files_with_prints:
        if f not in files_to_process:
            files_to_process.append(f)

    total_fixed = 0

    for file_path in files_to_process:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Count prints before
            prints_before = content.count('print(')

            # Skip if no prints
            if prints_before == 0:
                continue

            # Convert prints to logging
            new_content = convert_print_to_logger(content)

            # Count prints after
            prints_after = new_content.count('print(')

            # Write back if changed
            if prints_after < prints_before:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

                fixed = prints_before - prints_after
                total_fixed += fixed
                logger.info("✅ Removed {} print statements from {}", extra={"fixed": fixed, "file_path": file_path})
            else:
                logger.info("⚠️  Could not fix all prints in {}", extra={"file_path": file_path})

        except Exception as e:
            logger.info("❌ Error processing {}: {}", extra={"file_path": file_path, "e": e})

    logger.info("\n📊 Summary: Removed {} print statements total", extra={"total_fixed": total_fixed})

def verify_print_removal():
    """Verify no print statements remain in production code"""
    logger.info("\n🔍 Verifying print statement removal...")

    production_dirs = ['src', 'app.py']
    remaining_prints = []

    for dir_path in production_dirs:
        if os.path.isfile(dir_path):
            files = [Path(dir_path)]
        else:
            files = Path(dir_path).rglob('*.py')

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Count print statements
                count = content.count('print(')
                if count > 0:
                    remaining_prints.append(f"{file_path}: {count} print statement(s)")

            except Exception:
                pass

    if remaining_prints:
        logger.info("⚠️  Found remaining print statements in production code:")
        for issue in remaining_prints:
            logger.info("   - {}", extra={"issue": issue})
    else:
        logger.info("✅ No print statements found in production code!")

    # Also check test files
    test_prints = []
    for file_path in Path('tests').rglob('*.py'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            count = content.count('print(')
            if count > 0:
                test_prints.append(f"{file_path}: {count} print statement(s)")
        except Exception:
            pass

    if test_prints:
        logger.info("\nℹ️  Test files with prints (consider fixing): {} files", extra={"len_test_prints_": len(test_prints)})

if __name__ == "__main__":
    logger.info("🖨️  Removing print statements...")
    remove_print_statements()
    verify_print_removal()
