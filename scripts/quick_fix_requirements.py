#!/usr/bin/env python3
"""
Quick Fix for Common Requirements Issues
This script automatically fixes common version problems in requirements.txt
"""

import subprocess
import sys
import re
import logging

logger = logging.getLogger(__name__)


# Known problematic packages and their fixes
KNOWN_FIXES = {
    "llama-index-readers-file==0.5.0": "llama-index-readers-file==0.4.9",
    "duckduckgo-search==3.9.0": "duckduckgo-search==3.9.3",
    "langchain-core==0.1.0": "langchain-core==0.3.60",
    "langchain-openai==0.0.5": "langchain-openai==0.3.20",
    "langchain-community==0.0.10": "langchain-community==0.3.20",
    "langchain-experimental==0.0.49": "langchain-experimental==0.3.3",
    "langchain-groq==0.0.1": "langchain-groq==0.3.0",
    "langchain-tavily==0.0.1": "langchain-tavily==0.3.0",
    "llama-index-embeddings-openai==0.1.0": "llama-index-embeddings-openai==0.4.1",
    "llama-index-vector-stores-supabase==0.1.0": "llama-index-vector-stores-supabase==0.4.0",
}

def get_latest_version(package_name):
    """Get the latest version of a package from PyPI."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "index", "versions", package_name],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            # Parse the output to find available versions
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'Available versions:' in line:
                    versions = line.split(':', 1)[1].strip().split(', ')
                    # Filter out pre-release versions
                    stable_versions = [v for v in versions if not any(x in v for x in ['a', 'b', 'rc', 'dev'])]
                    if stable_versions:
                        return stable_versions[0]
                    elif versions:
                        return versions[0]
        return None
    except Exception:
        return None

def fix_requirements_file(filename):
    """Fix common issues in requirements.txt file."""
    logger.info("Fixing requirements in {}...", extra={"filename": filename})
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logger.info("Error: File '{}' not found", extra={"filename": filename})
        return False
    
    # Backup original file
    backup_filename = filename + '.backup'
    with open(backup_filename, 'w') as f:
        f.writelines(lines)
    logger.info("Created backup: {}", extra={"backup_filename": backup_filename})
    
    fixed_lines = []
    changes_made = 0
    
    for line in lines:
        original_line = line
        line_stripped = line.strip()
        
        # Skip comments and empty lines
        if not line_stripped or line_stripped.startswith('#'):
            fixed_lines.append(original_line)
            continue
        
        # Check if line matches any known problematic package
        fixed = False
        for problem, fix in KNOWN_FIXES.items():
            if line_stripped == problem:
                fixed_lines.append(fix + '\n')
                logger.info("  Fixed: {} → {}", extra={"problem": problem, "fix": fix})
                changes_made += 1
                fixed = True
                break
        
        if not fixed:
            # Check for LangChain 0.1.x to 0.3.x migration
            if 'langchain' in line_stripped and '==0.0.' in line_stripped:
                package_match = re.match(r'^(langchain[a-zA-Z0-9-_]*)==0\.0\.\d+', line_stripped)
                if package_match:
                    package_name = package_match.group(1)
                    # Try to get latest version
                    latest = get_latest_version(package_name)
                    if latest and latest.startswith('0.3'):
                        new_line = f"{package_name}=={latest}\n"
                        fixed_lines.append(new_line)
                        logger.info("  Updated: {} → {}=={}", extra={"line_stripped": line_stripped, "package_name": package_name, "latest": latest})
                        changes_made += 1
                        fixed = True
            
            # Check for llama-index version consistency
            elif line_stripped.startswith('llama-index=='):
                fixed_lines.append('llama-index==0.12.42\n')
                if line_stripped != 'llama-index==0.12.42':
                    logger.info("  Updated: {} → llama-index==0.12.42", extra={"line_stripped": line_stripped})
                    changes_made += 1
                fixed = True
            elif line_stripped.startswith('llama-index-core=='):
                fixed_lines.append('llama-index-core==0.12.42\n')
                if line_stripped != 'llama-index-core==0.12.42':
                    logger.info("  Updated: {} → llama-index-core==0.12.42", extra={"line_stripped": line_stripped})
                    changes_made += 1
                fixed = True
        
        if not fixed:
            fixed_lines.append(original_line)
    
    # Write fixed file
    with open(filename, 'w') as f:
        f.writelines(fixed_lines)
    
    logger.info("\nTotal changes made: {}", extra={"changes_made": changes_made})
    
    if changes_made > 0:
        logger.info("\n✅ Fixed {} successfully!", extra={"filename": filename})
        logger.info("Backup saved as {}", extra={"backup_filename": backup_filename})
        return True
    else:
        logger.info("\n✅ No issues found in requirements.txt")
        return True

def main():
    """Main function."""
    filename = 'requirements.txt'
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    logger.info("Requirements Quick Fix Tool")
    logger.info("==========================")
    logger.info("This tool fixes common version issues in requirements.txt\n")
    
    if fix_requirements_file(filename):
        logger.info("\nNext steps:")
        logger.info("1. Review the changes")
        logger.info("2. Test installation: pip install -r requirements.txt")
        logger.info("3. If issues persist, use the verify script to check all versions")

if __name__ == "__main__":
    main() 