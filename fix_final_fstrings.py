#!/usr/bin/env python3
"""
Fix remaining f-string logging issues
"""

import re
import os

# Files with f-string logging issues
FILES_TO_FIX = [
    "src/tools/registry.py",
    "src/application/agents/base_agent.py", 
    "src/infrastructure/agents/concrete_agents.py",
    "src/infrastructure/events/event_bus.py"
]

def fix_fstring_logging(file_path):
    """Fix f-string logging in a file"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern 1: logger.warning(f"string {var}", extra={...})
    content = re.sub(
        r'logger\.(\w+)\(f"([^"]+)"(.*?)\)',
        lambda m: convert_fstring_with_extra(m.group(1), m.group(2), m.group(3)),
        content
    )
    
    # Pattern 2: logger.info(f"string {var}")
    content = re.sub(
        r'logger\.(\w+)\(f"([^"]+)"\)',
        lambda m: convert_simple_fstring(m.group(1), m.group(2)),
        content
    )
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Fixed f-string logging in {file_path}")
        return True
    
    return False

def convert_fstring_with_extra(level, fstring_content, extra_part):
    """Convert f-string with extra to structured format"""
    # Extract variables from f-string
    var_pattern = r'\{([^{}:!]+)(?:[^}]*)?\}'
    variables = re.findall(var_pattern, fstring_content)
    
    # Replace {var} with {}
    message = re.sub(var_pattern, '{}', fstring_content)
    
    if variables:
        # Create argument list
        var_list = ', '.join(variables)
        return f'logger.{level}("{message}", {var_list}{extra_part})'
    else:
        return f'logger.{level}("{fstring_content}"{extra_part})'

def convert_simple_fstring(level, fstring_content):
    """Convert simple f-string to structured format"""
    # Extract variables from f-string
    var_pattern = r'\{([^{}:!]+)(?:[^}]*)?\}'
    variables = re.findall(var_pattern, fstring_content)
    
    # Replace {var} with {}
    message = re.sub(var_pattern, '{}', fstring_content)
    
    if variables:
        # Create argument list
        var_list = ', '.join(variables)
        return f'logger.{level}("{message}", {var_list})'
    else:
        return f'logger.{level}("{fstring_content}")'

def main():
    print("🔧 Fixing remaining f-string logging...")
    
    fixed_count = 0
    for file_path in FILES_TO_FIX:
        if fix_fstring_logging(file_path):
            fixed_count += 1
    
    print(f"\n✅ Fixed f-string logging in {fixed_count} files")
    
    # Final check
    remaining = []
    for file_path in FILES_TO_FIX:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if 'logger.*f"' in content or 'f"' in content:
                remaining.append(file_path)
    
    if remaining:
        print(f"⚠️  {len(remaining)} files still have f-string logging")
    else:
        print("✅ All f-string logging has been fixed!")

if __name__ == "__main__":
    main() 