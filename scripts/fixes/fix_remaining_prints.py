#!/usr/bin/env python3
"""
Fix remaining print statements in specific files
"""

import os
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Files that still have print statements
FILES_TO_FIX = [
    "src/core/optimized_chain_of_thought.py",
    "src/agents/advanced_hybrid_architecture.py",
    "src/utils/tools_introspection.py",
    "src/services/integration_hub_examples.py",
    "src/infrastructure/config/configuration_service.py",
    "src/core/services/working_memory.py"
]

def convert_print_to_logger(content: str, file_path: str) -> str:
    """Convert print statements to logger calls"""
    
    # Check if already has logging import
    has_logging = 'import logging' in content
    has_logger = 'logger = ' in content or 'logger=' in content
    
    # Add imports if needed
    if not has_logging:
        import_line = "import logging\n"
        # Find where to insert
        lines = content.split('\n')
        insert_idx = 0
        
        # Find the last import
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_idx = i + 1
            elif insert_idx > 0 and line and not line.startswith(('#', 'import', 'from')):
                break
        
        lines.insert(insert_idx, import_line)
        content = '\n'.join(lines)
    
    if not has_logger:
        # Add logger after imports
        logger_line = "logger = logging.getLogger(__name__)\n"
        lines = content.split('\n')
        insert_idx = 0
        
        # Find where to insert logger
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_idx = i + 1
            elif insert_idx > 0 and line and not line.startswith(('#', 'import', 'from', '"""', "'''")):
                break
        
        # Skip empty lines
        while insert_idx < len(lines) and not lines[insert_idx].strip():
            insert_idx += 1
        
        lines.insert(insert_idx, logger_line)
        content = '\n'.join(lines)
    
    # Convert print statements
    # Pattern 1: print("string")
    content = re.sub(
        r'print\s*\(\s*"([^"]+)"\s*\)',
        r'logger.info("\1")',
        content
    )
    
    # Pattern 2: print('string')
    content = re.sub(
        r"print\s*\(\s*'([^']+)'\s*\)",
        r'logger.info("\1")',
        content
    )
    
    # Pattern 3: print(f"string {var}")
    def convert_fstring_print(match):
        fstring_content = match.group(1)
        # Extract variables
        var_pattern = r'\{([^{}:]+)(?::[^}]+)?\}'
        variables = re.findall(var_pattern, fstring_content)
        
        # Replace {var} with {}
        message = re.sub(var_pattern, '{}', fstring_content)
        
        if variables:
            # Build extra dict
            extra_parts = []
            for var in variables:
                # Clean variable name for key
                key = var.strip().replace('.', '_').replace('[', '_').replace(']', '')
                key = re.sub(r'[^a-zA-Z0-9_]', '_', key)
                extra_parts.append(f'"{key}": {var}')
            
            extra_dict = "{" + ", ".join(extra_parts) + "}"
            return f'logger.info("{message}", extra={extra_dict})'
        else:
            return f'logger.info("{fstring_content}")'
    
    content = re.sub(
        r'print\s*\(\s*f"([^"]+)"\s*\)',
        convert_fstring_print,
        content
    )
    
    # Pattern 4: print(f'string {var}')  
    content = re.sub(
        r"print\s*\(\s*f'([^']+)'\s*\)",
        convert_fstring_print,
        content
    )
    
    # Pattern 5: print(variable)
    content = re.sub(
        r'print\s*\(\s*([a-zA-Z_]\w*)\s*\)',
        r'logger.info("Value: %s", \1)',
        content
    )
    
    # Pattern 6: print() - empty print
    content = re.sub(
        r'print\s*\(\s*\)',
        r'logger.info("")',
        content
    )
    
    # Pattern 7: print with multiple arguments
    content = re.sub(
        r'print\s*\(\s*"([^"]+)"\s*,\s*([^)]+)\s*\)',
        r'logger.info("\1: %s", \2)',
        content
    )
    
    return content

def fix_file(file_path: str) -> bool:
    """Fix prints in a single file"""
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count prints before
        prints_before = content.count('print(')
        
        if prints_before == 0:
            logger.info(f"No prints found in {file_path}")
            return False
        
        # Convert prints
        new_content = convert_print_to_logger(content, file_path)
        
        # Count prints after
        prints_after = new_content.count('print(')
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        fixed = prints_before - prints_after
        logger.info(f"✅ Fixed {fixed} print statements in {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return False

def main():
    """Fix remaining print statements"""
    logger.info("🔧 Fixing remaining print statements in specific files...")
    
    fixed_files = 0
    total_fixed = 0
    
    for file_path in FILES_TO_FIX:
        if fix_file(file_path):
            fixed_files += 1
    
    # Also search for any other files with prints in src/
    logger.info("\n🔍 Searching for additional files with print statements...")
    
    additional_files = []
    for root, dirs, files in os.walk('src'):
        # Skip __pycache__ directories
        if '__pycache__' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if file_path not in FILES_TO_FIX:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if 'print(' in content:
                            additional_files.append(file_path)
                    except:
                        pass
    
    if additional_files:
        logger.info(f"\nFound {len(additional_files)} additional files with print statements:")
        for file_path in additional_files[:10]:  # Show first 10
            logger.info(f"  - {file_path}")
        
        # Fix them too
        for file_path in additional_files:
            if fix_file(file_path):
                fixed_files += 1
    
    logger.info(f"\n✅ Summary: Fixed print statements in {fixed_files} files")
    
    # Final verification
    logger.info("\n🔍 Final verification...")
    remaining_prints = 0
    for root, dirs, files in os.walk('src'):
        if '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    count = content.count('print(')
                    if count > 0:
                        remaining_prints += count
                        logger.warning(f"Still has {count} prints: {file_path}")
                except:
                    pass
    
    if remaining_prints == 0:
        logger.info("✅ All print statements have been removed from src/!")
    else:
        logger.warning(f"⚠️  {remaining_prints} print statements still remain")

if __name__ == "__main__":
    main() 