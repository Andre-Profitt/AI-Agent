#!/usr/bin/env python3
"""
Fix the final 3 files with f-string logging
"""

import os
import re

# Files that still have f-string logging
TARGET_FILES = [
    "src/application/agents/base_agent.py",
    "src/infrastructure/agents/concrete_agents.py", 
    "src/infrastructure/events/event_bus.py"
]

def fix_fstring_logging(file_path: str):
    """Fix f-string logging in a file"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        fixes_made = 0
        
        # Pattern 1: logger.method(f"text {var}")
        def replace_fstring(match):
            nonlocal fixes_made
            level = match.group(1)
            fstring_content = match.group(2)
            
            # Extract variables from f-string
            var_pattern = r'\{([^{}:!]+)(?:[^}]*)?\}'
            variables = re.findall(var_pattern, fstring_content)
            
            # Replace {var} with %s
            message = re.sub(var_pattern, '%s', fstring_content)
            
            fixes_made += 1
            
            if variables:
                var_list = ', '.join(variables)
                return f'logger.{level}("{message}", {var_list})'
            else:
                return f'logger.{level}("{fstring_content}")'
        
        # Apply main pattern
        content = re.sub(
            r'logger\.(\w+)\(f"([^"]+)"\)',
            replace_fstring,
            content
        )
        
        # Pattern 2: logger.method(f"text", extra={...}) - fix just the f-string part
        def replace_fstring_with_extra(match):
            nonlocal fixes_made
            level = match.group(1)
            fstring_content = match.group(2)
            extra_part = match.group(3)
            
            # Extract variables from f-string
            var_pattern = r'\{([^{}:!]+)(?:[^}]*)?\}'
            variables = re.findall(var_pattern, fstring_content)
            
            # Replace {var} with %s
            message = re.sub(var_pattern, '%s', fstring_content)
            
            fixes_made += 1
            
            if variables:
                var_list = ', '.join(variables)
                return f'logger.{level}("{message}", {var_list}, {extra_part})'
            else:
                return f'logger.{level}("{fstring_content}", {extra_part})'
        
        # Apply pattern with extra
        content = re.sub(
            r'logger\.(\w+)\(f"([^"]+)"(,\s*extra=\{[^}]+\})\)',
            replace_fstring_with_extra,
            content
        )
        
        # Pattern 3: Multi-line f-strings
        def replace_multiline_fstring(match):
            nonlocal fixes_made
            level = match.group(1)
            quote_type = match.group(2)
            fstring_content = match.group(3)
            
            # Extract variables
            var_pattern = r'\{([^{}:!]+)(?:[^}]*)?\}'
            variables = re.findall(var_pattern, fstring_content)
            
            # Replace {var} with %s
            message = re.sub(var_pattern, '%s', fstring_content)
            
            fixes_made += 1
            
            if variables:
                var_list = ', '.join(variables)
                return f'logger.{level}({quote_type}{message}{quote_type}, {var_list})'
            else:
                return f'logger.{level}({quote_type}{fstring_content}{quote_type})'
        
        # Apply multi-line pattern
        content = re.sub(
            r'logger\.(\w+)\(f("""|\'\'\')([^"\']+)\2\)',
            replace_multiline_fstring,
            content,
            flags=re.DOTALL
        )
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed {fixes_made} f-string log calls in {file_path}")
            return True
        else:
            # Try line-by-line approach for complex cases
            lines = content.split('\n')
            modified = False
            
            for i, line in enumerate(lines):
                if 'logger.' in line and 'f"' in line:
                    # Simple line-by-line replacement
                    new_line = line
                    
                    # Extract the logger call
                    match = re.search(r'logger\.(\w+)\(f"([^"]+)"', line)
                    if match:
                        level = match.group(1)
                        fstring = match.group(2)
                        
                        # Extract variables
                        variables = re.findall(r'\{([^}]+)\}', fstring)
                        message = re.sub(r'\{[^}]+\}', '%s', fstring)
                        
                        if variables:
                            var_list = ', '.join(variables)
                            old_part = f'logger.{level}(f"{fstring}"'
                            new_part = f'logger.{level}("{message}", {var_list}'
                            new_line = line.replace(old_part, new_part)
                        else:
                            old_part = f'f"{fstring}"'
                            new_part = f'"{fstring}"'
                            new_line = line.replace(old_part, new_part)
                        
                        if new_line != line:
                            lines[i] = new_line
                            modified = True
                            fixes_made += 1
            
            if modified:
                content = '\n'.join(lines)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ Fixed {fixes_made} f-string log calls in {file_path} (line-by-line)")
                return True
            else:
                print(f"ℹ️  No f-string logging found in {file_path}")
                return False
                
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def main():
    print("🔧 Fixing Final F-String Logging Issues")
    print("=" * 60)
    
    fixed_count = 0
    
    for file_path in TARGET_FILES:
        print(f"\n📄 Processing: {file_path}")
        if fix_fstring_logging(file_path):
            fixed_count += 1
    
    print(f"\n✅ Fixed {fixed_count} out of {len(TARGET_FILES)} files")
    
    # Verify no f-string logging remains
    print("\n🔍 Verifying...")
    remaining = []
    
    for root, dirs, files in os.walk('src'):
        if '__pycache__' in root:
            continue
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Check for f-string logging
                    if re.search(r'logger\.\w+\(f["\"]', content):
                        remaining.append(file_path)
                except:
                    pass
    
    if not remaining:
        print("\n🎉 SUCCESS! No f-string logging remains in src/")
    else:
        print(f"\n⚠️  {len(remaining)} files still have f-string logging:")
        for f in remaining:
            print(f"   - {f}")

if __name__ == "__main__":
    main() 