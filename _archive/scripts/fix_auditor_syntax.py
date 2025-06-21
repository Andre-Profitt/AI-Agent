#!/usr/bin/env python3
"""
Fix syntax errors in comprehensive_code_auditor.py
"""

import re
from pathlib import Path

def fix_auditor_syntax():
    """Fix the syntax errors in comprehensive_code_auditor.py"""
    
    file_path = Path("comprehensive_code_auditor.py")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the AST visitor class
    lines = content.splitlines()
    
    # Fix the structural issues
    fixed_lines = []
    i = 0
    in_visitor_class = False
    indent_level = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Detect class ASTVisitor
        if 'class ASTVisitor' in line:
            in_visitor_class = True
            indent_level = len(line) - len(line.lstrip())
        
        # Fix line 176-178 issue
        if i >= 175 and i <= 178:
            if 'old_async = self.in_async' in line:
                # This should be inside a method
                fixed_lines.append(line)
            elif 'self.in_async = True' in line:
                # This should be inside a method
                fixed_lines.append(line)
            elif '"summary":' in line:
                # This is definitely wrong, skip it
                i += 1
                continue
        # Fix indentation issues in methods
        elif in_visitor_class and line.strip().startswith('def '):
            # Method definition - ensure proper indentation
            method_indent = indent_level + 4
            fixed_lines.append(' ' * method_indent + line.strip())
        elif in_visitor_class and line.strip() and not line.startswith(' '):
            # Line with no indentation in class - fix it
            if 'return' in line or 'if' in line or 'for' in line:
                fixed_lines.append(' ' * (indent_level + 8) + line.strip())
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
        
        i += 1
    
    # Join and do additional fixes
    content = '\n'.join(fixed_lines)
    
    # Fix specific patterns
    # Remove the orphaned code block
    content = re.sub(
        r'old_async = self\.in_async\s*\n\s*self\.in_async = True\s*\n\s*"summary": \{',
        '',
        content
    )
    
    # Fix method indentations
    content = re.sub(
        r'\n(\s*)def _get_line\(self',
        r'\n    def _get_line(self',
        content
    )
    
    content = re.sub(
        r'\n(\s*)def _add_issue\(self',
        r'\n    def _add_issue(self',
        content
    )
    
    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed syntax errors in comprehensive_code_auditor.py")

if __name__ == "__main__":
    fix_auditor_syntax()