#!/usr/bin/env python3
"""

from typing import Dict
from typing import Tuple
Fix orphaned methods with undefined 'self'
These are methods that were defined outside of their classes
"""

import re
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict

class OrphanedMethodFixer:
    def __init__(self, report_path: str):
        self.report_path = report_path
        self.orphaned_methods = {}
        self.files_to_fix = set()
        
    def load_report(self):
        """Load the error report and find all undefined 'self' errors"""
        with open(self.report_path, 'r') as f:
            data = json.load(f)
            
        # Find all undefined 'self' errors
        for error in data.get('by_severity', {}).get('error', []):
            if error['message'] == "Undefined variable: 'self'":
                file_path = error['file']
                line = error['line']
                
                if file_path not in self.orphaned_methods:
                    self.orphaned_methods[file_path] = []
                    
                self.orphaned_methods[file_path].append({
                    'line': line,
                    'snippet': error.get('snippet', '')
                })
                
                self.files_to_fix.add(file_path)
                
    def analyze_file(self, file_path: str) -> List[Dict]:
        """Analyze a file to find orphaned methods"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            orphaned = []
            in_class = False
            current_class = None
            class_indent = 0
            
            for i, line in enumerate(lines):
                # Check if we're entering a class
                class_match = re.match(r'^(\s*)class\s+(\w+)', line)
                if class_match:
                    in_class = True
                    current_class = class_match.group(2)
                    class_indent = len(class_match.group(1))
                    continue
                    
                # Check if we're leaving a class
                if in_class and line.strip() and not line.startswith(' ' * (class_indent + 1)):
                    in_class = False
                    current_class = None
                    
                # Check for method definition
                method_match = re.match(r'^(\s*)def\s+(\w+)\s*\(.*self.*\)', line)
                if method_match:
                    indent = len(method_match.group(1))
                    method_name = method_match.group(2)
                    
                    # If not in a class or wrong indentation, it's orphaned
                    if not in_class or indent <= class_indent:
                        orphaned.append({
                            'line': i + 1,
                            'method_name': method_name,
                            'indent': indent,
                            'nearest_class': self.find_nearest_class(lines, i)
                        })
                        
            return orphaned
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return []
            
    def find_nearest_class(self, lines: List[str], method_line: int) -> Tuple[str, int]:
        """Find the nearest class definition before a method"""
        for i in range(method_line - 1, -1, -1):
            class_match = re.match(r'^(\s*)class\s+(\w+)', lines[i])
            if class_match:
                return (class_match.group(2), i)
        return (None, -1)
        
    def fix_file(self, file_path: str) -> bool:
        """Fix orphaned methods in a file"""
        orphaned = self.analyze_file(file_path)
        if not orphaned:
            return True
            
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            # Group orphaned methods by their nearest class
            methods_by_class = {}
            for method in orphaned:
                class_name, class_line = method['nearest_class']
                if class_name:
                    if class_name not in methods_by_class:
                        methods_by_class[class_name] = []
                    methods_by_class[class_name].append(method)
                    
            # Fix by moving methods into their classes
            if methods_by_class:
                # Sort methods by line number in reverse order to avoid index issues
                all_methods = []
                for methods in methods_by_class.values():
                    all_methods.extend(methods)
                all_methods.sort(key=lambda x: x['line'], reverse=True)
                
                # Process each orphaned method
                for method in all_methods:
                    class_name, class_line = method['nearest_class']
                    if not class_name:
                        continue
                        
                    # Find the end of the class
                    class_end = self.find_class_end(lines, class_line)
                    if class_end == -1:
                        continue
                        
                    # Extract the method
                    method_start = method['line'] - 1
                    method_end = self.find_method_end(lines, method_start)
                    
                    if method_end > method_start:
                        # Get method lines
                        method_lines = lines[method_start:method_end]
                        
                        # Determine proper indentation
                        class_indent = len(lines[class_line]) - len(lines[class_line].lstrip())
                        method_indent = class_indent + 4  # Standard Python indentation
                        
                        # Re-indent method
                        reindented_method = []
                        for line in method_lines:
                            if line.strip():
                                # Calculate current indent
                                current_indent = len(line) - len(line.lstrip())
                                # Add class indent
                                new_line = ' ' * method_indent + line.lstrip()
                                reindented_method.append(new_line)
                            else:
                                reindented_method.append(line)
                                
                        # Remove method from original location
                        del lines[method_start:method_end]
                        
                        # Insert method into class (before class end)
                        insert_pos = class_end - (method_end - method_start)
                        lines[insert_pos:insert_pos] = reindented_method
                        
                        print(f"Moved {method['method_name']} into class {class_name}")
                        
            # Write back
            with open(file_path, 'w') as f:
                f.writelines(lines)
                
            return True
            
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
            return False
            
    def find_class_end(self, lines: List[str], class_line: int) -> int:
        """Find where a class definition ends"""
        class_indent = len(lines[class_line]) - len(lines[class_line].lstrip())
        
        for i in range(class_line + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Non-empty line
                line_indent = len(line) - len(line.lstrip())
                # If we find something at the same or lower indent, class has ended
                if line_indent <= class_indent:
                    return i
                    
        # Class goes to end of file
        return len(lines)
        
    def find_method_end(self, lines: List[str], method_line: int) -> int:
        """Find where a method definition ends"""
        method_indent = len(lines[method_line]) - len(lines[method_line].lstrip())
        
        for i in range(method_line + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Non-empty line
                line_indent = len(line) - len(line.lstrip())
                # If we find something at the same or lower indent, method has ended
                if line_indent <= method_indent:
                    return i
                    
        # Method goes to end of file
        return len(lines)
        
    def fix_all(self):
        """Fix all files with orphaned methods"""
        self.load_report()
        
        print(f"Found {len(self.files_to_fix)} files with potential orphaned methods")
        
        fixed = 0
        failed = 0
        
        # Focus on files with the most self errors first
        sorted_files = sorted(
            self.files_to_fix, 
            key=lambda f: len(self.orphaned_methods.get(f, [])), 
            reverse=True
        )
        
        for file_path in sorted_files[:10]:  # Start with top 10 files
            print(f"\nAnalyzing {file_path} ({len(self.orphaned_methods[file_path])} self errors)...")
            
            orphaned = self.analyze_file(file_path)
            if orphaned:
                print(f"Found {len(orphaned)} orphaned methods")
                if self.fix_file(file_path):
                    fixed += 1
                else:
                    failed += 1
            else:
                print("No orphaned methods found (might be other issues)")
                
        print(f"\nFixed {fixed} files, failed {failed}")

if __name__ == '__main__':
    fixer = OrphanedMethodFixer('final_report_after_all_fixes.json')
    fixer.fix_all()