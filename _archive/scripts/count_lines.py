#!/usr/bin/env python3
"""

from typing import Tuple
Count Lines of Code Script for AI Agent Project
This script counts total lines, code lines, comment lines, and blank lines
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import re

class CodeCounter:
    def __init__(self):
        self.stats = {
            'total_lines': 0,
            'code_lines': 0,
            'comment_lines': 0,
            'blank_lines': 0,
            'files_counted': 0,
            'by_directory': {},
            'by_extension': {},
            'largest_files': []
        }
        
    def count_file(self, file_path: Path) -> Tuple[int, int, int, int]:
        """Count lines in a single file"""
        total = 0
        code = 0
        comments = 0
        blank = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                in_multiline_comment = False
                
                for line in f:
                    total += 1
                    line = line.strip()
                    
                    # Check for multiline comments
                    if '"""' in line or "'''" in line:
                        in_multiline_comment = not in_multiline_comment
                        comments += 1
                    elif in_multiline_comment:
                        comments += 1
                    elif not line:
                        blank += 1
                    elif line.startswith('#'):
                        comments += 1
                    else:
                        code += 1
                        
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
        return total, code, comments, blank
    
    def analyze_project(self, root_dir: str = '.') -> None:
        """Analyze all Python files in the project"""
        root_path = Path(root_dir)
        
        # Directories to exclude
        exclude_dirs = {
            '__pycache__', '.git', '.venv', 'venv', 'env',
            'build', 'dist', '.pytest_cache', '.mypy_cache',
            'htmlcov', '.tox', 'node_modules'
        }
        
        # Find all Python files
        python_files = []
        for file_path in root_path.rglob('*.py'):
            # Skip excluded directories
            if any(excluded in file_path.parts for excluded in exclude_dirs):
                continue
            python_files.append(file_path)
        
        # Count lines in each file
        file_sizes = []
        
        for file_path in python_files:
            total, code, comments, blank = self.count_file(file_path)
            
            # Update global stats
            self.stats['total_lines'] += total
            self.stats['code_lines'] += code
            self.stats['comment_lines'] += comments
            self.stats['blank_lines'] += blank
            self.stats['files_counted'] += 1
            
            # Track by directory
            dir_name = file_path.parent.name if file_path.parent != root_path else 'root'
            if dir_name not in self.stats['by_directory']:
                self.stats['by_directory'][dir_name] = {
                    'files': 0, 'total': 0, 'code': 0
                }
            self.stats['by_directory'][dir_name]['files'] += 1
            self.stats['by_directory'][dir_name]['total'] += total
            self.stats['by_directory'][dir_name]['code'] += code
            
            # Track by extension (for future use with other file types)
            ext = file_path.suffix
            if ext not in self.stats['by_extension']:
                self.stats['by_extension'][ext] = {
                    'files': 0, 'total': 0, 'code': 0
                }
            self.stats['by_extension'][ext]['files'] += 1
            self.stats['by_extension'][ext]['total'] += total
            self.stats['by_extension'][ext]['code'] += code
            
            # Track largest files
            file_sizes.append((file_path, total, code))
        
        # Sort and keep top 10 largest files
        file_sizes.sort(key=lambda x: x[1], reverse=True)
        self.stats['largest_files'] = [
            {'path': str(f[0]), 'total_lines': f[1], 'code_lines': f[2]}
            for f in file_sizes[:10]
        ]
    
    def print_report(self) -> None:
        """Print a formatted report"""
        print("\n" + "="*60)
        print("📊 AI AGENT PROJECT - LINES OF CODE REPORT")
        print("="*60)
        
        # Overall statistics
        print("\n📈 OVERALL STATISTICS:")
        print(f"  Total Files Analyzed: {self.stats['files_counted']:,}")
        print(f"  Total Lines:         {self.stats['total_lines']:,}")
        print(f"  Code Lines:          {self.stats['code_lines']:,} ({self.stats['code_lines']/max(1, self.stats['total_lines'])*100:.1f}%)")
        print(f"  Comment Lines:       {self.stats['comment_lines']:,} ({self.stats['comment_lines']/max(1, self.stats['total_lines'])*100:.1f}%)")
        print(f"  Blank Lines:         {self.stats['blank_lines']:,} ({self.stats['blank_lines']/max(1, self.stats['total_lines'])*100:.1f}%)")
        
        # By directory
        print("\n📁 BY DIRECTORY (Top 10):")
        sorted_dirs = sorted(
            self.stats['by_directory'].items(),
            key=lambda x: x[1]['total'],
            reverse=True
        )[:10]
        
        for dir_name, stats in sorted_dirs:
            print(f"  {dir_name:20} {stats['files']:4} files, {stats['total']:6,} lines ({stats['code']:6,} code)")
        
        # Largest files
        print("\n📄 LARGEST FILES:")
        for i, file_info in enumerate(self.stats['largest_files'], 1):
            path = Path(file_info['path'])
            # Show relative path from project root
            try:
                rel_path = path.relative_to('.')
            except:
                rel_path = path
            print(f"  {i:2}. {str(rel_path):50} {file_info['total_lines']:6,} lines ({file_info['code_lines']:5,} code)")
        
        # Summary
        print("\n📊 SUMMARY:")
        if self.stats['code_lines'] > 0:
            if self.stats['code_lines'] < 1000:
                size = "Small"
            elif self.stats['code_lines'] < 10000:
                size = "Medium"
            elif self.stats['code_lines'] < 50000:
                size = "Large"
            else:
                size = "Very Large"
            
            print(f"  This is a {size} Python project with {self.stats['code_lines']:,} lines of actual code.")
            print(f"  Code-to-comment ratio: {self.stats['code_lines']/max(1, self.stats['comment_lines']):.1f}:1")
            print(f"  Average file size: {self.stats['total_lines']/max(1, self.stats['files_counted']):.0f} lines")
        
        print("\n" + "="*60)

def main():
    """Main function"""
    counter = CodeCounter()
    
    # Analyze the project
    print("🔍 Analyzing AI Agent project...")
    counter.analyze_project('.')
    
    # Print the report
    counter.print_report()
    
    # Also save to file
    report_path = 'lines_of_code_report.txt'
    with open(report_path, 'w') as f:
        # Redirect print to file
        import sys
        old_stdout = sys.stdout
        sys.stdout = f
        counter.print_report()
        sys.stdout = old_stdout
    
    print(f"\n💾 Report saved to: {report_path}")

if __name__ == "__main__":
    main() 