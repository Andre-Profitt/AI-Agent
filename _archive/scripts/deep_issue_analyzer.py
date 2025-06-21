#!/usr/bin/env python3
from src.tools.base_tool import Tool

from src.agents.advanced_agent_fsm import Agent
from collections import Counter
from typing import List
# TODO: Fix undefined variables: Dict, List, Path, analysis, analyzer, args, class_name, content, contexts, count, critical_issues, ctx, defaultdict, f, file_issues, file_path, fix_script, fixes, guesses, i, import_map, imports, inst, instances, issue, issue_count, issue_types, json, local_variables, match, module_name, module_path, node, parser, part, possible_imports, project_classes, project_root, py_file, re, report, report_file, special_cases, stdlib_missing, sys, term, third_party_missing, tree, undefined_contexts, undefined_vars, var, var_name, worst_files, x
from prometheus_client import Counter
import pattern

# TODO: Fix undefined variables: analysis, analyzer, argparse, args, ast, class_name, content, contexts, count, critical_issues, ctx, f, file_issues, file_path, fix_script, fixes, guesses, i, import_map, imports, inst, instances, issue, issue_count, issue_types, local_variables, match, module_name, module_path, node, parser, part, pattern, possible_imports, project_classes, project_root, py_file, report, report_file, self, special_cases, stdlib_missing, term, third_party_missing, tree, undefined_contexts, undefined_vars, var, var_name, worst_files, x

"""
Deep Issue Analyzer and Targeted Fixer
Analyzes patterns in remaining issues and creates specific fixes
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
import ast

class DeepIssueAnalyzer:
    """Analyze and fix specific patterns in code issues"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.issue_patterns = defaultdict(list)
        self.undefined_patterns = defaultdict(set)
        self.critical_patterns = defaultdict(list)
        
    def analyze_report(self, report_file: str = "final_report.json"):
        """Deep analysis of audit report"""
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        print("🔍 DEEP ISSUE ANALYSIS")
        print("=" * 60)
        
        # 1. Analyze Critical Issues
        print("\n🚨 CRITICAL ISSUES (Must Fix First):")
        critical_issues = report.get('by_severity', {}).get('critical', [])
        
        for issue in critical_issues:
            self.critical_patterns[issue['message']].append({
                'file': issue['file'],
                'line': issue['line'],
                'snippet': issue.get('snippet', '')
            })
        
        # Group and display critical issues
        for i, (pattern, instances) in enumerate(self.critical_patterns.items(), 1):
            print(f"\n{i}. {pattern} ({len(instances)} instances)")
            for inst in instances[:2]:  # Show first 2 examples
                print(f"   📁 {Path(inst['file']).name}:{inst['line']}")
                if inst['snippet']:
                    print(f"      Code: {inst['snippet'][:60]}...")
        
        # 2. Analyze Undefined Variables
        print("\n\n📊 UNDEFINED VARIABLE ANALYSIS:")
        undefined_vars = Counter()
        undefined_contexts = defaultdict(list)
        
        for issue in report.get('all_issues', []):
            if issue['category'] == 'undefined_variable':
                match = re.search(r"Undefined variable: '(\w+)'", issue['message'])
                if match:
                    var_name = match.group(1)
                    undefined_vars[var_name] += 1
                    undefined_contexts[var_name].append({
                        'file': issue['file'],
                        'line': issue['line'],
                        'snippet': issue.get('snippet', '')
                    })
        
        # Categorize undefined variables
        print("\n🔍 Undefined Variable Categories:")
        
        # Category 1: Likely missing imports from project modules
        project_classes = []
        stdlib_missing = []
        third_party_missing = []
        local_variables = []
        special_cases = []
        
        for var, count in undefined_vars.most_common():
            if var[0].isupper() and not var.isupper():  # ClassNames
                # Check if it's likely a project class
                if any(term in var for term in ['Agent', 'Tool', 'Memory', 'FSM', 'GAIA']):
                    project_classes.append((var, count))
                else:
                    third_party_missing.append((var, count))
            elif var in ['self', 'cls']:
                special_cases.append((var, count))
            elif var in ['__name__', '__file__', '__all__']:
                special_cases.append((var, count))
            elif var.islower() and count > 10:
                # Likely a missing common variable
                local_variables.append((var, count))
            else:
                # Check contexts to categorize better
                contexts = undefined_contexts[var]
                if any('import' in ctx['snippet'] for ctx in contexts[:3]):
                    stdlib_missing.append((var, count))
                else:
                    local_variables.append((var, count))
        
        print("\n🏢 Project Classes (need correct imports):")
        for var, count in project_classes[:10]:
            print(f"  {var}: {count} occurrences")
            # Try to guess the import
            possible_imports = self._guess_project_import(var)
            if possible_imports:
                print(f"    💡 Try: {possible_imports[0]}")
        
        print("\n📦 Missing Standard Library:")
        for var, count in sorted(stdlib_missing, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {var}: {count} occurrences")
        
        print("\n🔧 Special Cases:")
        for var, count in special_cases:
            print(f"  {var}: {count} occurrences")
            if var == 'self':
                print("    💡 Missing 'self' in method definition")
            elif var == 'cls':
                print("    💡 Missing 'cls' in classmethod definition")
        
        # 3. File-specific analysis
        print("\n\n📁 FILES WITH MOST ISSUES:")
        worst_files = report.get('worst_files', [])[:5]
        
        for file_path, issue_count in worst_files:
            print(f"\n{Path(file_path).name}: {issue_count} issues")
            # Analyze specific issues in this file
            file_issues = [i for i in report['all_issues'] if i['file'] == file_path]
            issue_types = Counter(i['category'] for i in file_issues)
            print(f"  Issue breakdown: {dict(issue_types.most_common(3))}")
        
        return {
            'critical_patterns': dict(self.critical_patterns),
            'undefined_vars': dict(undefined_vars.most_common(50)),
            'project_classes': project_classes[:20],
            'worst_files': worst_files[:10]
        }
    
    def _guess_project_import(self, class_name: str) -> List[str]:
        """Guess the import path for a project class"""
        guesses = []
        
        # Common patterns in your project
        if 'Agent' in class_name:
            guesses.append(f"from src.agents.advanced_agent_fsm import {class_name}")
            guesses.append(f"from src.agents.enhanced_fsm import {class_name}")
        elif 'Tool' in class_name:
            guesses.append(f"from src.tools.base_tool import {class_name}")
            guesses.append(f"from src.utils.tools_enhanced import {class_name}")
        elif 'Memory' in class_name:
            guesses.append(f"from src.gaia_components.enhanced_memory_system import {class_name}")
        elif 'FSM' in class_name:
            guesses.append(f"from src.agents.enhanced_fsm import {class_name}")
        elif 'GAIA' in class_name:
            guesses.append(f"from src.gaia_components.advanced_reasoning_engine import {class_name}")
        
        return guesses
    
    def generate_targeted_fixes(self, analysis: Dict) -> str:
        """Generate specific fix script based on analysis"""
        fixes = []
        
        fixes.append("#!/usr/bin/env python3")
        fixes.append('"""Targeted fixes for remaining issues"""')
        fixes.append("import re")
        fixes.append("import os")
        fixes.append("from pathlib import Path")
        fixes.append("")
        
        # Fix 1: Critical syntax errors
        fixes.append("# Fix 1: Critical Syntax Errors")
        fixes.append("critical_fixes = {")
        
        for pattern, instances in analysis['critical_patterns'].items():
            if 'parenthes' in pattern.lower() or 'bracket' in pattern.lower():
                for inst in instances:
                    fixes.append(f"    '{inst['file']}': {{")
                    fixes.append(f"        'line': {inst['line']},")
                    fixes.append(f"        'fix': 'add_closing_paren'")
                    fixes.append("    },")
        
        fixes.append("}")
        fixes.append("")
        
        # Fix 2: Project class imports
        fixes.append("# Fix 2: Project Class Imports")
        fixes.append("project_imports = {")
        
        for class_name, count in analysis['project_classes']:
            imports = self._guess_project_import(class_name)
            if imports:
                fixes.append(f"    '{class_name}': '{imports[0]}',")
        
        fixes.append("}")
        fixes.append("")
        
        # Main fix function
        fixes.append("def apply_targeted_fixes():")
        fixes.append("    fixes_applied = 0")
        fixes.append("    ")
        fixes.append("    # Apply critical fixes")
        fixes.append("    for file_path, fix_info in critical_fixes.items():")
        fixes.append("        try:")
        fixes.append("            with open(file_path, 'r') as f:")
        fixes.append("                lines = f.readlines()")
        fixes.append("            ")
        fixes.append("            line_idx = fix_info['line'] - 1")
        fixes.append("            if 0 <= line_idx < len(lines):")
        fixes.append("                line = lines[line_idx]")
        fixes.append("                if fix_info['fix'] == 'add_closing_paren':")
        fixes.append("                    # Add closing parenthesis if missing")
        fixes.append("                    open_count = line.count('(')")
        fixes.append("                    close_count = line.count(')')")
        fixes.append("                    if open_count > close_count:")
        fixes.append("                        lines[line_idx] = line.rstrip() + ')' * (open_count - close_count) + '\\n'")
        fixes.append("                        fixes_applied += 1")
        fixes.append("            ")
        fixes.append("            with open(file_path, 'w') as f:")
        fixes.append("                f.writelines(lines)")
        fixes.append("        except Exception as e:")
        fixes.append("            print(f'Error fixing {file_path}: {e}')")
        fixes.append("    ")
        fixes.append("    # Apply import fixes")
        fixes.append("    for file_path in Path('.').rglob('*.py'):")
        fixes.append("        try:")
        fixes.append("            with open(file_path, 'r') as f:")
        fixes.append("                content = f.read()")
        fixes.append("            ")
        fixes.append("            original = content")
        fixes.append("            lines = content.splitlines()")
        fixes.append("            ")
        fixes.append("            # Check for undefined project classes")
        fixes.append("            for class_name, import_stmt in project_imports.items():")
        fixes.append("                if class_name in content and import_stmt not in content:")
        fixes.append("                    # Add import after other imports")
        fixes.append("                    for i, line in enumerate(lines):")
        fixes.append("                        if line.strip() and not line.startswith(('import', 'from', '#')):")
        fixes.append("                            lines.insert(i, import_stmt)")
        fixes.append("                            lines.insert(i+1, '')")
        fixes.append("                            fixes_applied += 1")
        fixes.append("                            break")
        fixes.append("            ")
        fixes.append("            if lines != content.splitlines():")
        fixes.append("                with open(file_path, 'w') as f:")
        fixes.append("                    f.write('\\n'.join(lines))")
        fixes.append("        except Exception as e:")
        fixes.append("            pass")
        fixes.append("    ")
        fixes.append("    return fixes_applied")
        fixes.append("")
        fixes.append("if __name__ == '__main__':")
        fixes.append("    fixes = apply_targeted_fixes()")
        fixes.append("    print(f'Applied {fixes} targeted fixes')")
        
        return '\n'.join(fixes)
    
    def create_import_map(self) -> Dict[str, str]:
        """Create a comprehensive import map for the project"""
        import_map = {}
        
        print("\n🗺️ Building Project Import Map...")
        
        # Scan all Python files to find class definitions
        for py_file in self.project_root.rglob("*.py"):
            if any(part.startswith('.') for part in py_file.parts):
                continue  # Skip hidden directories
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to find classes and functions
                tree = ast.parse(content)
                
                # Get module path
                module_path = py_file.relative_to(self.project_root)
                module_name = str(module_path.with_suffix('')).replace('/', '.')
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        import_map[node.name] = f"from {module_name} import {node.name}"
                    elif isinstance(node, ast.FunctionDef) and node.name[0].isupper():
                        # Include factory functions that return classes
                        import_map[node.name] = f"from {module_name} import {node.name}"
                        
            except Exception:
                pass
        
        return import_map


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deep analysis of code issues")
    parser.add_argument("--report", "-r", default="final_report.json",
                       help="Audit report to analyze")
    parser.add_argument("--fix-script", "-f", action="store_true",
                       help="Generate targeted fix script")
    parser.add_argument("--import-map", "-i", action="store_true",
                       help="Build project import map")
    
    args = parser.parse_args()
    
    analyzer = DeepIssueAnalyzer()
    
    # Analyze report
    print("🔍 Analyzing audit report...")
    analysis = analyzer.analyze_report(args.report)
    
    # Build import map if requested
    if args.import_map:
        import_map = analyzer.create_import_map()
        print(f"\n📦 Found {len(import_map)} importable items")
        
        # Save import map
        with open('project_import_map.json', 'w') as f:
            json.dump(import_map, f, indent=2)
        print("💾 Saved to project_import_map.json")
    
    # Generate fix script if requested
    if args.fix_script:
        fix_script = analyzer.generate_targeted_fixes(analysis)
        
        with open('targeted_fixes.py', 'w') as f:
            f.write(fix_script)
        
        print("\n🔧 Generated targeted_fixes.py")
        print("   Run: python targeted_fixes.py")
    
    print("\n✨ Analysis complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main()) 