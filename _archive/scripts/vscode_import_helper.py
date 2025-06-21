#!/usr/bin/env python3
"""
VS Code Import Helper
Generates import suggestions for undefined variables
"""

import ast
import os
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

class ProjectImportMapper:
    """Maps all exportable names in the project to their import paths"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root).resolve()
        self.exports_map = defaultdict(list)  # name -> [(module, type)]
        
    def scan_project(self):
        """Scan all Python files and build export map"""
        print("🔍 Scanning project for exportable names...")
        
        for py_file in self.project_root.rglob("*.py"):
            # Skip test files and scripts
            if any(part in str(py_file) for part in ['test_', '__pycache__', '.venv', 'venv']):
                continue
                
            self._scan_file(py_file)
        
        print(f"✅ Found {len(self.exports_map)} unique exportable names")
        
    def _scan_file(self, file_path: Path):
        """Scan a single file for exports"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Get module path relative to project root
            try:
                rel_path = file_path.relative_to(self.project_root)
                module_path = str(rel_path).replace('/', '.').replace('\\', '.')[:-3]
                
                # Skip if it's a script in root
                if '.' not in module_path and module_path != '__init__':
                    return
                    
            except ValueError:
                return
            
            # Extract all definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self.exports_map[node.name].append((module_path, 'class'))
                elif isinstance(node, ast.FunctionDef):
                    if not node.name.startswith('_'):
                        self.exports_map[node.name].append((module_path, 'function'))
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and not target.id.startswith('_'):
                            self.exports_map[target.id].append((module_path, 'variable'))
                            
        except Exception:
            pass  # Skip files that can't be parsed
    
    def generate_import_suggestions(self, undefined_vars: List[str]) -> Dict[str, List[str]]:
        """Generate import suggestions for undefined variables"""
        suggestions = {}
        
        for var in undefined_vars:
            if var in self.exports_map:
                imports = []
                for module, var_type in self.exports_map[var]:
                    if module.startswith('src.'):
                        imports.append(f"from {module} import {var}")
                    elif '.' in module:
                        imports.append(f"from {module} import {var}")
                
                if imports:
                    suggestions[var] = imports[:3]  # Top 3 suggestions
        
        return suggestions
    
    def generate_vscode_snippets(self):
        """Generate VS Code snippets for common imports"""
        snippets = {}
        
        # Group by module
        module_exports = defaultdict(list)
        for name, locations in self.exports_map.items():
            for module, _ in locations:
                if module.startswith('src.'):
                    module_exports[module].append(name)
        
        # Create snippets
        for module, names in module_exports.items():
            if len(names) > 2:  # Only for modules with multiple exports
                snippet_name = f"Import from {module}"
                snippets[snippet_name] = {
                    "prefix": f"from {module.split('.')[-1]}",
                    "body": [
                        f"from {module} import (",
                        "\t${1|" + ",".join(sorted(names)) + "|}",
                        ")"
                    ],
                    "description": f"Import from {module}"
                }
        
        return snippets

def analyze_undefined_variables(report_file: str) -> List[str]:
    """Extract undefined variables from audit report"""
    with open(report_file, 'r') as f:
        report = json.load(f)
    
    undefined = set()
    for issue in report.get('all_issues', []):
        if issue.get('category') == 'undefined_variable':
            # Extract variable name from message
            msg = issue.get('message', '')
            if "Undefined variable: '" in msg:
                var = msg.split("'")[1]
                undefined.add(var)
    
    return sorted(undefined)

def generate_quick_fix_imports(project_root: str, report_file: str):
    """Generate quick fix imports for VS Code"""
    # Load undefined variables
    undefined_vars = analyze_undefined_variables(report_file)
    print(f"📊 Found {len(undefined_vars)} undefined variables")
    
    # Build export map
    mapper = ProjectImportMapper(project_root)
    mapper.scan_project()
    
    # Generate suggestions
    suggestions = mapper.generate_import_suggestions(undefined_vars)
    
    # Create quick fix file
    quick_fixes = []
    for var, imports in suggestions.items():
        quick_fixes.append({
            "variable": var,
            "suggested_imports": imports,
            "preferred": imports[0] if imports else None
        })
    
    # Save suggestions
    with open('.vscode/import_suggestions.json', 'w') as f:
        json.dump({
            "suggestions": quick_fixes,
            "total_undefined": len(undefined_vars),
            "total_with_suggestions": len(suggestions)
        }, f, indent=2)
    
    print(f"✅ Generated import suggestions for {len(suggestions)} variables")
    print("📁 Saved to .vscode/import_suggestions.json")
    
    # Generate VS Code snippets
    snippets = mapper.generate_vscode_snippets()
    
    # Save snippets
    snippets_file = Path('.vscode/python.code-snippets')
    with open(snippets_file, 'w') as f:
        json.dump(snippets, f, indent=2)
    
    print(f"✅ Generated {len(snippets)} VS Code snippets")
    print("📁 Saved to .vscode/python.code-snippets")
    
    # Print top undefined variables without suggestions
    no_suggestions = [var for var in undefined_vars if var not in suggestions]
    if no_suggestions:
        print(f"\n⚠️  {len(no_suggestions)} variables need manual import resolution:")
        for var in no_suggestions[:20]:
            print(f"  • {var}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VS Code import helper")
    parser.add_argument("--report", "-r", default="final_report_complete.json",
                       help="Audit report file")
    parser.add_argument("--project", "-p", default=".",
                       help="Project root directory")
    
    args = parser.parse_args()
    
    generate_quick_fix_imports(args.project, args.report)
    
    print("\n💡 VS Code Tips:")
    print("  1. Restart VS Code to pick up new settings")
    print("  2. Use Ctrl+Space for auto-completion")
    print("  3. Use Ctrl+. on undefined variables for quick fixes")
    print("  4. Use Ctrl+Shift+O to organize imports")
    print("  5. Check .vscode/import_suggestions.json for manual fixes")

if __name__ == "__main__":
    main()