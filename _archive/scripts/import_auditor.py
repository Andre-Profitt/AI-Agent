#!/usr/bin/env python3
from functools import partial
# TODO: Fix undefined variables: Any, Dict, List, Optional, Path, Set, Tuple, alias, alt, args, auditor, content, count, current_module, cycle, cycle_start, d, dataclass, defaultdict, detailed, dirs, e, elt, error, error_type, exports, f, field, file, file_fixes, file_path, files, fix, fixes, fixes_by_file, from_module, i, ignore_patterns, import_graph, import_info, json, known_module, level, match, message, module, module_exports, module_lower, module_name, module_parts, name, names, names_str, neighbor, new_import, new_module, node, old_import, old_module, os, output_file, package_path, parser, partial, parts, path, project_name, project_root, python_files, re, rec_stack, relative_path, report, root, spec, suggestions, summary, sys, target, to_module, tree, visited
import pattern

# TODO: Fix undefined variables: alias, alt, argparse, args, ast, auditor, content, count, current_module, cycle, cycle_start, d, detailed, dirs, e, elt, error, error_type, exports, f, file, file_fixes, file_path, files, fix, fixes, fixes_by_file, fnmatch, from_module, i, ignore_patterns, import_graph, import_info, importlib, known_module, level, match, message, module, module_exports, module_lower, module_name, module_parts, name, names, names_str, neighbor, new_import, new_module, node, old_import, old_module, output_file, package_path, parser, parts, path, pattern, project_name, project_root, python_files, rec_stack, relative_path, report, root, self, spec, suggestions, summary, target, to_module, tree, visited

"""
Comprehensive Import Auditor for Python Projects
Finds all import errors and missing modules/classes/functions
"""

from dataclasses import field
from typing import List
from typing import Tuple
from typing import Set
from typing import Optional
from typing import Any

import os
import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import importlib.util
import json
from dataclasses import dataclass, field
import re

@dataclass
class ImportInfo:
    """Information about an import statement"""
    file_path: str
    line_number: int
    import_type: str  # 'import' or 'from'
    module: str
    names: List[str] = field(default_factory=list)  # What's being imported
    alias: Optional[str] = None

    def __str__(self):
        if self.import_type == 'import':
            return f"{self.module}" + (f" as {self.alias}" if self.alias else "")
        else:
            names_str = ", ".join(self.names)
            return f"from {self.module} import {names_str}"

@dataclass
class ImportError:
    """Represents an import error"""
    file_path: str
    line_number: int
    error_type: str  # 'module_not_found', 'attribute_not_found', 'circular_import'
    import_info: ImportInfo
    error_message: str
    suggested_fixes: List[str] = field(default_factory=list)

class ImportAuditor:
    """Comprehensive import auditor for Python projects"""

    def __init__(self, project_root: str, ignore_patterns: Optional[List[str]] = None):
        self.project_root = Path(project_root).resolve()
        self.ignore_patterns = ignore_patterns or [
            '__pycache__', '.git', '.venv', 'venv', 'env',
            'build', 'dist', '*.egg-info', '.pytest_cache'
        ]

        # Results storage
        self.all_imports: List[ImportInfo] = []
        self.import_errors: List[ImportError] = []
        self.module_exports: Dict[str, Set[str]] = {}  # module -> exported names
        self.file_modules: Dict[str, str] = {}  # file_path -> module_name
        self.circular_imports: List[Tuple[str, str]] = []

        # Add project root to sys.path for import checking
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

    def audit(self) -> Dict[str, Any]:
        """Run the complete audit"""
        print(f"🔍 Starting import audit of {self.project_root}")

        # Step 1: Discover all Python files
        python_files = self._discover_python_files()
        print(f"📁 Found {len(python_files)} Python files")

        # Step 2: Parse all files and extract imports
        for file_path in python_files:
            self._parse_file(file_path)
        print(f"📝 Found {len(self.all_imports)} import statements")

        # Step 3: Map all available modules and their exports
        self._map_module_exports(python_files)
        print(f"🗺️  Mapped {len(self.module_exports)} modules")

        # Step 4: Check all imports for errors
        self._check_all_imports()
        print(f"❌ Found {len(self.import_errors)} import errors")

        # Step 5: Detect circular imports
        self._detect_circular_imports()
        print(f"🔄 Found {len(self.circular_imports)} circular imports")

        # Step 6: Generate report
        report = self._generate_report()

        return report

    def _discover_python_files(self) -> List[Path]:
        """Discover all Python files in the project"""
        python_files = []

        for root, dirs, files in os.walk(self.project_root):
            # Remove ignored directories
            dirs[:] = [d for d in dirs if not any(
                self._matches_pattern(d, pattern) for pattern in self.ignore_patterns
            )]

            # Find Python files
            for file in files:
                if file.endswith('.py') and not any(
                    self._matches_pattern(file, pattern) for pattern in self.ignore_patterns
                ):
                    python_files.append(Path(root) / file)

        return sorted(python_files)

    def _matches_pattern(self, name: str, pattern: str) -> bool:
        """Check if name matches pattern (supports wildcards)"""
        if '*' in pattern:
            import fnmatch
            return fnmatch.fnmatch(name, pattern)
        return pattern in name

    def _parse_file(self, file_path: Path):
        """Parse a Python file and extract imports"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_info = ImportInfo(
                            file_path=str(file_path),
                            line_number=node.lineno,
                            import_type='import',
                            module=alias.name,
                            alias=alias.asname
                        )
                        self.all_imports.append(import_info)

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    # Handle relative imports
                    if node.level > 0:
                        module = '.' * node.level + module

                    names = [alias.name for alias in node.names]
                    import_info = ImportInfo(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        import_type='from',
                        module=module,
                        names=names
                    )
                    self.all_imports.append(import_info)

        except Exception as e:
            print(f"⚠️  Error parsing {file_path}: {e}")

    def _map_module_exports(self, python_files: List[Path]):
        """Map what each module exports"""
        for file_path in python_files:
            # Convert file path to module name
            try:
                relative_path = file_path.relative_to(self.project_root)
                module_parts = relative_path.with_suffix('').parts

                # Skip __init__.py in module name
                if module_parts[-1] == '__init__':
                    module_parts = module_parts[:-1]

                module_name = '.'.join(module_parts)
                self.file_modules[str(file_path)] = module_name

                # Parse file to find exports
                exports = self._get_module_exports(file_path)
                if module_name:
                    self.module_exports[module_name] = exports

            except Exception as e:
                print(f"⚠️  Error mapping {file_path}: {e}")

    def _get_module_exports(self, file_path: Path) -> Set[str]:
        """Get all names exported by a module"""
        exports = set()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            # Check for __all__
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == '__all__':
                            # Extract __all__ values
                            if isinstance(node.value, ast.List):
                                for elt in node.value.elts:
                                    if isinstance(elt, ast.Str):
                                        exports.add(elt.s)
                                    elif isinstance(elt, ast.Constant):
                                        exports.add(elt.value)
                            return exports

            # If no __all__, get all top-level definitions
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    if not node.name.startswith('_'):
                        exports.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    if not node.name.startswith('_'):
                        exports.add(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and not target.id.startswith('_'):
                            exports.add(target.id)
                elif isinstance(node, ast.ImportFrom):
                    # Re-exports
                    for alias in node.names:
                        if alias.name != '*':
                            exports.add(alias.asname or alias.name)

        except Exception as e:
            print(f"⚠️  Error getting exports from {file_path}: {e}")

        return exports

    def _check_all_imports(self):
        """Check all imports for errors"""
        for import_info in self.all_imports:
            self._check_import(import_info)

    def _check_import(self, import_info: ImportInfo):
        """Check a single import for errors"""
        module = import_info.module

        # Handle relative imports
        if module.startswith('.'):
            module = self._resolve_relative_import(import_info.file_path, module)
            if not module:
                self._add_import_error(
                    import_info,
                    'module_not_found',
                    f"Cannot resolve relative import '{import_info.module}'"
                )
                return

        # Check if module exists
        if self._is_builtin_module(module):
            return  # Skip builtin modules

        if self._is_third_party_module(module):
            # Check if installed
            if not self._is_module_installed(module):
                self._add_import_error(
                    import_info,
                    'module_not_found',
                    f"Third-party module '{module}' is not installed"
                )
            return

        # Check local modules
        if not self._is_local_module_available(module):
            # Try to find similar modules
            suggestions = self._find_similar_modules(module)
            error = ImportError(
                file_path=import_info.file_path,
                line_number=import_info.line_number,
                error_type='module_not_found',
                import_info=import_info,
                error_message=f"Module '{module}' not found",
                suggested_fixes=suggestions
            )
            self.import_errors.append(error)
            return

        # For 'from' imports, check if names exist
        if import_info.import_type == 'from' and import_info.names:
            module_exports = self.module_exports.get(module, set())
            for name in import_info.names:
                if name == '*':
                    continue
                if name not in module_exports:
                    # Try to find the name in other modules
                    suggestions = self._find_name_in_modules(name)
                    error = ImportError(
                        file_path=import_info.file_path,
                        line_number=import_info.line_number,
                        error_type='attribute_not_found',
                        import_info=import_info,
                        error_message=f"Cannot import '{name}' from '{module}'",
                        suggested_fixes=suggestions
                    )
                    self.import_errors.append(error)

    def _resolve_relative_import(self, file_path: str, module: str) -> Optional[str]:
        """Resolve relative import to absolute module name"""
        try:
            current_module = self.file_modules.get(file_path, '')
            if not current_module:
                return None

            parts = current_module.split('.')
            level = len(module) - len(module.lstrip('.'))

            if level > len(parts):
                return None

            if level > 0:
                parts = parts[:-level]

            if module.lstrip('.'):
                parts.append(module.lstrip('.'))

            return '.'.join(parts)

        except Exception:
            return None

    def _is_builtin_module(self, module: str) -> bool:
        """Check if module is a Python builtin"""
        return module.split('.')[0] in sys.builtin_module_names

    def _is_third_party_module(self, module: str) -> bool:
        """Check if module is likely a third-party module"""
        # Simple heuristic: not starting with project name
        project_name = self.project_root.name
        return not module.startswith(('src', project_name, '.'))

    def _is_module_installed(self, module: str) -> bool:
        """Check if a module is installed"""
        try:
            spec = importlib.util.find_spec(module.split('.')[0])
            return spec is not None
        except (ImportError, ModuleNotFoundError):
            return False

    def _is_local_module_available(self, module: str) -> bool:
        """Check if local module exists"""
        # Check in our mapped modules
        if module in self.module_exports:
            return True

        # Check parent modules (e.g., 'src' for 'src.agents')
        parts = module.split('.')
        for i in range(len(parts)):
            partial = '.'.join(parts[:i+1])
            if partial in self.module_exports:
                continue
            # Check if it's a package (__init__.py exists)
            package_path = self.project_root / Path(*partial.split('.')) / '__init__.py'
            if not package_path.exists():
                return False

        return True

    def _find_similar_modules(self, module: str) -> List[str]:
        """Find modules with similar names"""
        suggestions = []
        module_lower = module.lower()

        for known_module in self.module_exports.keys():
            if module_lower in known_module.lower() or known_module.lower() in module_lower:
                suggestions.append(f"Did you mean '{known_module}'?")

        # Check common mistakes
        if module.startswith('src.'):
            alt = module.replace('src.', 'src.agents.', 1)
            if alt in self.module_exports:
                suggestions.append(f"Try 'from {alt} import ...'")

        return suggestions[:3]  # Limit suggestions

    def _find_name_in_modules(self, name: str) -> List[str]:
        """Find which modules export a specific name"""
        suggestions = []

        for module, exports in self.module_exports.items():
            if name in exports:
                suggestions.append(f"'{name}' found in '{module}'")

        return suggestions[:3]

    def _add_import_error(self, import_info: ImportInfo, error_type: str, message: str):
        """Add an import error"""
        error = ImportError(
            file_path=import_info.file_path,
            line_number=import_info.line_number,
            error_type=error_type,
            import_info=import_info,
            error_message=message
        )
        self.import_errors.append(error)

    def _detect_circular_imports(self):
        """Detect circular import dependencies"""
        # Build import graph
        import_graph = defaultdict(set)

        for import_info in self.all_imports:
            from_module = self.file_modules.get(import_info.file_path, '')
            to_module = import_info.module

            if from_module and to_module and not to_module.startswith('.'):
                import_graph[from_module].add(to_module)

        # Find cycles using DFS
        visited = set()
        rec_stack = set()

        def find_cycle(self, module: str, path: List[str]) -> Optional[List[str]]:
            visited.add(module)
            rec_stack.add(module)
            path.append(module)

            for neighbor in import_graph.get(module, []):
                if neighbor not in visited:
                    cycle = find_cycle(neighbor, path.copy())
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # Found cycle - find the cycle start
                    try:
                        cycle_start = path.index(neighbor)
                        return path[cycle_start:] + [neighbor]
                    except ValueError:
                        # Handle case where neighbor might not be in path
                        continue

            rec_stack.remove(module)
            return None

        # Check all modules
        for module in import_graph:
            if module not in visited:
                cycle = find_cycle(module, [])
                if cycle:
                    self.circular_imports.append(tuple(cycle))

    def _generate_report(self) -> Dict[str, Any]:
        """Generate the final audit report"""
        report = {
            'summary': {
                'total_files': len(self.file_modules),
                'total_imports': len(self.all_imports),
                'total_errors': len(self.import_errors),
                'circular_imports': len(self.circular_imports)
            },
            'errors_by_type': defaultdict(int),
            'errors_by_file': defaultdict(list),
            'circular_imports': self.circular_imports,
            'detailed_errors': []
        }

        # Group errors
        for error in self.import_errors:
            report['errors_by_type'][error.error_type] += 1
            report['errors_by_file'][error.file_path].append(error)

            # Detailed error info
            detailed = {
                'file': error.file_path,
                'line': error.line_number,
                'type': error.error_type,
                'import': str(error.import_info),
                'error': error.error_message,
                'fixes': error.suggested_fixes
            }
            report['detailed_errors'].append(detailed)

        return dict(report)

    def print_report(self, report: Dict[str, Any]):
        """Print a formatted report"""
        print("\n" + "="*60)
        print("📊 IMPORT AUDIT REPORT")
        print("="*60)

        # Summary
        summary = report['summary']
        print(f"\n📈 Summary:")
        print(f"  • Files analyzed: {summary['total_files']}")
        print(f"  • Import statements: {summary['total_imports']}")
        print(f"  • Import errors: {summary['total_errors']}")
        print(f"  • Circular imports: {summary['circular_imports']}")

        # Errors by type
        if report['errors_by_type']:
            print(f"\n🔍 Errors by Type:")
            for error_type, count in report['errors_by_type'].items():
                print(f"  • {error_type}: {count}")

        # Circular imports
        if report['circular_imports']:
            print(f"\n🔄 Circular Imports:")
            for cycle in report['circular_imports']:
                print(f"  • {' -> '.join(cycle)}")

        # Detailed errors
        if report['detailed_errors']:
            print(f"\n❌ Detailed Errors:")
            for i, error in enumerate(report['detailed_errors'][:10], 1):
                print(f"\n  {i}. {error['file']}:{error['line']}")
                print(f"     Import: {error['import']}")
                print(f"     Error: {error['error']}")
                if error['fixes']:
                    print(f"     Suggestions:")
                    for fix in error['fixes']:
                        print(f"       - {fix}")

            if len(report['detailed_errors']) > 10:
                print(f"\n  ... and {len(report['detailed_errors']) - 10} more errors")

        print("\n" + "="*60)

    def save_report(self, report: Dict[str, Any], output_file: str = "import_audit_report.json"):
        """Save report to file"""
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Report saved to {output_file}")

    def generate_fix_script(self, report: Dict[str, Any], output_file: str = "fix_imports.py"):
        """Generate a script to fix common import issues"""
        fixes = []

        fixes.append("#!/usr/bin/env python3")
        fixes.append('"""Auto-generated import fixes"""')
        fixes.append("import os")
        fixes.append("import re")
        fixes.append("")
        fixes.append("def fix_imports():")
        fixes.append("    fixes_applied = 0")

        # Group fixes by file
        fixes_by_file = defaultdict(list)

        for error in report['detailed_errors']:
            if error['type'] == 'module_not_found' and error['fixes']:
                for fix in error['fixes']:
                    if "Did you mean" in fix:
                        # Extract suggested module
                        match = re.search(r"Did you mean '(.+)'\?", fix)
                        if match:
                            old_import = error['import']
                            new_module = match.group(1)

                            if old_import.startswith('from'):
                                old_module = old_import.split()[1]
                                new_import = old_import.replace(old_module, new_module)
                            else:
                                new_import = f"import {new_module}"

                            fixes_by_file[error['file']].append({
                                'line': error['line'],
                                'old': old_import,
                                'new': new_import
                            })

        # Generate fix code
        for file_path, file_fixes in fixes_by_file.items():
            fixes.append(f"\n    # Fix {file_path}")
            fixes.append(f"    try:")
            fixes.append(f"        with open('{file_path}', 'r') as f:")
            fixes.append(f"            lines = f.readlines()")
            fixes.append(f"        ")

            for fix in file_fixes:
                fixes.append(f"        # Line {fix['line']}: {fix['old']} -> {fix['new']}")
                fixes.append(f"        if lines[{fix['line']-1}].strip() == '{fix['old']}':")
                fixes.append(f"            lines[{fix['line']-1}] = '{fix['new']}\\n'")
                fixes.append(f"            fixes_applied += 1")

            fixes.append(f"        ")
            fixes.append(f"        with open('{file_path}', 'w') as f:")
            fixes.append(f"            f.writelines(lines)")
            fixes.append(f"    except Exception as e:")
            fixes.append(f"        print(f'Error fixing {file_path}: {{e}}')")

        fixes.append("\n    return fixes_applied")
        fixes.append("\n\nif __name__ == '__main__':")
        fixes.append("    fixes = fix_imports()")
        fixes.append("    print(f'Applied {fixes} fixes')")

        with open(output_file, 'w') as f:
            f.write('\n'.join(fixes))

        print(f"\n🔧 Fix script generated: {output_file}")
        print("   Review the script before running it!")

def main():
    """Main function to run the import auditor"""
    import argparse

    parser = argparse.ArgumentParser(description="Audit Python imports in a project")
    parser.add_argument("project_root", nargs='?', default=".",
                       help="Project root directory (default: current directory)")
    parser.add_argument("--output", "-o", default="import_audit_report.json",
                       help="Output file for the report")
    parser.add_argument("--fix-script", "-f", action="store_true",
                       help="Generate a fix script")
    parser.add_argument("--ignore", "-i", nargs='*',
                       help="Additional patterns to ignore")

    args = parser.parse_args()

    # Run audit
    auditor = ImportAuditor(args.project_root, ignore_patterns=args.ignore)
    report = auditor.audit()

    # Print report
    auditor.print_report(report)

    # Save report
    auditor.save_report(report, args.output)

    # Generate fix script if requested
    if args.fix_script:
        auditor.generate_fix_script(report)

    # Return exit code based on errors
    return 1 if report['summary']['total_errors'] > 0 else 0

if __name__ == "__main__":
    sys.exit(main())
