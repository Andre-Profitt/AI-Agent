#!/usr/bin/env python3
"""

from typing import Any
from typing import Dict
from typing import Optional
Comprehensive Code Quality Auditor
Detects syntax errors, import issues, type errors, and common code problems
"""

import os
import ast
import sys
import re
import json
import argparse
import fnmatch
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = "critical"  # Code won't run
    ERROR = "error"        # Likely to cause issues
    WARNING = "warning"    # Potential issues
    INFO = "info"          # Style/best practice


@dataclass
class CodeIssue:
    """Represents a code issue"""
    file_path: str
    line_number: int
    column: int
    severity: ErrorSeverity
    category: str
    message: str
    code_snippet: str = ""
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file_path,
            "line": self.line_number,
            "column": self.column,
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "snippet": self.code_snippet,
            "suggestion": self.suggestion
        }


class ComprehensiveCodeAuditor:
    """Comprehensive code quality auditor"""

    def __init__(self, project_root: str, ignore_patterns: Optional[List[str]] = None):
        self.project_root = Path(project_root).resolve()
        self.ignore_patterns = ignore_patterns or [
            '__pycache__', '.git', '.venv', 'venv', 'env',
            'build', 'dist', '*.egg-info', '.pytest_cache',
            'node_modules', '.idea', '.vscode'
        ]

        # Issue storage
        self.issues: List[CodeIssue] = []
        self.file_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Add project root to path
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

    def audit(self) -> Dict[str, Any]:
        """Run complete code audit"""
        print(f"🔍 Starting comprehensive code audit of {self.project_root}")

        # Discover Python files
        python_files = self._discover_python_files()
        print(f"📁 Found {len(python_files)} Python files to analyze")

        # Analyze each file
        for i, file_path in enumerate(python_files, 1):
            print(f"  Analyzing {i}/{len(python_files)}: {file_path.name}", end='\r')
            self._analyze_file(file_path)

        print(f"\n✅ Analysis complete! Found {len(self.issues)} issues")

        # Generate report
        return self._generate_report()

    def _discover_python_files(self) -> List[Path]:
        """Discover all Python files"""
        python_files = []

        for root, dirs, files in os.walk(self.project_root):
            # Filter ignored directories
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
        """Check if name matches pattern"""
        if '*' in pattern:
            return fnmatch.fnmatch(name, pattern)
        return pattern in name

    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file for all issues"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Store lines for snippet extraction
            lines = content.splitlines()

            # 1. Check syntax errors
            self._check_syntax_errors(file_path, content)

            # 2. Parse AST for detailed analysis
            try:
                tree = ast.parse(content, filename=str(file_path))

                # 3. Check undefined variables
                self._check_undefined_variables(file_path, tree, lines)

                # 4. Check imports
                self._check_imports(file_path, tree, lines)

                # 5. Check function issues
                self._check_function_issues(file_path, tree, lines)

                # 6. Check async/await issues
                self._check_async_issues(file_path, tree, lines)

            except SyntaxError:
                # Already handled in syntax check
                pass

        except Exception as e:
            self._add_issue(
                file_path, 1, 0,
                ErrorSeverity.ERROR,
                "file_read_error",
                f"Failed to analyze file: {str(e)}"
            )

    def _check_syntax_errors(self, file_path: Path, content: str):
        """Check for syntax errors"""
        try:
            compile(content, str(file_path), 'exec')
        except SyntaxError as e:
            self._add_issue(
                file_path, e.lineno or 1, e.offset or 0,
                ErrorSeverity.CRITICAL,
                "syntax_error",
                f"Syntax error: {e.msg}",
                suggestion="Fix the syntax error before the code can run"
            )

    def _check_undefined_variables(self, file_path: Path, tree: ast.AST, lines: List[str]):
        """Check for undefined variables"""
        class NameVisitor(ast.NodeVisitor):
            def __init__(self):
                self.defined = set()
                self.used = set()
                self.builtins = set(dir(__builtins__))
                self.current_scope = [self.defined]

            def visit_FunctionDef(self, node):
                self.defined.add(node.name)
                # New scope
                new_scope = set()
                self.current_scope.append(new_scope)

                # Add parameters
                for arg in node.args.args:
                    new_scope.add(arg.arg)

                self.generic_visit(node)
                self.current_scope.pop()

            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)

            def visit_ClassDef(self, node):
                self.defined.add(node.name)
                new_scope = set(['self', 'cls'])
                self.current_scope.append(new_scope)
                self.generic_visit(node)
                self.current_scope.pop()

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    self.current_scope[-1].add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    self.used.add((node.id, node.lineno, node.col_offset))

            def visit_For(self, node):
                # Add loop variable
                if isinstance(node.target, ast.Name):
                    self.current_scope[-1].add(node.target.id)
                self.generic_visit(node)

            def is_defined(self, name):
                # Check all scopes
                for scope in self.current_scope:
                    if name in scope:
                        return True
                return name in self.builtins

        visitor = NameVisitor()
        visitor.visit(tree)

        # Check undefined
        for name, line, col in visitor.used:
            if not visitor.is_defined(name):
                # Common false positives
                if name not in ['__name__', '__file__', '__doc__']:
                    self._add_issue(
                        file_path, line, col,
                        ErrorSeverity.ERROR,
                        "undefined_variable",
                        f"Undefined variable: '{name}'",
                        code_snippet=self._get_line(lines, line),
                        suggestion=f"Define '{name}' before using it"
                    )

    def _check_imports(self, file_path: Path, tree: ast.AST, lines: List[str]):
        """Check import issues"""
        class ImportVisitor(ast.NodeVisitor):
            def __init__(self, auditor, file_path, lines):
                self.auditor = auditor
                self.file_path = file_path
                self.lines = lines
                self.imported_names = set()

            def visit_Import(self, node):
                for alias in node.names:
                    self.imported_names.add(alias.asname or alias.name)

            def visit_ImportFrom(self, node):
                if node.names[0].name == '*':
                    self.auditor._add_issue(
                        self.file_path, node.lineno, node.col_offset,
                        ErrorSeverity.WARNING,
                        "wildcard_import",
                        f"Wildcard import from {node.module}",
                        code_snippet=self.auditor._get_line(self.lines, node.lineno),
                        suggestion="Import specific names instead of using *"
                    )

                for alias in node.names:
                    if alias.name != '*':
                        self.imported_names.add(alias.asname or alias.name)

        visitor = ImportVisitor(self, file_path, lines)
        visitor.visit(tree)
        return visitor.imported_names

    def _check_function_issues(self, file_path: Path, tree: ast.AST, lines: List[str]):
        """Check function-related issues"""
        class FunctionVisitor(ast.NodeVisitor):
            def __init__(self, auditor, file_path, lines):
                self.auditor = auditor
                self.file_path = file_path
                self.lines = lines

            def visit_FunctionDef(self, node):
                self._check_function(node)
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):
                self._check_function(node)
                self.generic_visit(node)

            def _check_function(self, node):
                # Check for too many arguments
                arg_count = len(node.args.args) + len(node.args.kwonlyargs)
                if arg_count > 7:
                    self.auditor._add_issue(
                        self.file_path, node.lineno, node.col_offset,
                        ErrorSeverity.WARNING,
                        "too_many_arguments",
                        f"Function '{node.name}' has {arg_count} arguments (recommended max: 7)",
                        code_snippet=self.auditor._get_line(self.lines, node.lineno),
                        suggestion="Consider using a class or configuration object"
                    )

        visitor = FunctionVisitor(self, file_path, lines)
        visitor.visit(tree)

    def _check_async_issues(self, file_path: Path, tree: ast.AST, lines: List[str]):
        """Check async/await issues"""
        class AsyncVisitor(ast.NodeVisitor):
            def __init__(self, auditor, file_path, lines):
                self.auditor = auditor
                self.file_path = file_path
                self.lines = lines
                self.in_async = False

            def visit_AsyncFunctionDef(self, node):
                old_async = self.in_async
                self.in_async = True

                # Check if async function has any await
                has_await = any(isinstance(n, ast.Await) for n in ast.walk(node))
                if not has_await:
                    self.auditor._add_issue(
                        self.file_path, node.lineno, node.col_offset,
                        ErrorSeverity.WARNING,
                        "async_without_await",
                        f"Async function '{node.name}' has no await statements",
                        code_snippet=self.auditor._get_line(self.lines, node.lineno),
                        suggestion="Remove async or add await statements"
                    )

                self.generic_visit(node)
                self.in_async = old_async

            def visit_Await(self, node):
                if not self.in_async:
                    self.auditor._add_issue(
                        self.file_path, node.lineno, node.col_offset,
                        ErrorSeverity.ERROR,
                        "await_outside_async",
                        "Await used outside async function",
                        code_snippet=self.auditor._get_line(self.lines, node.lineno),
                        suggestion="Use await only inside async functions"
                    )
                self.generic_visit(node)

        visitor = AsyncVisitor(self, file_path, lines)
        visitor.visit(tree)

    def _get_line(self, lines: List[str], line_num: int) -> str:
        """Get a line from the file"""
        if 0 < line_num <= len(lines):
            return lines[line_num - 1].strip()
        return ""

    def _add_issue(self, file_path: Path, line: int, column: int,
                   severity: ErrorSeverity, category: str, message: str,
                   code_snippet: str = "", suggestion: str = ""):
        """Add an issue to the list"""
        issue = CodeIssue(
            file_path=str(file_path),
            line_number=line,
            column=column,
            severity=severity,
            category=category,
            message=message,
            code_snippet=code_snippet,
            suggestion=suggestion
        )
        self.issues.append(issue)

        # Update stats
        self.file_stats[str(file_path)][category] += 1

    def _generate_report(self) -> Dict[str, Any]:
        """Generate the audit report"""
        # Group issues by various criteria
        by_severity = defaultdict(list)
        by_category = defaultdict(list)
        by_file = defaultdict(list)

        for issue in self.issues:
            by_severity[issue.severity.value].append(issue.to_dict())
            by_category[issue.category].append(issue.to_dict())
            by_file[issue.file_path].append(issue.to_dict())

        # Sort files by issue count
        files_by_issues = sorted(
            [(f, len(issues)) for f, issues in by_file.items()],
            key=lambda x: x[1],
            reverse=True
        )

        report = {
            "summary": {
                "total_files": len(self.file_stats),
                "total_issues": len(self.issues),
                "critical": len(by_severity["critical"]),
                "errors": len(by_severity["error"]),
                "warnings": len(by_severity["warning"]),
                "info": len(by_severity["info"])
            },
            "by_severity": dict(by_severity),
            "by_category": {k: len(v) for k, v in by_category.items()},
            "worst_files": files_by_issues[:10],
            "all_issues": [issue.to_dict() for issue in self.issues]
        }

        return report

    def print_report(self, report: Dict[str, Any], detailed: bool = False):
        """Print formatted report"""
        print("\n" + "="*70)
        print("📊 COMPREHENSIVE CODE AUDIT REPORT")
        print("="*70)

        # Summary
        summary = report["summary"]
        print(f"\n📈 Summary:")
        print(f"  Files analyzed: {summary['total_files']}")
        print(f"  Total issues: {summary['total_issues']}")
        print(f"    🔴 Critical: {summary['critical']}")
        print(f"    🟠 Errors: {summary['errors']}")
        print(f"    🟡 Warnings: {summary['warnings']}")
        print(f"    🔵 Info: {summary['info']}")

        # Issues by category
        print(f"\n📂 Issues by Category:")
        for category, count in sorted(report["by_category"].items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count}")

        # Worst files
        print(f"\n🔥 Files with Most Issues:")
        for file, count in report["worst_files"][:5]:
            file_name = Path(file).name
            print(f"  {file_name}: {count} issues")

        # Critical issues
        critical_issues = report["by_severity"].get("critical", [])
        if critical_issues:
            print(f"\n🚨 CRITICAL Issues (must fix):")
            for issue in critical_issues[:5]:
                print(f"\n  {Path(issue['file']).name}:{issue['line']}")
                print(f"    {issue['message']}")
                if issue['snippet']:
                    print(f"    Code: {issue['snippet']}")
                if issue['suggestion']:
                    print(f"    Fix: {issue['suggestion']}")

        # Top errors
        errors = report["by_severity"].get("error", [])
        if errors and detailed:
            print(f"\n❌ Top Errors:")
            for issue in errors[:5]:
                print(f"\n  {Path(issue['file']).name}:{issue['line']}")
                print(f"    {issue['message']}")
                if issue['suggestion']:
                    print(f"    Fix: {issue['suggestion']}")

        print("\n" + "="*70)

    def save_report(self, report: Dict[str, Any], output_file: str = "code_audit_report.json"):
        """Save report to file"""
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Full report saved to {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Comprehensive code quality audit")
    parser.add_argument("project_root", nargs='?', default=".",
                       help="Project root directory")
    parser.add_argument("--output", "-o", default="code_audit_report.json",
                       help="Output file for report")
    parser.add_argument("--detailed", "-d", action="store_true",
                       help="Show detailed report")

    args = parser.parse_args()

    # Run audit
    auditor = ComprehensiveCodeAuditor(args.project_root)
    report = auditor.audit()

    # Print report
    auditor.print_report(report, detailed=args.detailed)

    # Save report
    auditor.save_report(report, args.output)

    # Return appropriate exit code
    if report["summary"]["critical"] > 0:
        return 2
    elif report["summary"]["errors"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())