#!/usr/bin/env python3
"""
AI Agent Codebase Analyzer
Scans your codebase to identify upgrade opportunities in:
1. Monitoring & Metrics
2. Tool Integration & Orchestration  
3. Testing Infrastructure
4. AI/Agent-Specific Patterns
"""

import os
import ast
import re
import yaml
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class UpgradePoint:
    """Represents a potential upgrade opportunity"""
    category: str  # monitoring, orchestration, testing, or agent_specific
    priority: str  # high, medium, low
    file_path: str
    line_number: int
    description: str
    suggestion: str
    code_snippet: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "priority": self.priority,
            "file": self.file_path,
            "line": self.line_number,
            "description": self.description,
            "suggestion": self.suggestion,
            "snippet": self.code_snippet[:100] + "..." if len(self.code_snippet) > 100 else self.code_snippet
        }


class CodebaseAnalyzer:
    """Analyzes codebase for upgrade opportunities"""
    
    def __init__(self, root_dir: str = ".", patterns_path: str = "analyzer_patterns.yaml", max_workers: int = 4):
        self.root_dir = Path(root_dir)
        self.upgrade_points: List[UpgradePoint] = []
        self.stats = defaultdict(int)
        self.max_workers = max_workers
        self.patterns = self._load_patterns(patterns_path)
        self.monitoring_patterns = self.patterns.get("monitoring_patterns", {})
        self.orchestration_patterns = self.patterns.get("orchestration_patterns", {})
        self.testing_patterns = self.patterns.get("testing_patterns", {})
        self.agent_patterns = self.patterns.get("agent_patterns", {})
        
    def _load_patterns(self, path: str) -> Dict[str, Any]:
        """Load patterns from YAML config file"""
        try:
            with open(path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Patterns file {path} not found, using defaults")
            return self._get_default_patterns()
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            return self._get_default_patterns()
    
    def _get_default_patterns(self) -> Dict[str, Any]:
        """Fallback default patterns if YAML file is not available"""
        return {
            "monitoring_patterns": {
                "missing_metrics": [
                    {"pattern": "def\\s+(\\w+)\\s*\\([^)]*\\):", "description": "Functions without performance metrics"}
                ]
            },
            "orchestration_patterns": {
                "sequential_execution": [
                    {"pattern": "for\\s+tool\\s+in\\s+tools:", "description": "Sequential tool execution"}
                ]
            },
            "testing_patterns": {
                "missing_tests": [
                    {"pattern": "class\\s+(\\w+)(?!Test)", "description": "Classes without corresponding tests"}
                ]
            },
            "agent_patterns": {
                "missing_tool_registration": [
                    {"pattern": "class\\s+\\w+Tool", "description": "Tools not registered with orchestrator"}
                ]
            }
        }
        
    def analyze(self) -> Dict[str, Any]:
        """Main analysis entry point with parallel file processing"""
        logger.info(f"🔍 Analyzing codebase at: {self.root_dir}")
        
        # Scan Python files
        python_files = list(self.root_dir.glob("**/*.py"))
        logger.info(f"📁 Found {len(python_files)} Python files")
        
        # Parallel file analysis
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self._analyze_file, file_path): file_path for file_path in python_files}
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"⚠️  Error analyzing {file_path}: {e}")
        
        # Analyze project structure
        self._analyze_project_structure()
        
        # Generate report
        return self._generate_report()
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_patterns = ["__pycache__", ".git", "venv", ".env", "migrations", "node_modules", "dist", "build"]
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file"""
        if self._should_skip_file(file_path):
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            self.stats["files_analyzed"] += 1
            
            # Check monitoring patterns
            self._check_monitoring(file_path, content, lines)
            
            # Check orchestration patterns
            self._check_orchestration(file_path, content, lines)
            
            # Check testing patterns
            self._check_testing(file_path, content, lines)
            
            # Check agent-specific patterns
            self._check_agent_patterns(file_path, content, lines)
            
            # AST-based analysis
            self._analyze_ast(file_path, content)
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
    
    def _check_monitoring(self, file_path: Path, content: str, lines: List[str]) -> None:
        """Check for monitoring/metrics upgrade points"""
        
        # Check for missing OpenTelemetry
        if "opentelemetry" not in content and "tool" in str(file_path).lower():
            self.upgrade_points.append(UpgradePoint(
                category="monitoring",
                priority="high",
                file_path=str(file_path),
                line_number=1,
                description="Missing OpenTelemetry instrumentation",
                suggestion="Add OpenTelemetry spans for distributed tracing",
                code_snippet="from opentelemetry import trace\ntracer = trace.get_tracer(__name__)"
            ))
        
        # Check for metrics collection
        if "metrics" not in content and any(pattern in content for pattern in ["class", "def"]):
            self.upgrade_points.append(UpgradePoint(
                category="monitoring", 
                priority="medium",
                file_path=str(file_path),
                line_number=1,
                description="No metrics collection found",
                suggestion="Add Prometheus metrics for performance monitoring",
                code_snippet="from prometheus_client import Counter, Histogram"
            ))
        
        # Check patterns from config
        for pattern_type, patterns in self.monitoring_patterns.items():
            for pattern_data in patterns:
                pattern = pattern_data["pattern"]
                description = pattern_data["description"]
                for i, line in enumerate(lines):
                    if re.search(pattern, line):
                        self.upgrade_points.append(UpgradePoint(
                            category="monitoring",
                            priority="medium" if pattern_type == "basic_logging" else "high",
                            file_path=str(file_path),
                            line_number=i + 1,
                            description=description,
                            suggestion=self._get_monitoring_suggestion(pattern_type),
                            code_snippet=line.strip()
                        ))
    
    def _check_orchestration(self, file_path: Path, content: str, lines: List[str]) -> None:
        """Check for tool orchestration upgrade points"""
        
        # Check for workflow orchestration
        if "agent" in str(file_path).lower() and "workflow" not in content:
            self.upgrade_points.append(UpgradePoint(
                category="orchestration",
                priority="high",
                file_path=str(file_path),
                line_number=1,
                description="Missing workflow orchestration",
                suggestion="Implement workflow engine (e.g., Temporal, Airflow)",
                code_snippet="Consider using LangGraph or custom FSM"
            ))
        
        # Check for parallel execution
        if "async" in content and "asyncio.gather" not in content:
            self.upgrade_points.append(UpgradePoint(
                category="orchestration",
                priority="medium",
                file_path=str(file_path),
                line_number=1,
                description="Async code without parallel execution",
                suggestion="Use asyncio.gather() for parallel tool execution",
                code_snippet="results = await asyncio.gather(*tasks)"
            ))
        
        # Check patterns from config
        for pattern_type, patterns in self.orchestration_patterns.items():
            for pattern_data in patterns:
                pattern = pattern_data["pattern"]
                description = pattern_data["description"]
                for i, line in enumerate(lines):
                    if re.search(pattern, line):
                        self.upgrade_points.append(UpgradePoint(
                            category="orchestration",
                            priority="high" if "circuit" in pattern_type else "medium",
                            file_path=str(file_path),
                            line_number=i + 1,
                            description=description,
                            suggestion=self._get_orchestration_suggestion(pattern_type),
                            code_snippet=line.strip()
                        ))
    
    def _check_testing(self, file_path: Path, content: str, lines: List[str]) -> None:
        """Check for testing infrastructure upgrade points"""
        
        # Check test coverage
        if "test_" in str(file_path) and "pytest" in content:
            # Check for parametrized tests
            if "@pytest.mark.parametrize" not in content:
                self.upgrade_points.append(UpgradePoint(
                    category="testing",
                    priority="medium",
                    file_path=str(file_path),
                    line_number=1,
                    description="Tests without parametrization",
                    suggestion="Use @pytest.mark.parametrize for comprehensive testing",
                    code_snippet="@pytest.mark.parametrize('input,expected', [...])"
                ))
            
            # Check for fixtures
            if "@pytest.fixture" not in content and "def test_" in content:
                self.upgrade_points.append(UpgradePoint(
                    category="testing",
                    priority="medium",
                    file_path=str(file_path),
                    line_number=1,
                    description="Tests without fixtures",
                    suggestion="Use pytest fixtures for better test organization",
                    code_snippet="@pytest.fixture\ndef sample_data():"
                ))
        
        # Check for integration tests
        if "test_" in str(file_path) and all(x not in content for x in ["integration", "e2e", "end_to_end"]):
            self.upgrade_points.append(UpgradePoint(
                category="testing",
                priority="high",
                file_path=str(file_path),
                line_number=1,
                description="Missing integration tests",
                suggestion="Add integration tests for critical workflows",
                code_snippet="class TestIntegration:"
            ))
        
        # Check patterns from config
        for pattern_type, patterns in self.testing_patterns.items():
            for pattern_data in patterns:
                pattern = pattern_data["pattern"]
                description = pattern_data["description"]
                for i, line in enumerate(lines):
                    if re.search(pattern, line):
                        self.upgrade_points.append(UpgradePoint(
                            category="testing",
                            priority="medium",
                            file_path=str(file_path),
                            line_number=i + 1,
                            description=description,
                            suggestion=self._get_testing_suggestion(pattern_type),
                            code_snippet=line.strip()
                        ))
    
    def _check_agent_patterns(self, file_path: Path, content: str, lines: List[str]) -> None:
        """Check for AI/Agent-specific patterns"""
        
        # Check for LangGraph patterns
        if "langgraph" in content.lower() and "state" not in content:
            self.upgrade_points.append(UpgradePoint(
                category="agent_specific",
                priority="high",
                file_path=str(file_path),
                line_number=1,
                description="LangGraph without proper state management",
                suggestion="Implement proper state management for LangGraph workflows",
                code_snippet="from langgraph.graph import StateGraph"
            ))
        
        # Check for FSM patterns
        if "fsm" in content.lower() and "error" not in content:
            self.upgrade_points.append(UpgradePoint(
                category="agent_specific",
                priority="high",
                file_path=str(file_path),
                line_number=1,
                description="FSM without error handling",
                suggestion="Add error states and timeout handling to FSM",
                code_snippet="ERROR_STATE = 'error'\nTIMEOUT_STATE = 'timeout'"
            ))
        
        # Check for tool registration
        if "tool" in str(file_path).lower() and "@tool" in content:
            if "register" not in content and "orchestrator" not in content:
                self.upgrade_points.append(UpgradePoint(
                    category="agent_specific",
                    priority="medium",
                    file_path=str(file_path),
                    line_number=1,
                    description="Tools not registered with orchestrator",
                    suggestion="Register tools with the integration hub orchestrator",
                    code_snippet="orchestrator.register_tool(tool_name, tool_function)"
                ))
        
        # Check for async tool calls
        if "async def" in content and "await tool" in content:
            if "asyncio.gather" not in content:
                self.upgrade_points.append(UpgradePoint(
                    category="agent_specific",
                    priority="medium",
                    file_path=str(file_path),
                    line_number=1,
                    description="Sequential async tool calls",
                    suggestion="Use asyncio.gather() for parallel tool execution",
                    code_snippet="results = await asyncio.gather(*[tool() for tool in tools])"
                ))
    
    def _analyze_ast(self, file_path: Path, content: str) -> None:
        """AST-based code analysis"""
        try:
            tree = ast.parse(content)
            
            # Find classes without proper testing
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            # Check for dependency injection
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                    if not any(isinstance(arg.annotation, ast.Name) for arg in node.args.args[1:]):
                        self.upgrade_points.append(UpgradePoint(
                            category="orchestration",
                            priority="medium",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            description="Constructor without type hints for DI",
                            suggestion="Add type hints for dependency injection",
                            code_snippet=f"def __init__(self, service: ServiceType):"
                        ))
            
        except Exception as e:
            logger.debug(f"AST analysis failed for {file_path}: {e}")
    
    def _analyze_project_structure(self) -> None:
        """Analyze overall project structure"""
        
        # Check for monitoring setup
        if not (self.root_dir / "monitoring").exists():
            self.upgrade_points.append(UpgradePoint(
                category="monitoring",
                priority="high",
                file_path="project_root",
                line_number=0,
                description="Missing monitoring directory",
                suggestion="Create monitoring/ with Grafana dashboards and alerts",
                code_snippet="monitoring/dashboards/, monitoring/alerts/"
            ))
        
        # Check for load testing
        if not any((self.root_dir / name).exists() for name in ["locust", "load_tests", "performance"]):
            self.upgrade_points.append(UpgradePoint(
                category="testing",
                priority="high",
                file_path="project_root",
                line_number=0,
                description="Missing load testing setup",
                suggestion="Add load testing with Locust or similar",
                code_snippet="load_tests/locustfile.py"
            ))
        
        # Check for CI/CD monitoring
        if (self.root_dir / ".github/workflows").exists():
            workflow_files = list((self.root_dir / ".github/workflows").glob("*.yml"))
            has_monitoring = any("datadog" in f.read_text() or "prometheus" in f.read_text() 
                               for f in workflow_files if f.exists())
            if not has_monitoring:
                self.upgrade_points.append(UpgradePoint(
                    category="monitoring",
                    priority="medium",
                    file_path=".github/workflows",
                    line_number=0,
                    description="CI/CD without monitoring integration",
                    suggestion="Add deployment metrics to CI/CD pipeline",
                    code_snippet="- name: Report deployment metrics"
                ))
    
    def _get_monitoring_suggestion(self, pattern_type: str) -> str:
        """Get specific monitoring suggestions"""
        suggestions = {
            "missing_metrics": "Add @metrics_decorator or use prometheus_client",
            "basic_logging": "Use structured logging with extra fields",
            "missing_telemetry": "Wrap with OpenTelemetry spans"
        }
        return suggestions.get(pattern_type, "Add comprehensive monitoring")
    
    def _get_orchestration_suggestion(self, pattern_type: str) -> str:
        """Get specific orchestration suggestions"""
        suggestions = {
            "sequential_execution": "Use asyncio.gather() or ThreadPoolExecutor",
            "missing_retry": "Add @retry decorator or use tenacity",
            "no_circuit_breaker": "Implement circuit breaker pattern"
        }
        return suggestions.get(pattern_type, "Improve orchestration")
    
    def _get_testing_suggestion(self, pattern_type: str) -> str:
        """Get specific testing suggestions"""
        suggestions = {
            "missing_tests": "Add comprehensive test coverage",
            "weak_assertions": "Use stronger assertions with specific comparisons",
            "no_mocking": "Add proper mocking for external dependencies"
        }
        return suggestions.get(pattern_type, "Improve testing")
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate analysis report"""
        
        # Group by category
        by_category = defaultdict(list)
        for point in self.upgrade_points:
            by_category[point.category].append(point)
        
        # Priority counts
        priority_counts = defaultdict(lambda: defaultdict(int))
        for point in self.upgrade_points:
            priority_counts[point.category][point.priority] += 1
        
        report = {
            "summary": {
                "total_upgrade_points": len(self.upgrade_points),
                "files_analyzed": self.stats.get("files_analyzed", 0),
                "by_category": {
                    cat: len(points) for cat, points in by_category.items()
                },
                "by_priority": dict(priority_counts)
            },
            "monitoring": {
                "high_priority": [p.to_dict() for p in by_category["monitoring"] if p.priority == "high"],
                "medium_priority": [p.to_dict() for p in by_category["monitoring"] if p.priority == "medium"],
                "low_priority": [p.to_dict() for p in by_category["monitoring"] if p.priority == "low"],
                "recommendations": [
                    "Implement distributed tracing with OpenTelemetry",
                    "Add Prometheus metrics for all critical operations",
                    "Set up centralized logging with ELK or similar",
                    "Create Grafana dashboards for real-time monitoring"
                ]
            },
            "orchestration": {
                "high_priority": [p.to_dict() for p in by_category["orchestration"] if p.priority == "high"],
                "medium_priority": [p.to_dict() for p in by_category["orchestration"] if p.priority == "medium"],
                "low_priority": [p.to_dict() for p in by_category["orchestration"] if p.priority == "low"],
                "recommendations": [
                    "Implement workflow orchestration with Temporal or Airflow",
                    "Add circuit breakers for external services",
                    "Use async/parallel execution for tool calls",
                    "Implement saga pattern for distributed transactions"
                ]
            },
            "testing": {
                "high_priority": [p.to_dict() for p in by_category["testing"] if p.priority == "high"],
                "medium_priority": [p.to_dict() for p in by_category["testing"] if p.priority == "medium"],
                "low_priority": [p.to_dict() for p in by_category["testing"] if p.priority == "low"],
                "recommendations": [
                    "Add integration tests for all major workflows",
                    "Implement contract testing for APIs",
                    "Set up load testing with Locust",
                    "Add mutation testing for critical components"
                ]
            },
            "agent_specific": {
                "high_priority": [p.to_dict() for p in by_category["agent_specific"] if p.priority == "high"],
                "medium_priority": [p.to_dict() for p in by_category["agent_specific"] if p.priority == "medium"],
                "low_priority": [p.to_dict() for p in by_category["agent_specific"] if p.priority == "low"],
                "recommendations": [
                    "Register all tools with the integration hub orchestrator",
                    "Implement proper error handling in FSM workflows",
                    "Use parallel execution for tool calls",
                    "Add comprehensive agent testing with mock tools"
                ]
            }
        }
        
        return report


def print_report(report: Dict[str, Any]) -> None:
    """Pretty print the analysis report"""
    print("\n" + "="*80)
    print("🔍 CODEBASE ANALYSIS REPORT")
    print("="*80 + "\n")
    
    # Summary
    summary = report["summary"]
    print(f"📊 SUMMARY")
    print(f"   Total upgrade points: {summary['total_upgrade_points']}")
    print(f"   Files analyzed: {summary['files_analyzed']}")
    print(f"   By category:")
    for cat, count in summary["by_category"].items():
        print(f"      - {cat.capitalize()}: {count}")
    print()
    
    # Details by category
    for category in ["monitoring", "orchestration", "testing", "agent_specific"]:
        if category in report:
            cat_data = report[category]
            print(f"\n📌 {category.upper()} UPGRADE POINTS")
            print("-" * 60)
            
            for priority in ["high", "medium", "low"]:
                points = cat_data[f"{priority}_priority"]
                if points:
                    print(f"\n🔴 {priority.upper()} Priority ({len(points)} items):")
                    for point in points[:3]:  # Show top 3
                        print(f"   📁 {point['file']}:{point['line']}")
                        print(f"      Issue: {point['description']}")
                        print(f"      Fix: {point['suggestion']}")
                        print()
            
            print("\n💡 Recommendations:")
            for rec in cat_data["recommendations"]:
                print(f"   • {rec}")


def generate_markdown_report(report: Dict[str, Any]) -> str:
    """Generate Markdown report"""
    md = []
    md.append("# AI Agent Codebase Analysis Report")
    md.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("")
    
    # Summary
    summary = report["summary"]
    md.append("## 📊 Summary")
    md.append(f"- **Total upgrade points:** {summary['total_upgrade_points']}")
    md.append(f"- **Files analyzed:** {summary['files_analyzed']}")
    md.append("")
    
    md.append("### By Category")
    for cat, count in summary["by_category"].items():
        md.append(f"- **{cat.capitalize()}:** {count}")
    md.append("")
    
    # Details by category
    for category in ["monitoring", "orchestration", "testing", "agent_specific"]:
        if category in report:
            cat_data = report[category]
            md.append(f"## 📌 {category.upper()} Upgrade Points")
            md.append("")
            
            for priority in ["high", "medium", "low"]:
                points = cat_data[f"{priority}_priority"]
                if points:
                    md.append(f"### 🔴 {priority.upper()} Priority ({len(points)} items)")
                    md.append("")
                    
                    for point in points:
                        md.append(f"#### {point['file']}:{point['line']}")
                        md.append(f"- **Issue:** {point['description']}")
                        md.append(f"- **Fix:** {point['suggestion']}")
                        if point['snippet']:
                            md.append(f"- **Code:** `{point['snippet']}`")
                        md.append("")
            
            md.append("### 💡 Recommendations")
            for rec in cat_data["recommendations"]:
                md.append(f"- {rec}")
            md.append("")
    
    return "\n".join(md)


def generate_html_report(report: Dict[str, Any]) -> str:
    """Generate HTML report"""
    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html><head>")
    html.append("<title>AI Agent Codebase Analysis Report</title>")
    html.append("<style>")
    html.append("body { font-family: Arial, sans-serif; margin: 40px; }")
    html.append(".summary { background: #f5f5f5; padding: 20px; border-radius: 5px; }")
    html.append(".high { color: #d32f2f; }")
    html.append(".medium { color: #f57c00; }")
    html.append(".low { color: #388e3c; }")
    html.append(".category { margin: 30px 0; }")
    html.append("</style>")
    html.append("</head><body>")
    
    html.append("<h1>AI Agent Codebase Analysis Report</h1>")
    html.append(f"<p><em>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>")
    
    # Summary
    summary = report["summary"]
    html.append('<div class="summary">')
    html.append("<h2>📊 Summary</h2>")
    html.append(f"<p><strong>Total upgrade points:</strong> {summary['total_upgrade_points']}</p>")
    html.append(f"<p><strong>Files analyzed:</strong> {summary['files_analyzed']}</p>")
    html.append("</div>")
    
    # Details by category
    for category in ["monitoring", "orchestration", "testing", "agent_specific"]:
        if category in report:
            cat_data = report[category]
            html.append(f'<div class="category">')
            html.append(f"<h2>📌 {category.upper()} Upgrade Points</h2>")
            
            for priority in ["high", "medium", "low"]:
                points = cat_data[f"{priority}_priority"]
                if points:
                    html.append(f'<h3 class="{priority}">🔴 {priority.upper()} Priority ({len(points)} items)</h3>')
                    html.append("<ul>")
                    for point in points:
                        html.append(f"<li><strong>{point['file']}:{point['line']}</strong>")
                        html.append(f"<br>Issue: {point['description']}")
                        html.append(f"<br>Fix: {point['suggestion']}")
                        html.append("</li>")
                    html.append("</ul>")
            
            html.append("<h3>💡 Recommendations</h3>")
            html.append("<ul>")
            for rec in cat_data["recommendations"]:
                html.append(f"<li>{rec}</li>")
            html.append("</ul>")
            html.append("</div>")
    
    html.append("</body></html>")
    return "\n".join(html)


def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Analyze codebase for upgrade opportunities")
    parser.add_argument("path", nargs="?", default=".", help="Path to analyze (default: current directory)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--markdown", action="store_true", help="Output as Markdown")
    parser.add_argument("--html", action="store_true", help="Output as HTML")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument("--patterns", default="analyzer_patterns.yaml", help="Patterns config file")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--self-test", action="store_true", help="Run self-test mode")
    
    args = parser.parse_args()
    
    # Self-test mode
    if args.self_test:
        logger.info("Running self-test mode...")
        analyzer = CodebaseAnalyzer(".", args.patterns, args.workers)
        report = analyzer.analyze()
        if report["summary"]["total_upgrade_points"] == 0:
            logger.error("Self-test failed: No upgrade points found in analyzer itself!")
            exit(1)
        else:
            logger.info(f"Self-test passed: Found {report['summary']['total_upgrade_points']} upgrade points")
            return
    
    # Run analysis
    analyzer = CodebaseAnalyzer(args.path, args.patterns, args.workers)
    report = analyzer.analyze()
    
    # Output results
    if args.json:
        output = json.dumps(report, indent=2)
    elif args.markdown:
        output = generate_markdown_report(report)
    elif args.html:
        output = generate_html_report(report)
    else:
        print_report(report)
        output = json.dumps(report, indent=2)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        logger.info(f"Report written to {args.output}")
    elif args.json or args.markdown or args.html:
        print(output)


if __name__ == "__main__":
    main() 