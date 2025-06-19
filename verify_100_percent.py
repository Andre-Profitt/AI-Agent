#!/usr/bin/env python3
"""
Comprehensive verification script to check 100% completion
"""

import os
import re
import ast
from pathlib import Path
from typing import Dict, List, Tuple
import json
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('verification.log')
    ]
)
logger = logging.getLogger(__name__)


class ImplementationVerifier:
    """Verify implementation completeness"""
    
    def __init__(self):
        self.results = {}
        self.details = {}
        
    def check_circuit_breakers(self) -> int:
        """Check circuit breaker implementation"""
        logger.info("\n🔍 Checking Circuit Breakers...")
        
        issues = []
        good_examples = []
        
        # Check for circuit breaker imports and usage
        for file_path in Path('src').rglob('*.py'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for good patterns
                if '@circuit_breaker' in content:
                    count = content.count('@circuit_breaker')
                    good_examples.append(f"{file_path}: {count} circuit breakers")
                
                # Check for database calls without protection
                if 'supabase' in content.lower():
                    # Look for unprotected patterns
                    if re.search(r'client\.(table|rpc|auth|storage)\(', content):
                        if '@circuit_breaker' not in content and '@track_database_operation' not in content:
                            issues.append(f"{file_path}: Unprotected database calls")
                            
            except Exception:
                pass
        
        score = 100 - len(issues) * 5
        self.details['circuit_breakers'] = {
            'score': max(0, score),
            'issues': issues,
            'good_examples': good_examples[:5]  # Show top 5
        }
        
        return max(0, score)
    
    def check_config_validation(self) -> int:
        """Check configuration validation implementation"""
        logger.info("\n🔍 Checking Config Validation...")
        
        issues = []
        protected_configs = []
        
        config_files = ['src/config/integrations.py', 'config/integrations.py']
        config_file = None
        
        for cf in config_files:
            if os.path.exists(cf):
                config_file = cf
                break
        
        if config_file:
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for protected methods
            if '_safe_get_env' in content:
                protected_configs.append("✓ Has _safe_get_env method")
            else:
                issues.append("Missing _safe_get_env method")
            
            if 'is_configured_safe' in content:
                protected_configs.append("✓ Has is_configured_safe method")
            else:
                issues.append("Missing is_configured_safe method")
            
            if '@circuit_breaker' in content:
                protected_configs.append("✓ Uses circuit breaker decorators")
            else:
                issues.append("No circuit breaker protection")
        
        # Check for unprotected config access
        for file_path in Path('src').rglob('*.py'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for direct config access
                if re.search(r'\.config\.\w+\.\w+(?!\()', content):
                    if '_get_safe_config_value' not in content:
                        issues.append(f"{file_path}: Direct config access")
                        
            except Exception:
                pass
        
        score = 100 - len(issues) * 10
        self.details['config_validation'] = {
            'score': max(0, score),
            'issues': issues,
            'protected_configs': protected_configs
        }
        
        return max(0, score)
    
    def check_structured_logging(self) -> int:
        """Check structured logging implementation"""
        logger.info("\n🔍 Checking Structured Logging...")
        
        print_count = 0
        fstring_count = 0
        files_with_prints = []
        files_with_fstrings = []
        
        # Exclude certain directories
        exclude_dirs = {'venv', 'env', '.env', 'build', 'dist', '__pycache__', '.git'}
        
        for file_path in Path('.').rglob('*.py'):
            # Skip excluded directories
            if any(excluded in str(file_path) for excluded in exclude_dirs):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Count prints (excluding test files for production check)
                if 'print(' in content and 'tests' not in str(file_path):
                    count = content.count('print(')
                    print_count += count
                    files_with_prints.append(f"{file_path}: {count} prints")
                
                # Count f-string logging
                fstring_matches = re.findall(r'logger\.\w+\(f["\']', content)
                if fstring_matches:
                    fstring_count += len(fstring_matches)
                    files_with_fstrings.append(f"{file_path}: {len(fstring_matches)} f-strings")
                    
            except Exception:
                pass
        
        # Calculate score
        total_issues = print_count + fstring_count
        score = 100 if total_issues == 0 else max(0, 100 - (total_issues * 2))
        
        self.details['structured_logging'] = {
            'score': score,
            'print_count': print_count,
            'fstring_count': fstring_count,
            'files_with_prints': files_with_prints[:5],
            'files_with_fstrings': files_with_fstrings[:5]
        }
        
        return score
    
    def check_type_hints(self) -> int:
        """Check type hints coverage"""
        logger.info("\n🔍 Checking Type Hints...")
        
        total_functions = 0
        typed_functions = 0
        total_inits = 0
        typed_inits = 0
        missing_hints = []
        
        for file_path in Path('src').rglob('*.py'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        total_functions += 1
                        
                        # Check if has return type
                        if node.returns is not None:
                            typed_functions += 1
                        elif node.name == '__init__':
                            total_inits += 1
                            if node.returns is not None:
                                typed_inits += 1
                            else:
                                missing_hints.append(f"{file_path}::{node.name}")
                                
            except Exception:
                pass
        
        # Calculate score
        if total_functions > 0:
            overall_ratio = typed_functions / total_functions
            init_ratio = typed_inits / total_inits if total_inits > 0 else 1
            score = int((overall_ratio * 0.5 + init_ratio * 0.5) * 100)
        else:
            score = 100
        
        self.details['type_hints'] = {
            'score': score,
            'total_functions': total_functions,
            'typed_functions': typed_functions,
            'total_inits': total_inits,
            'typed_inits': typed_inits,
            'missing_hints': missing_hints[:10]
        }
        
        return score
    
    def check_parallel_execution(self) -> int:
        """Check parallel execution usage"""
        logger.info("\n🔍 Checking Parallel Execution...")
        
        sequential_patterns = []
        parallel_usage = []
        
        for file_path in Path('src').rglob('*.py'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for sequential await in loops
                if re.search(r'for .+ in .+:\s*\n\s*.*await', content):
                    sequential_patterns.append(f"{file_path}: Sequential await in loop")
                
                # Check for parallel patterns
                if 'asyncio.gather' in content or 'ParallelExecutor' in content:
                    parallel_usage.append(f"{file_path}: Uses parallel execution")
                    
            except Exception:
                pass
        
        # Calculate score
        issues = len(sequential_patterns)
        good = len(parallel_usage)
        score = min(100, 70 + good * 10 - issues * 5)
        
        self.details['parallel_execution'] = {
            'score': max(0, score),
            'sequential_patterns': sequential_patterns[:5],
            'parallel_usage': parallel_usage[:5]
        }
        
        return max(0, score)
    
    def check_workflow_orchestration(self) -> int:
        """Check workflow orchestration integration"""
        logger.info("\n🔍 Checking Workflow Orchestration...")
        
        integration_points = []
        missing_integration = []
        
        key_files = [
            'src/agents/advanced_agent_fsm.py',
            'src/multi_agent_system.py',
            'src/application/agents/agent_executor.py'
        ]
        
        for file_path in key_files:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'AgentOrchestrator' in content or 'WorkflowEngine' in content:
                    integration_points.append(f"✓ {file_path}: Integrated")
                else:
                    missing_integration.append(f"✗ {file_path}: Not integrated")
        
        score = 100 - len(missing_integration) * 20
        
        self.details['workflow_orchestration'] = {
            'score': max(0, score),
            'integration_points': integration_points,
            'missing_integration': missing_integration
        }
        
        return max(0, score)
    
    def check_http_retry(self) -> int:
        """Check HTTP retry logic implementation"""
        logger.info("\n🔍 Checking HTTP Retry Logic...")
        
        unprotected_calls = []
        protected_calls = []
        
        patterns = [
            r'requests\.(get|post|put|delete)\(',
            r'aiohttp.*\.(get|post|put|delete)\(',
            r'urllib.*urlopen\('
        ]
        
        for file_path in Path('src').rglob('*.py'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in patterns:
                    if re.search(pattern, content):
                        if '@retry' in content or 'retry' in content.lower():
                            protected_calls.append(f"✓ {file_path}: Has retry logic")
                        else:
                            unprotected_calls.append(f"✗ {file_path}: No retry logic")
                            
            except Exception:
                pass
        
        score = 100 - len(unprotected_calls) * 10
        
        self.details['http_retry'] = {
            'score': max(0, score),
            'unprotected_calls': unprotected_calls[:5],
            'protected_calls': protected_calls[:5]
        }
        
        return max(0, score)
    
    def generate_report(self):
        """Generate comprehensive report"""
        
        print("\n" + "=" * 60)
        logger.info("📊 COMPREHENSIVE IMPLEMENTATION REPORT")
        print("=" * 60)
        
        checks = {
            "Circuit Breakers": self.check_circuit_breakers(),
            "Config Validation": self.check_config_validation(),
            "Structured Logging": self.check_structured_logging(),
            "Type Hints": self.check_type_hints(),
            "Parallel Execution": self.check_parallel_execution(),
            "Workflow Orchestration": self.check_workflow_orchestration(),
            "HTTP Retry Logic": self.check_http_retry()
        }
        
        # Display summary
        logger.info("\n📈 SUMMARY SCORES:")
        print("-" * 40)
        
        for category, score in checks.items():
            if score >= 90:
                status = "✅"
                color = "\033[92m"  # Green
            elif score >= 70:
                status = "⚠️"
                color = "\033[93m"  # Yellow
            else:
                status = "❌"
                color = "\033[91m"  # Red
            
            logger.info("{} {}{}: {}%\033[0m", extra={"status": status, "color": color, "category": category, "score": score})
        
        overall = sum(checks.values()) / len(checks)
        
        print("-" * 40)
        logger.info("🎯 OVERALL COMPLETION: {}%", extra={"overall": overall})
        
        # Detailed issues for categories below 90%
        logger.info("\n📋 DETAILED FINDINGS:")
        print("-" * 40)
        
        for category, score in checks.items():
            if score < 90:
                details = self.details.get(category.lower().replace(' ', '_'), {})
                logger.info("\n❗ {} ({}%):", extra={"category": category, "score": score})
                
                if 'issues' in details and details['issues']:
                    logger.info("  Issues found:")
                    for issue in details['issues'][:5]:
                        logger.info("    - {}", extra={"issue": issue})
                    if len(details['issues']) > 5:
                        logger.info("    ... and {} more", extra={"len_details__issues_____5": len(details['issues']) - 5})
                
                if 'files_with_prints' in details and details['files_with_prints']:
                    logger.info("  Files with print statements:")
                    for file in details['files_with_prints'][:3]:
                        logger.info("    - {}", extra={"file": file})
                
                if 'missing_hints' in details and details['missing_hints']:
                    logger.info("  Missing type hints:")
                    for hint in details['missing_hints'][:3]:
                        logger.info("    - {}", extra={"hint": hint})
        
        # Save detailed report
        report_file = "implementation_report.json"
        with open(report_file, 'w') as f:
            json.dump({
                'scores': checks,
                'overall': overall,
                'details': self.details
            }, f, indent=2)
        
        logger.info("\n💾 Detailed report saved to: {}", extra={"report_file": report_file})
        
        # Final verdict
        print("\n" + "=" * 60)
        if overall >= 95:
            logger.info("🎉 EXCELLENT! You've achieved near-perfect implementation!")
            logger.info("   Just a few minor tweaks needed for 100%.")
        elif overall >= 85:
            logger.info("👍 VERY GOOD! You're close to completion.")
            logger.info("   Focus on the remaining issues listed above.")
        elif overall >= 75:
            logger.info("💪 GOOD PROGRESS! Several areas need attention.")
            logger.info("   Review the detailed findings and fix the issues.")
        else:
            logger.info("⚠️  MORE WORK NEEDED!")
            logger.info("   Focus on the high-priority issues first.")
        print("=" * 60)
        
        return overall

if __name__ == "__main__":
    verifier = ImplementationVerifier()
    overall_score = verifier.generate_report()
    
    # Exit with appropriate code
    exit(0 if overall_score >= 95 else 1) 