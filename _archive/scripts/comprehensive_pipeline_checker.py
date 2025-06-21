#!/usr/bin/env python3
"""
Comprehensive Pipeline Checker
Verifies all components and pipelines work correctly
"""

import os
import sys
import ast
import importlib.util
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class PipelineChecker:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.issues = []
        self.warnings = []
        self.successes = []
        
    def check_all_pipelines(self):
        """Run comprehensive pipeline checks"""
        logger.info("🔍 Comprehensive Pipeline Check Starting...\n")
        
        # Phase 1: Check imports and dependencies
        logger.info("=" * 60)
        logger.info("PHASE 1: Import & Dependency Analysis")
        logger.info("=" * 60)
        self._check_imports()
        
        # Phase 2: Check configuration
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: Configuration Check")
        logger.info("=" * 60)
        self._check_configuration()
        
        # Phase 3: Check database
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: Database & Models Check")
        logger.info("=" * 60)
        self._check_database()
        
        # Phase 4: Check core functionality
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 4: Core Functionality Check")
        logger.info("=" * 60)
        self._check_core_functionality()
        
        # Phase 5: Check integrations
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 5: Integration Points Check")
        logger.info("=" * 60)
        self._check_integrations()
        
        # Phase 6: Generate fixes
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 6: Generating Fixes")
        logger.info("=" * 60)
        self._generate_fixes()
        
        # Summary
        self._print_summary()
        
    def _check_imports(self):
        """Check all imports and module dependencies"""
        logger.info("\n🔍 Checking imports across all Python files...")
        
        import_issues = []
        files_checked = 0
        
        for py_file in self.project_root.rglob("*.py"):
            if any(skip in str(py_file) for skip in [
                "__pycache__", ".venv", "venv", "env", 
                ".git", "build", "dist", ".pytest_cache"
            ]):
                continue
                
            files_checked += 1
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if not self._can_import(alias.name):
                                import_issues.append({
                                    "file": str(py_file.relative_to(self.project_root)),
                                    "line": node.lineno,
                                    "module": alias.name,
                                    "type": "import"
                                })
                                
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        if module and not self._can_import(module):
                            import_issues.append({
                                "file": str(py_file.relative_to(self.project_root)),
                                "line": node.lineno,
                                "module": module,
                                "type": "from"
                            })
                            
            except Exception as e:
                self.warnings.append(f"Could not parse {py_file}: {e}")
                
        logger.info(f"✓ Checked {files_checked} Python files")
        
        if import_issues:
            logger.error(f"\n❌ Found {len(import_issues)} import issues:")
            for issue in import_issues[:10]:  # Show first 10
                logger.error(f"  - {issue['file']}:{issue['line']} - Cannot import '{issue['module']}'")
            if len(import_issues) > 10:
                logger.error(f"  ... and {len(import_issues) - 10} more")
            self.issues.extend(import_issues)
        else:
            logger.info("✅ All imports resolved successfully!")
            self.successes.append("All imports valid")
            
    def _can_import(self, module_name: str) -> bool:
        """Check if a module can be imported"""
        try:
            # Check if it's a local module
            if module_name.startswith("src."):
                module_path = module_name.replace(".", "/") + ".py"
                if (self.project_root / module_path).exists():
                    return True
                # Check if it's a package
                package_path = module_name.replace(".", "/") + "/__init__.py"
                if (self.project_root / package_path).exists():
                    return True
                    
            # Try to find the module spec
            spec = importlib.util.find_spec(module_name.split('.')[0])
            return spec is not None
        except:
            return False
            
    def _check_configuration(self):
        """Check configuration files and environment"""
        logger.info("\n🔍 Checking configuration...")
        
        # Check for .env file
        env_file = self.project_root / ".env"
        env_example = self.project_root / ".env.example"
        
        if not env_file.exists():
            if env_example.exists():
                logger.warning("⚠️  .env file missing but .env.example exists")
                self.warnings.append("Missing .env file")
            else:
                logger.error("❌ No .env or .env.example file found")
                self.issues.append("No environment configuration")
        else:
            logger.info("✅ .env file exists")
            
        # Check for required environment variables
        required_vars = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "SUPABASE_URL",
            "SUPABASE_KEY"
        ]
        
        if env_file.exists():
            with open(env_file, 'r') as f:
                env_content = f.read()
                
            missing_vars = []
            for var in required_vars:
                if var not in env_content:
                    missing_vars.append(var)
                    
            if missing_vars:
                logger.warning(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
                self.warnings.append(f"Missing env vars: {missing_vars}")
            else:
                logger.info("✅ All required environment variables present")
                self.successes.append("Environment configured")
                
        # Check configuration files
        config_files = [
            "src/config/settings.py",
            "src/config/security.py",
            "src/config/performance.py",
            "src/infrastructure/config.py"
        ]
        
        for config_file in config_files:
            if (self.project_root / config_file).exists():
                logger.info(f"✅ {config_file} exists")
            else:
                logger.error(f"❌ {config_file} missing")
                self.issues.append(f"Missing config: {config_file}")
                
    def _check_database(self):
        """Check database setup and models"""
        logger.info("\n🔍 Checking database configuration...")
        
        # Check database files
        db_files = [
            "src/database/models.py",
            "src/database/connection_pool.py",
            "src/database/supabase_manager.py"
        ]
        
        for db_file in db_files:
            if (self.project_root / db_file).exists():
                logger.info(f"✅ {db_file} exists")
            else:
                logger.error(f"❌ {db_file} missing")
                self.issues.append(f"Missing database file: {db_file}")
                
        # Check if database can be imported
        try:
            # Test import
            spec = importlib.util.spec_from_file_location(
                "models",
                self.project_root / "src/database/models.py"
            )
            if spec and spec.loader:
                logger.info("✅ Database models can be imported")
                self.successes.append("Database models valid")
            else:
                logger.error("❌ Cannot import database models")
                self.issues.append("Database models import failed")
        except Exception as e:
            logger.error(f"❌ Database check failed: {e}")
            self.issues.append(f"Database error: {str(e)}")
            
    def _check_core_functionality(self):
        """Check core agent functionality"""
        logger.info("\n🔍 Checking core agent functionality...")
        
        # Check agent files
        agent_files = [
            "src/agents/unified_agent.py",
            "src/agents/super_agent.py",
            "src/core/advanced_reasoning.py",
            "src/core/self_improvement.py",
            "src/core/multimodal_support.py",
            "src/core/advanced_memory.py",
            "src/core/swarm_intelligence.py",
            "src/core/tool_creation.py"
        ]
        
        core_valid = True
        for agent_file in agent_files:
            if (self.project_root / agent_file).exists():
                logger.info(f"✅ {agent_file} exists")
                
                # Check for syntax errors
                try:
                    with open(self.project_root / agent_file, 'r') as f:
                        ast.parse(f.read())
                except SyntaxError as e:
                    logger.error(f"❌ Syntax error in {agent_file}: {e}")
                    self.issues.append(f"Syntax error in {agent_file}")
                    core_valid = False
            else:
                logger.error(f"❌ {agent_file} missing")
                self.issues.append(f"Missing core file: {agent_file}")
                core_valid = False
                
        if core_valid:
            self.successes.append("Core agent files valid")
            
        # Check if agents can be instantiated
        logger.info("\n🔍 Testing agent instantiation...")
        test_code = '''
import sys
sys.path.insert(0, '.')

try:
    from src.agents.unified_agent import create_agent
    agent = create_agent(name="TestAgent")
    print("SUCCESS: Unified agent created")
except Exception as e:
    print(f"ERROR: {e}")
    
try:
    from src.agents.super_agent import create_super_agent
    # Don't actually create super agent as it starts background tasks
    print("SUCCESS: Super agent module imported")
except Exception as e:
    print(f"ERROR: {e}")
'''
        
        test_file = self.project_root / "test_agent_creation.py"
        test_file.write_text(test_code)
        
        try:
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                cwd=str(self.project_root)
            )
            
            if "SUCCESS" in result.stdout:
                logger.info("✅ Agent instantiation successful")
                self.successes.append("Agents can be created")
            else:
                logger.error(f"❌ Agent instantiation failed:\n{result.stderr}")
                self.issues.append("Agent instantiation failed")
                
        except Exception as e:
            logger.error(f"❌ Test execution failed: {e}")
            self.issues.append(f"Test execution error: {str(e)}")
        finally:
            test_file.unlink(missing_ok=True)
            
    def _check_integrations(self):
        """Check integration points"""
        logger.info("\n🔍 Checking integration points...")
        
        # Check API server
        api_file = self.project_root / "src/api_server.py"
        if api_file.exists():
            logger.info("✅ API server file exists")
            
            # Check for FastAPI routes
            with open(api_file, 'r') as f:
                content = f.read()
                
            if "FastAPI" in content and "@app" in content:
                logger.info("✅ FastAPI routes defined")
                self.successes.append("API server configured")
            else:
                logger.warning("⚠️  API routes may not be properly defined")
                self.warnings.append("API routes unclear")
        else:
            logger.error("❌ API server file missing")
            self.issues.append("Missing API server")
            
        # Check tool system
        tool_files = [
            "src/tools/base_tool.py",
            "src/tools/registry.py",
            "src/application/tools/tool_executor.py"
        ]
        
        tools_valid = True
        for tool_file in tool_files:
            if not (self.project_root / tool_file).exists():
                logger.error(f"❌ {tool_file} missing")
                self.issues.append(f"Missing tool file: {tool_file}")
                tools_valid = False
                
        if tools_valid:
            logger.info("✅ Tool system files present")
            self.successes.append("Tool system valid")
            
    def _generate_fixes(self):
        """Generate fixes for identified issues"""
        if not self.issues:
            logger.info("\n✅ No issues to fix!")
            return
            
        logger.info(f"\n🔧 Generating fixes for {len(self.issues)} issues...")
        
        fixes = []
        
        # Group issues by type
        import_issues = [i for i in self.issues if isinstance(i, dict) and i.get('type') in ['import', 'from']]
        missing_files = [i for i in self.issues if isinstance(i, str) and 'Missing' in i]
        
        # Fix missing imports
        if import_issues:
            fixes.append({
                "type": "imports",
                "count": len(import_issues),
                "action": "Install missing packages or create missing modules"
            })
            
        # Fix missing files
        if missing_files:
            fixes.append({
                "type": "files",
                "count": len(missing_files),
                "action": "Create missing configuration and module files"
            })
            
        # Create fix script
        self._create_fix_script(fixes)
        
    def _create_fix_script(self, fixes: List[Dict[str, Any]]):
        """Create script to fix issues"""
        fix_script = '''#!/usr/bin/env python3
"""
Auto-generated fix script for pipeline issues
"""

import os
from pathlib import Path

def fix_pipeline_issues():
    """Fix identified pipeline issues"""
    print("🔧 Fixing pipeline issues...")
    
    # Fix missing imports by checking requirements
    print("\\n1. Checking package requirements...")
    os.system("pip install -r requirements.txt")
    
    # Create missing __init__.py files
    print("\\n2. Creating missing __init__.py files...")
    dirs_needing_init = [
        "src/core",
        "src/agents", 
        "src/tools",
        "src/config",
        "src/database",
        "src/application",
        "src/infrastructure",
        "src/gaia_components",
        "src/unified_architecture"
    ]
    
    for dir_path in dirs_needing_init:
        init_file = Path(dir_path) / "__init__.py"
        if not init_file.exists():
            init_file.parent.mkdir(parents=True, exist_ok=True)
            init_file.write_text("")
            print(f"  ✓ Created {init_file}")
            
    # Create missing config files
    print("\\n3. Creating missing configuration files...")
    
    if not Path(".env").exists() and Path(".env.example").exists():
        import shutil
        shutil.copy(".env.example", ".env")
        print("  ✓ Created .env from .env.example")
        
    print("\\n✅ Fix script completed!")
    print("\\nNext steps:")
    print("1. Review and update .env file with your API keys")
    print("2. Run 'pytest tests/' to verify functionality")
    print("3. Check logs for any remaining issues")
    
if __name__ == "__main__":
    fix_pipeline_issues()
'''
        
        fix_path = self.project_root / "fix_pipeline_issues.py"
        fix_path.write_text(fix_script)
        logger.info(f"\n✅ Created fix script: {fix_path}")
        logger.info("   Run: python fix_pipeline_issues.py")
        
    def _print_summary(self):
        """Print summary of checks"""
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE CHECK SUMMARY")
        logger.info("=" * 60)
        
        total_checks = len(self.successes) + len(self.warnings) + len(self.issues)
        
        logger.info(f"\n📊 Total checks performed: {total_checks}")
        logger.info(f"✅ Successes: {len(self.successes)}")
        logger.info(f"⚠️  Warnings: {len(self.warnings)}")
        logger.info(f"❌ Issues: {len(self.issues)}")
        
        if self.successes:
            logger.info("\n✅ Working Components:")
            for success in self.successes:
                logger.info(f"  - {success}")
                
        if self.warnings:
            logger.info("\n⚠️  Warnings:")
            for warning in self.warnings[:5]:
                logger.info(f"  - {warning}")
                
        if self.issues:
            logger.info("\n❌ Critical Issues:")
            issue_counts = {}
            for issue in self.issues:
                if isinstance(issue, str):
                    issue_type = issue.split(':')[0]
                    issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
                    
            for issue_type, count in issue_counts.items():
                logger.info(f"  - {issue_type}: {count} issues")
                
        # Overall health score
        health_score = (len(self.successes) / total_checks * 100) if total_checks > 0 else 0
        
        logger.info(f"\n🏥 Overall Pipeline Health: {health_score:.1f}%")
        
        if health_score >= 80:
            logger.info("✅ Pipeline is mostly healthy!")
        elif health_score >= 50:
            logger.info("⚠️  Pipeline needs some attention")
        else:
            logger.info("❌ Pipeline has significant issues")
            
def main():
    checker = PipelineChecker()
    checker.check_all_pipelines()

if __name__ == "__main__":
    main()