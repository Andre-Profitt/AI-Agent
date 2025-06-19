#!/usr/bin/env python3
"""
Complete fix script - runs all fixes in the correct order
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(cmd, description, critical=True):
    """Run a command and report status"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}...")
    print(f"{'='*60}")
    
    try:
        # Use shell=False with list of arguments for better security
        if isinstance(cmd, str):
            cmd = cmd.split()
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ {description} - Failed (exit code: {result.returncode})")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            if result.stdout:
                print("Standard output:")
                print(result.stdout)
            
            if critical:
                print(f"\n⚠️  Critical step failed. Fix the issue and run again.")
                sys.exit(1)
            return False
            
    except Exception as e:
        print(f"❌ {description} - Error: {e}")
        if critical:
            sys.exit(1)
        return False

def main():
    """Run complete fix process"""
    print("🚀 AI Agent System - Complete Fix Process")
    print("=" * 70)
    print("This will fix all remaining issues to achieve 100% completion")
    print("=" * 70)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return 1
    
    print(f"✅ Using Python {sys.version}")
    
    # Step 1: Install dependencies
    print("\n📦 Step 1: Installing dependencies...")
    dependencies = ["structlog", "tenacity", "pytest", "pytest-asyncio", "pytest-cov"]
    
    for dep in dependencies:
        try:
            __import__(dep.replace("-", "_"))
            print(f"  ✅ {dep} already installed")
        except ImportError:
            print(f"  📥 Installing {dep}...")
            run_command([sys.executable, "-m", "pip", "install", dep], f"Install {dep}")
    
    # Step 2: Create necessary directories
    print("\n📁 Step 2: Creating directories...")
    dirs_to_create = [
        "src/utils",
        "src/shared/types",
        "src/config",
        "src/infrastructure/resilience"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        init_file = Path(dir_path) / "__init__.py"
        if not init_file.exists():
            init_file.write_text('"""Package initialization"""')
        print(f"  ✅ Created {dir_path}")
    
    # Step 3: Verify implementation files exist
    print("\n📄 Step 3: Verifying implementation files...")
    impl_files = {
        "src/utils/structured_logging.py": "Structured logging",
        "src/shared/types/di_types.py": "DI type protocols",
        "src/utils/http_retry.py": "HTTP retry logic",
        "src/config/integrations.py": "Protected configuration"
    }
    
    for file_path, description in impl_files.items():
        if os.path.exists(file_path):
            print(f"  ✅ {description} exists")
        else:
            print(f"  ❌ Missing: {file_path}")
            print("     Please create this file from the implementation artifacts")
            return 1
    
    # Step 4: Run fix scripts in order
    print("\n🏃 Step 4: Running fix scripts...")
    
    # Make scripts executable
    scripts = [
        "fix_config_checks.py",
        "fix_fstring_logging.py",
        "remove_print_statements.py",
        "fix_remaining_prints.py",
        "fix_integration_hub.py",
        "add_type_hints.py"
    ]
    
    for script in scripts:
        if os.path.exists(script):
            os.chmod(script, 0o755)
    
    # Run scripts in optimal order
    fix_steps = [
        ("fix_remaining_prints.py", "Fix remaining print statements", True),
        ("fix_config_checks.py", "Fix config validation", True),
        ("fix_fstring_logging.py", "Fix f-string logging", True),
        ("fix_integration_hub.py", "Fix integration hub", True),
        ("add_type_hints.py", "Add type hints", False),  # Non-critical
    ]
    
    success_count = 0
    for script, desc, critical in fix_steps:
        if os.path.exists(script):
            if run_command([sys.executable, script], desc, critical):
                success_count += 1
        else:
            print(f"⚠️  Script {script} not found")
    
    print(f"\n📊 Completed {success_count}/{len(fix_steps)} fixes successfully")
    
    # Step 5: Run verification
    print("\n🔍 Step 5: Running verification...")
    time.sleep(1)  # Brief pause before verification
    
    if os.path.exists("verify_100_percent.py"):
        run_command([sys.executable, "verify_100_percent.py"], "Verify implementation", critical=False)
    else:
        print("⚠️  Verification script not found")
    
    # Step 6: Final checks
    print("\n✅ Step 6: Final checks...")
    
    # Check for remaining prints
    print("\nChecking for remaining print statements...")
    remaining_prints = []
    for root, dirs, files in os.walk('src'):
        if '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    if 'print(' in content:
                        count = content.count('print(')
                        remaining_prints.append((file_path, count))
                except:
                    pass
    
    if remaining_prints:
        print(f"⚠️  Found {len(remaining_prints)} files with print statements:")
        for file_path, count in remaining_prints[:5]:
            print(f"   - {file_path}: {count} prints")
    else:
        print("✅ No print statements found in src/")
    
    # Show next steps
    print("\n" + "="*70)
    print("📝 Next Steps:")
    print("="*70)
    print("1. Review the changes:")
    print("   git diff")
    print("\n2. Run your test suite:")
    print("   pytest tests/ -v")
    print("\n3. If all tests pass, commit your changes:")
    print("   git add -A")
    print("   git commit -m 'feat: achieve 100% implementation completion'")
    print("\n4. Push to GitHub:")
    print("   git push origin main")
    print("\n5. Celebrate! 🎉")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 