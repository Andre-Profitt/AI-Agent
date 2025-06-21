#!/usr/bin/env python3
"""
Auto-generated fix script for pipeline issues
"""

import os
from pathlib import Path

def fix_pipeline_issues():
    """Fix identified pipeline issues"""
    print("🔧 Fixing pipeline issues...")
    
    # Fix missing imports by checking requirements
    print("\n1. Checking package requirements...")
    os.system("pip install -r requirements.txt")
    
    # Create missing __init__.py files
    print("\n2. Creating missing __init__.py files...")
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
    print("\n3. Creating missing configuration files...")
    
    if not Path(".env").exists() and Path(".env.example").exists():
        import shutil
        shutil.copy(".env.example", ".env")
        print("  ✓ Created .env from .env.example")
        
    print("\n✅ Fix script completed!")
    print("\nNext steps:")
    print("1. Review and update .env file with your API keys")
    print("2. Run 'pytest tests/' to verify functionality")
    print("3. Check logs for any remaining issues")
    
if __name__ == "__main__":
    fix_pipeline_issues()
