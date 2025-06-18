#!/usr/bin/env python3
"""
Python Version Compatibility Check
Ensures the runtime environment meets the project requirements.
"""

import sys
import subprocess
import os

def check_python_version():
    """Check if the current Python version meets requirements."""
    version_info = sys.version_info
    current_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    
    print(f"Current Python version: {current_version}")
    print(f"Full version info: {sys.version}")
    
    # Check against requirements
    if version_info.major != 3:
        print("❌ ERROR: Python 3 is required!")
        return False
    
    if version_info.minor < 10:
        print("❌ ERROR: Python 3.10 or higher is required (Gradio 5.x requirement)")
        return False
    
    if version_info.minor > 11:
        print("⚠️  WARNING: Python 3.12+ detected. PyTorch 2.0.x officially supports up to Python 3.11.")
        print("   Some dependencies may have compatibility issues.")
        print("   Recommended: Use Python 3.11 for best compatibility.")
        return True  # Allow but warn
    
    if version_info.minor == 10:
        print("✅ Python 3.10 detected - Minimum supported version")
        return True
    
    if version_info.minor == 11:
        print("✅ Python 3.11 detected - Optimal version for this project!")
        return True
    
    return True

def check_pip_version():
    """Check pip version and suggest upgrade if needed."""
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"\nPip version: {result.stdout.strip()}")
        else:
            print("⚠️  WARNING: Could not determine pip version")
    except Exception as e:
        print(f"⚠️  WARNING: Error checking pip version: {e}")

def suggest_python_installation():
    """Provide installation suggestions for Python 3.11."""
    print("\n" + "="*60)
    print("Python 3.11 Installation Guide:")
    print("="*60)
    
    if sys.platform == "darwin":  # macOS
        print("\nFor macOS:")
        print("1. Using Homebrew:")
        print("   brew install python@3.11")
        print("   brew link python@3.11")
        print("\n2. Using pyenv:")
        print("   pyenv install 3.11")
        print("   pyenv local 3.11")
        
    elif sys.platform.startswith("linux"):
        print("\nFor Linux:")
        print("1. Using apt (Ubuntu/Debian):")
        print("   sudo apt update")
        print("   sudo apt install python3.11 python3.11-venv python3.11-dev")
        print("\n2. Using pyenv:")
        print("   pyenv install 3.11")
        print("   pyenv local 3.11")
        
    elif sys.platform == "win32":
        print("\nFor Windows:")
        print("1. Download from python.org:")
        print("   https://www.python.org/downloads/release/python-3119/")
        print("\n2. Using Chocolatey:")
        print("   choco install python311")
    
    print("\n3. Using conda (all platforms):")
    print("   conda create -n myenv python=3.11")
    print("   conda activate myenv")
    
    print("\n" + "="*60)

def main():
    print("Python Version Compatibility Check")
    print("="*60)
    
    # Check Python version
    version_ok = check_python_version()
    
    # Check pip version
    check_pip_version()
    
    # Check for .python-version file
    if os.path.exists(".python-version"):
        with open(".python-version", "r") as f:
            required_version = f.read().strip()
        print(f"\n📋 .python-version file specifies: {required_version}")
    
    # Provide recommendations
    if not version_ok:
        print("\n❌ Python version check failed!")
        suggest_python_installation()
        sys.exit(1)
    else:
        if sys.version_info.minor != 11:
            print("\n💡 Recommendation: Install Python 3.11 for optimal compatibility")
            suggest_python_installation()
        print("\n✅ Python version check passed!")
        sys.exit(0)

if __name__ == "__main__":
    main() 