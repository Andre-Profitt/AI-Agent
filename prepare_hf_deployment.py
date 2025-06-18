#!/usr/bin/env python3
"""
Prepare for Hugging Face Spaces Deployment
Ensures all requirements are met for successful deployment.
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime

def check_file_exists(filename):
    """Check if a required file exists."""
    if os.path.exists(filename):
        print(f"✅ {filename} exists")
        return True
    else:
        print(f"❌ {filename} is missing!")
        return False

def validate_requirements():
    """Validate the requirements.txt file."""
    print("\nValidating requirements.txt...")
    
    critical_packages = {
        "pydantic==2.8.2": "Core dependency for data validation",
        "langchain==0.3.25": "Main AI framework",
        "gradio==5.25.2": "UI framework",
        "torch==2.0.1": "GPU acceleration",
        "numpy==1.26.4": "Scientific computing (v1.x for compatibility)"
    }
    
    try:
        with open("requirements.txt", "r") as f:
            requirements = f.read()
        
        missing = []
        for package, description in critical_packages.items():
            if package in requirements:
                print(f"✅ {package} - {description}")
            else:
                print(f"❌ {package} - {description}")
                missing.append(package)
        
        return len(missing) == 0
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False

def check_python_runtime_txt():
    """Check or create runtime.txt for Python version specification."""
    print("\nChecking Python runtime specification...")
    
    if os.path.exists("runtime.txt"):
        with open("runtime.txt", "r") as f:
            runtime = f.read().strip()
        print(f"✅ runtime.txt exists: {runtime}")
        
        if "3.11" not in runtime:
            print("⚠️  WARNING: runtime.txt doesn't specify Python 3.11")
            print("   Updating to python-3.11")
            with open("runtime.txt", "w") as f:
                f.write("python-3.11")
    else:
        print("📝 Creating runtime.txt with Python 3.11 specification")
        with open("runtime.txt", "w") as f:
            f.write("python-3.11")
        print("✅ runtime.txt created")

def check_app_file():
    """Verify app.py exists and has proper Gradio launch configuration."""
    print("\nChecking app.py configuration...")
    
    if not os.path.exists("app.py"):
        print("❌ app.py not found!")
        return False
    
    with open("app.py", "r") as f:
        content = f.read()
    
    # Check for Gradio launch
    if "demo.launch(" in content or "iface.launch(" in content or ".launch(" in content:
        print("✅ app.py contains Gradio launch")
        
        # Check for proper server configuration
        if 'server_name="0.0.0.0"' in content:
            print("✅ Server configured for external access")
        else:
            print("⚠️  WARNING: Consider adding server_name='0.0.0.0' to launch()")
        
        return True
    else:
        print("❌ app.py doesn't contain Gradio launch!")
        return False

def generate_deployment_report():
    """Generate a deployment readiness report."""
    print("\n" + "="*60)
    print("HUGGING FACE SPACES DEPLOYMENT READINESS REPORT")
    print("="*60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nChecking deployment requirements...")
    
    all_good = True
    
    # Check required files
    print("\n1. Required Files:")
    all_good &= check_file_exists("app.py")
    all_good &= check_file_exists("requirements.txt")
    all_good &= check_file_exists("README.md")
    
    # Validate requirements
    print("\n2. Dependency Validation:")
    all_good &= validate_requirements()
    
    # Check Python runtime
    print("\n3. Runtime Configuration:")
    check_python_runtime_txt()
    
    # Check app configuration
    print("\n4. Application Configuration:")
    all_good &= check_app_file()
    
    # Check for large files
    print("\n5. File Size Check:")
    large_files = []
    for root, dirs, files in os.walk("."):
        # Skip git directory
        if ".git" in root:
            continue
        for file in files:
            filepath = os.path.join(root, file)
            try:
                size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                if size > 10:
                    large_files.append((filepath, size))
            except:
                pass
    
    if large_files:
        print("⚠️  WARNING: Large files detected (>10MB):")
        for filepath, size in sorted(large_files, key=lambda x: x[1], reverse=True):
            print(f"   - {filepath}: {size:.1f}MB")
        print("   Consider using Git LFS or external storage")
    else:
        print("✅ No large files detected")
    
    # Summary
    print("\n" + "="*60)
    if all_good:
        print("✅ DEPLOYMENT READY!")
        print("\nNext steps:")
        print("1. Commit all changes: git add -A && git commit -m 'Ready for HF deployment'")
        print("2. Push to GitHub: git push origin main")
        print("3. Create a new Space on Hugging Face")
        print("4. Link your GitHub repository to the Space")
        print("5. The Space will automatically build and deploy")
    else:
        print("❌ DEPLOYMENT ISSUES DETECTED!")
        print("\nPlease fix the issues above before deploying.")
    
    print("="*60)

def create_space_readme():
    """Create a README specifically for Hugging Face Spaces."""
    print("\nCreating Hugging Face Spaces README...")
    
    readme_content = """---
title: AI Agent Assistant
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
python_version: 3.11
---

# AI Agent Assistant

An advanced AI agent with strategic planning, reflection, and tool use capabilities.

## Features

- 🎯 Strategic Planning with step-by-step execution
- 🔧 Multiple specialized tools (web search, code execution, file reading, etc.)
- 🧠 Self-reflection and error correction
- 💡 Intelligent routing between different approaches
- 🚀 Powered by LangChain, LlamaIndex, and Groq

## Configuration

This Space requires the following environment variables:
- `GROQ_API_KEY`: Your Groq API key
- `OPENAI_API_KEY`: Your OpenAI API key (optional)
- `TAVILY_API_KEY`: Your Tavily API key (optional)

## Usage

1. Enter your query in the text box
2. The agent will plan and execute steps to answer your question
3. View the step-by-step process and final answer

## Technical Stack

- Python 3.11
- LangChain 0.3.25 with Pydantic v2
- LlamaIndex 0.12.42
- Gradio 5.25.2
- PyTorch 2.0.1
"""
    
    with open("README_HF_SPACE.md", "w") as f:
        f.write(readme_content)
    
    print("✅ README_HF_SPACE.md created")
    print("   Copy this content to your Space's README.md after creating the Space")

def main():
    """Main deployment preparation function."""
    print("🚀 Hugging Face Spaces Deployment Preparation")
    print("="*60)
    
    # Generate deployment report
    generate_deployment_report()
    
    # Create Space README
    create_space_readme()
    
    print("\n📚 Additional Resources:")
    print("- Spaces Documentation: https://huggingface.co/docs/hub/spaces")
    print("- Gradio + Spaces: https://huggingface.co/docs/hub/spaces-sdks-gradio")
    print("- Environment Variables: https://huggingface.co/docs/hub/spaces-overview#managing-secrets")

if __name__ == "__main__":
    main() 