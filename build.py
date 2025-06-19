#!/usr/bin/env python3
import subprocess
import sys
import os
import shutil

def run_command(cmd, description):
    print(f"\n🔄 {description}...")
    try:
        subprocess.run(cmd, check=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {str(e)}")
        return False

def main():
    # Step 1: Copy Hugging Face Space requirements
    if os.path.exists("requirements-hf.txt"):
        shutil.copy("requirements-hf.txt", "requirements.txt")
        print("✅ Copied Hugging Face Space requirements")

    # Step 2: Install dependencies in stages
    stages = [
        # Core dependencies first
        ["numpy>=1.24.0,<2.0.0", "langchain==0.1.10", "langgraph==0.4.8"],
        # AI/ML dependencies
        ["torch==2.7.1", "torchvision==0.22.1", "transformers==4.37.2"],
        # Web framework
        ["fastapi==0.109.2", "gradio==5.25.2", "uvicorn==0.27.1"],
        # Data processing
        ["pandas>=2.0.0,<2.2.0", "openpyxl==3.1.5", "pypdf==5.6.0"],
        # Remaining dependencies
        ["-r", "requirements.txt"]
    ]

    for stage in stages:
        if not run_command(
            [sys.executable, "-m", "pip", "install", "--no-deps"] + stage,
            f"Installing stage: {stage}"
        ):
            print(f"❌ Failed to install stage: {stage}")
            sys.exit(1)

    # Step 3: Verify the installation
    if not run_command(
        [sys.executable, "-c", "import torch; import transformers; import langchain; print('Dependencies verified')"],
        "Verifying dependencies"
    ):
        print("❌ Dependency verification failed")
        sys.exit(1)

    print("\n✨ Build completed successfully!")

if __name__ == "__main__":
    main() 