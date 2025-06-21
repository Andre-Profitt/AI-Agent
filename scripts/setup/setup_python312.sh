#!/bin/bash
# Setup Python 3.12 for AI Agent

echo "🐍 Python 3.12 Setup Guide for AI Agent"
echo "======================================"
echo ""
echo "To use Python 3.12, you have several options:"
echo ""
echo "Option 1: Using pyenv (Recommended)"
echo "-----------------------------------"
echo "1. Install pyenv if not already installed:"
echo "   curl https://pyenv.run | bash"
echo ""
echo "2. Install Python 3.12:"
echo "   pyenv install 3.12.0"
echo ""
echo "3. Set as local version for this project:"
echo "   cd $(pwd)"
echo "   pyenv local 3.12.0"
echo ""
echo "Option 2: Using Homebrew (macOS)"
echo "--------------------------------"
echo "1. Install Python 3.12:"
echo "   brew install python@3.12"
echo ""
echo "2. Create virtual environment:"
echo "   python3.12 -m venv venv"
echo "   source venv/bin/activate"
echo ""
echo "Option 3: Using official installer"
echo "----------------------------------"
echo "1. Download from: https://www.python.org/downloads/release/python-3120/"
echo "2. Run the installer"
echo "3. Create virtual environment as shown above"
echo ""
echo "After setting up Python 3.12, run:"
echo "  pip install -r requirements_simplified.txt"
echo ""

# Create a Python version check script
cat > check_python_version.py << 'EOF'
import sys
print(f"Current Python version: {sys.version}")
if sys.version_info[:2] == (3, 12):
    print("✅ Python 3.12 is active!")
else:
    print(f"⚠️  Python {sys.version_info.major}.{sys.version_info.minor} is active. Please switch to 3.12")
EOF

python3 check_python_version.py