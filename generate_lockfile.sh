#!/bin/bash
# Generate Requirements Lockfile Script
# This creates a complete snapshot of all installed dependencies

echo "🔒 Generating Requirements Lockfile"
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  No virtual environment found."
    echo "Creating a new virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || { echo "❌ Failed to activate venv"; exit 1; }

echo "Python version: $(python --version)"
echo ""

# Install dependencies
echo "📦 Installing dependencies from requirements.txt..."
echo "This may take several minutes..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
    
    # Generate lockfile
    echo ""
    echo "🔒 Generating requirements.lock..."
    pip freeze > requirements.lock
    
    # Count packages
    PACKAGE_COUNT=$(wc -l < requirements.lock)
    echo "✅ Lockfile generated with $PACKAGE_COUNT packages"
    
    # Show sample
    echo ""
    echo "📋 Sample of locked dependencies:"
    head -20 requirements.lock
    echo "..."
    echo ""
    echo "✅ Complete! Use 'pip install -r requirements.lock --no-deps' for exact reproduction"
else
    echo "❌ Failed to install dependencies"
    echo "Please check for compatibility issues"
    exit 1
fi 