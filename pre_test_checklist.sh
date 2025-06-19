#!/bin/bash
# pre_test_checklist.sh
# Pre-test environment verification

echo "🔍 Pre-Test Environment Checklist"
echo "=================================="

# Check Python version
echo "1. Checking Python version..."
python --version
if [ $? -eq 0 ]; then
    echo "✅ Python is available"
else
    echo "❌ Python not found"
    exit 1
fi

# Check pytest installation
echo "2. Checking pytest installation..."
pip list | grep pytest
if [ $? -eq 0 ]; then
    echo "✅ Pytest is installed"
else
    echo "❌ Pytest not found"
    exit 1
fi

# Set test environment
echo "3. Setting test environment..."
export ENVIRONMENT=test
export LOG_LEVEL=DEBUG
export TESTING=true

# Verify test environment variables
echo "4. Verifying environment variables..."
echo "ENVIRONMENT: $ENVIRONMENT"
echo "LOG_LEVEL: $LOG_LEVEL"
echo "TESTING: $TESTING"

# Check if we're in the right directory
echo "5. Checking project structure..."
if [ -f "app.py" ] && [ -d "src" ] && [ -d "tests" ]; then
    echo "✅ Project structure looks correct"
else
    echo "❌ Wrong directory or missing files"
    exit 1
fi

# Check test database configuration
echo "6. Checking test database configuration..."
if [ -n "$SUPABASE_URL" ]; then
    echo "✅ Test database URL configured"
else
    echo "⚠️  No test database URL found (tests will use mocks)"
fi

echo ""
echo "🎯 Pre-test checklist complete!"
echo "Ready to run tests..." 