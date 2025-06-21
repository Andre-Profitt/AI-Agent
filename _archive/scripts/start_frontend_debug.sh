#!/bin/bash

echo "🔍 Starting Frontend in Debug Mode..."
echo "===================================="
echo ""

cd "/Users/test/Desktop/ai agent/AI-Agent/saas-ui"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

echo ""
echo "🚀 Starting Vite development server..."
echo ""

# Run with explicit host to ensure it binds properly
npx vite --host 0.0.0.0 --port 3000