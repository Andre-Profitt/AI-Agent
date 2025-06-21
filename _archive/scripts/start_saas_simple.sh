#!/bin/bash

echo "🚀 AI Agent SaaS Platform - Simple Startup"
echo "=========================================="
echo ""

# Start test backend
echo "1. Starting test backend server..."
echo "   Run this in Terminal 1:"
echo ""
echo "   cd \"$(pwd)\""
echo "   source venv/bin/activate"
echo "   python test_server.py"
echo ""
echo "----------------------------------------"
echo ""

# Start frontend
echo "2. Starting frontend (this terminal)..."
echo ""
cd saas-ui

if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

echo ""
echo "Starting Vite development server..."
echo ""
echo "🌐 Frontend will be available at: http://localhost:5173"
echo "📚 Backend API docs at: http://localhost:8000/docs"
echo ""
echo "Make sure to run the backend server first (see instructions above)!"
echo ""

npm run dev