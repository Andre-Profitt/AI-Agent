#!/bin/bash

echo "🚀 Starting AI Agent SaaS Platform"
echo "=================================="
echo ""

# Function to kill processes on exit
cleanup() {
    echo -e "\n🛑 Stopping all services..."
    pkill -f "python test_server.py"
    pkill -f "npm run dev"
    exit 0
}

trap cleanup INT

# Start backend in background
echo "📡 Starting backend server..."
cd "/Users/test/Desktop/ai agent/AI-Agent"
source venv/bin/activate 2>/dev/null || python3 -m venv venv && source venv/bin/activate
python test_server.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 2

# Start frontend in background
echo "🎨 Starting frontend..."
cd saas-ui
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ AI Agent SaaS Platform is running!"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Keep script running
wait