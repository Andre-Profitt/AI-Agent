#!/bin/bash

echo "🚀 AI Agent SaaS - Simple Start"
echo "==============================="
echo ""

# Kill any existing processes
echo "Cleaning up old processes..."
pkill -f "python.*test_server.py" 2>/dev/null
pkill -f "python.*serve_saas.py" 2>/dev/null
lsof -ti :3000 | xargs kill -9 2>/dev/null
lsof -ti :8000 | xargs kill -9 2>/dev/null

sleep 1

# Start backend
echo "Starting backend server..."
cd "/Users/test/Desktop/ai agent/AI-Agent"
source venv/bin/activate 2>/dev/null || (python3 -m venv venv && source venv/bin/activate)
python test_server.py > backend.log 2>&1 &
BACKEND_PID=$!

# Start frontend
echo "Starting frontend server..."
python3 serve_saas.py > frontend.log 2>&1 &
FRONTEND_PID=$!

sleep 2

echo ""
echo "✅ Services started!"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "📚 Backend API: http://localhost:8000/docs"
echo ""
echo "📋 Logs:"
echo "   Backend: tail -f backend.log"
echo "   Frontend: tail -f frontend.log"
echo ""
echo "🛑 To stop: ./stop_saas.sh"
echo ""

# Create stop script
cat > stop_saas.sh << 'EOF'
#!/bin/bash
echo "Stopping services..."
pkill -f "python.*test_server.py"
pkill -f "python.*serve_saas.py"
lsof -ti :3000 | xargs kill -9 2>/dev/null
lsof -ti :8000 | xargs kill -9 2>/dev/null
echo "Services stopped!"
EOF
chmod +x stop_saas.sh