#!/bin/bash

echo "🚀 Starting AI Agent SaaS Platform..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed. Please install Node.js 16 or higher."
    exit 1
fi

# Function to start the backend
start_backend() {
    echo -e "${BLUE}Starting AI Agent Backend...${NC}"
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies if needed
    if [ ! -f ".backend_installed" ]; then
        echo "Installing backend dependencies..."
        pip install -r requirements.txt
        touch .backend_installed
    fi
    
    # Start the FastAPI server
    echo -e "${GREEN}✓ Starting FastAPI server on http://localhost:8000${NC}"
    python src/api_server.py &
    BACKEND_PID=$!
    echo $BACKEND_PID > .backend.pid
}

# Function to start the frontend
start_frontend() {
    echo -e "${BLUE}Starting SaaS UI Frontend...${NC}"
    
    cd saas-ui
    
    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        echo "Installing frontend dependencies..."
        npm install
    fi
    
    # Start the Vite dev server
    echo -e "${GREEN}✓ Starting React app on http://localhost:3000${NC}"
    npm run dev &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../.frontend.pid
    
    cd ..
}

# Function to stop all services
stop_services() {
    echo -e "\n${YELLOW}Stopping services...${NC}"
    
    # Stop backend
    if [ -f ".backend.pid" ]; then
        kill $(cat .backend.pid) 2>/dev/null
        rm .backend.pid
    fi
    
    # Stop frontend
    if [ -f ".frontend.pid" ]; then
        kill $(cat .frontend.pid) 2>/dev/null
        rm .frontend.pid
    fi
    
    echo -e "${GREEN}✓ All services stopped${NC}"
    exit 0
}

# Trap Ctrl+C
trap stop_services INT

# Main execution
echo -e "${YELLOW}Setting up AI Agent SaaS Platform${NC}"
echo "=================================="

# Start services
start_backend
sleep 3  # Give backend time to start

start_frontend

echo ""
echo -e "${GREEN}🎉 AI Agent SaaS Platform is running!${NC}"
echo ""
echo "Access the platform at:"
echo -e "  ${BLUE}Frontend:${NC} http://localhost:3000"
echo -e "  ${BLUE}Backend API:${NC} http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for services
wait