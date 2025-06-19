#!/bin/bash
# install_dependencies.sh - Comprehensive installation script for GAIA system

set -e  # Exit on any error

echo "🚀 GAIA System Dependency Installation"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        print_success "Python $PYTHON_VERSION found"
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
        print_success "Python $PYTHON_VERSION found"
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python 3.8 or higher."
        exit 1
    fi
    
    # Check Python version
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        print_error "Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Removing old one..."
        rm -rf venv
    fi
    
    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created"
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        # Windows
        source venv/Scripts/activate
    else
        # Unix/Linux/macOS
        source venv/bin/activate
    fi
    
    print_success "Virtual environment activated"
}

# Upgrade pip
upgrade_pip() {
    print_status "Upgrading pip..."
    pip install --upgrade pip
    print_success "Pip upgraded"
}

# Install core requirements
install_requirements() {
    print_status "Installing core requirements..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Core requirements installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Install optional dependencies
install_optional_deps() {
    print_status "Installing optional dependencies..."
    
    # Install spaCy language model
    print_status "Installing spaCy language model..."
    python -m spacy download en_core_web_sm
    
    # Install additional ML libraries
    pip install --no-deps torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    print_success "Optional dependencies installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    directories=(
        "data/agent_memories"
        "data/tool_learning"
        "data/vector_store"
        "logs"
        "cache"
        "models"
        "config"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    done
    
    print_success "All directories created"
}

# Verify installations
verify_installations() {
    print_status "Verifying installations..."
    
    # List of packages to verify
    packages=(
        "langchain"
        "chromadb"
        "sentence_transformers"
        "sklearn"
        "numpy"
        "pandas"
        "prometheus_client"
        "opentelemetry_api"
        "pydantic"
        "python_dotenv"
    )
    
    for package in "${packages[@]}"; do
        if python -c "import $package" 2>/dev/null; then
            version=$(python -c "import $package; print($package.__version__)" 2>/dev/null || echo "unknown")
            print_success "$package: $version"
        else
            print_error "$package: NOT FOUND"
        fi
    done
}

# Create .env file template
create_env_template() {
    print_status "Creating .env template..."
    
    if [ ! -f ".env" ]; then
        cat > .env << 'EOF'
# GAIA System Environment Configuration

# Required API Keys
GROQ_API_KEY=your-groq-api-key-here
OPENAI_API_KEY=your-openai-api-key-here  # For embeddings

# Optional API Keys (for additional tools)
ANTHROPIC_API_KEY=your-anthropic-key  # For Claude
PINECONE_API_KEY=your-pinecone-key  # For vector store
PINECONE_ENVIRONMENT=your-pinecone-env
SERPAPI_API_KEY=your-serpapi-key  # For web search
WOLFRAM_ALPHA_APPID=your-wolfram-key  # For calculations

# Configuration
LOG_LEVEL=INFO
MAX_RETRIES=3
TIMEOUT_SECONDS=30
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_STORE_TYPE=chroma  # or pinecone, weaviate

# Memory Settings
MEMORY_PERSIST_PATH=./data/agent_memories
MEMORY_MAX_SIZE=10000
MEMORY_CONSOLIDATION_INTERVAL=3600

# Tool Settings
TOOL_LEARNING_PATH=./data/tool_learning
TOOL_CONFIDENCE_THRESHOLD=0.7
MAX_TOOL_RETRIES=3

# Multi-Agent Settings
MAX_AGENTS=10
AGENT_CONSENSUS_THRESHOLD=0.7
TASK_TIMEOUT=300

# Performance Settings
CACHE_SIZE=1000
MAX_CONCURRENT_TASKS=5
BATCH_SIZE=32

# Monitoring
PROMETHEUS_PORT=9090
ENABLE_TRACING=true
JAEGER_ENDPOINT=http://localhost:14268/api/traces
EOF
        print_success ".env template created"
        print_warning "Please update .env with your actual API keys"
    else
        print_warning ".env file already exists"
    fi
}

# Run tests
run_tests() {
    print_status "Running basic tests..."
    
    if python -c "
import sys
sys.path.append('.')
try:
    from src.gaia_components.advanced_reasoning_engine import AdvancedReasoningEngine
    from src.gaia_components.enhanced_memory_system import EnhancedMemorySystem
    from src.gaia_components.adaptive_tool_system import AdaptiveToolSystem
    from src.gaia_components.multi_agent_orchestrator import MultiAgentOrchestrator
    print('✅ All GAIA components imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"; then
        print_success "Basic tests passed"
    else
        print_error "Basic tests failed"
        exit 1
    fi
}

# Main installation process
main() {
    echo "Starting GAIA system installation..."
    echo ""
    
    check_python
    create_venv
    activate_venv
    upgrade_pip
    install_requirements
    install_optional_deps
    create_directories
    create_env_template
    verify_installations
    run_tests
    
    echo ""
    echo "🎉 GAIA System Installation Complete!"
    echo "====================================="
    echo ""
    echo "Next steps:"
    echo "1. Update .env file with your API keys"
    echo "2. Run: python examples/gaia_usage_example.py"
    echo "3. Check logs/ directory for any issues"
    echo ""
    echo "For more information, see README_GAIA.md"
    echo ""
}

# Run main function
main "$@" 