#!/usr/bin/env python3
"""
Quick setup script for GAIA-Enhanced FSMReActAgent
Run this to set up the complete environment
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil


def create_directory_structure():
    """Create the required directory structure"""
    directories = [
        "src/agents",
        "src/gaia_components",
        "src/tools",
        "src/reasoning",
        "src/errors",
        "src/infrastructure/workflow",
        "data/agent_memories",
        "data/tool_learning",
        "data/cache",
        "logs",
        "tests",
        "examples"
    ]
    
    print("📁 Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {directory}")


def create_init_files():
    """Create __init__.py files for Python packages"""
    packages = [
        "src",
        "src/agents",
        "src/gaia_components",
        "src/tools",
        "src/reasoning",
        "src/errors",
        "src/infrastructure",
        "src/infrastructure/workflow",
        "tests"
    ]
    
    print("\n📄 Creating __init__.py files...")
    for package in packages:
        init_file = Path(package) / "__init__.py"
        init_file.touch()
        print(f"   ✓ {init_file}")


def save_gaia_components():
    """Save the GAIA component files"""
    print("\n💾 Saving GAIA components...")
    
    # Component files to create
    components = {
        "src/gaia_components/advanced_reasoning_engine.py": "# Advanced Reasoning Engine\n# Copy content from artifact: advanced_reasoning_engine",
        "src/gaia_components/enhanced_memory_system.py": "# Enhanced Memory System\n# Copy content from artifact: enhanced_memory_system",
        "src/gaia_components/adaptive_tool_system.py": "# Adaptive Tool System\n# Copy content from artifact: adaptive_tool_system",
        "src/gaia_components/multi_agent_orchestrator.py": "# Multi-Agent Orchestrator\n# Copy content from artifact: multi_agent_orchestrator_complete"
    }
    
    for filepath, content in components.items():
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"   ✓ {filepath}")
    
    print("\n⚠️  Note: You need to copy the actual component code from the artifacts above!")


def create_env_file():
    """Create .env file template"""
    print("\n🔐 Creating .env file template...")
    
    env_content = """# GAIA-Enhanced Agent Environment Variables

# Required: Your Groq API key
GROQ_API_KEY=your-groq-api-key-here

# Optional: Other API keys
# OPENAI_API_KEY=your-openai-key-here
# PINECONE_API_KEY=your-pinecone-key-here

# Configuration
LOG_LEVEL=INFO
MAX_RETRIES=3
TIMEOUT_SECONDS=30

# Memory settings
MEMORY_PERSIST_PATH=./data/agent_memories
MEMORY_CONSOLIDATION_INTERVAL=100

# Tool settings
TOOL_LEARNING_PATH=./data/tool_learning
TOOL_SELECTION_CONFIDENCE_THRESHOLD=0.7

# Multi-agent settings
MAX_AGENTS=10
AGENT_CONSENSUS_THRESHOLD=0.6
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("   ✓ .env file created")
    print("   ⚠️  Remember to add your GROQ_API_KEY!")


def create_example_files():
    """Create example files"""
    print("\n📝 Creating example files...")
    
    # Create basic usage example
    example_content = '''"""
Basic usage example for GAIA-Enhanced FSMReActAgent
"""

import asyncio
from examples.gaia_usage_example import GAIAEnhancedAgent

async def main():
    # Initialize the agent
    agent = GAIAEnhancedAgent()
    
    # Simple query
    result = await agent.solve("What is 2 + 2?")
    print(f"Answer: {result['answer']}")
    
    # Complex query
    result = await agent.solve("Calculate the population density of New York City")
    print(f"Answer: {result['answer']}")

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    with open('examples/basic_usage.py', 'w') as f:
        f.write(example_content)
    
    print("   ✓ examples/basic_usage.py")


def create_test_files():
    """Create basic test files"""
    print("\n🧪 Creating test files...")
    
    test_content = '''"""
Basic tests for GAIA-Enhanced FSMReActAgent
"""

import pytest
import asyncio
from examples.gaia_usage_example import GAIAEnhancedAgent

@pytest.fixture
def agent():
    """Create a test agent"""
    return GAIAEnhancedAgent()

@pytest.mark.asyncio
async def test_basic_query(agent):
    """Test basic query functionality"""
    result = await agent.solve("What is 2 + 2?")
    assert result["success"] is True
    assert "answer" in result

@pytest.mark.asyncio
async def test_memory_system(agent):
    """Test memory system functionality"""
    stats = agent.get_memory_insights()
    assert isinstance(stats, dict)

@pytest.mark.asyncio
async def test_tool_system(agent):
    """Test adaptive tool system"""
    insights = agent.get_tool_insights()
    assert isinstance(insights, dict)
'''
    
    with open('tests/test_gaia_agent.py', 'w') as f:
        f.write(test_content)
    
    print("   ✓ tests/test_gaia_agent.py")


def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Check if requirements_gaia.txt exists
        if Path("requirements_gaia.txt").exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_gaia.txt"], check=True)
            print("   ✓ Dependencies installed from requirements_gaia.txt")
        else:
            print("   ⚠️  requirements_gaia.txt not found, skipping dependency installation")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install dependencies: {e}")
        print("   💡 Try installing manually: pip install -r requirements_gaia.txt")


def create_readme():
    """Create README file"""
    print("\n📖 Creating README...")
    
    readme_content = """# GAIA-Enhanced FSMReActAgent

A production-ready AI agent that combines:
- **Finite State Machine (FSM)** for reliable execution flow
- **Advanced Reasoning** with multiple strategies
- **Persistent Memory** with vector search
- **Adaptive Tool Selection** using ML
- **Multi-Agent Orchestration** for complex tasks

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements_gaia.txt
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

3. **Run the example:**
   ```bash
   python examples/gaia_usage_example.py
   ```

## Features

- 🚀 **High Performance**: Optimized for GAIA benchmark tasks
- 🧠 **Smart Memory**: Learns from past interactions
- 🔧 **Adaptive Tools**: Improves tool selection over time
- 👥 **Multi-Agent**: Handles complex queries with specialized agents
- 🛡️ **Fault Tolerant**: Comprehensive error handling and recovery

## Architecture

The agent uses a Finite State Machine with the following states:
- **PLANNING**: Create execution plan
- **TOOL_EXECUTION**: Execute tools
- **SYNTHESIZING**: Combine results
- **VERIFYING**: Validate answer
- **FINISHED**: Complete

## Components

- **Advanced Reasoning Engine**: Multi-strategy reasoning
- **Enhanced Memory System**: Vector-based memory with consolidation
- **Adaptive Tool System**: ML-based tool selection
- **Multi-Agent Orchestrator**: Coordinate specialized agents

## Usage

```python
from examples.gaia_usage_example import GAIAEnhancedAgent

# Initialize agent
agent = GAIAEnhancedAgent()

# Solve queries
result = await agent.solve("What is 2 + 2?")
print(result["answer"])
```

## Testing

```bash
pytest tests/
```

## Documentation

See the `docs/` directory for detailed documentation.
"""
    
    with open('README_GAIA.md', 'w') as f:
        f.write(readme_content)
    
    print("   ✓ README_GAIA.md")


def main():
    """Main setup function"""
    print("🚀 GAIA-Enhanced FSMReActAgent Setup")
    print("=" * 50)
    
    # Create directory structure
    create_directory_structure()
    
    # Create init files
    create_init_files()
    
    # Save GAIA components
    save_gaia_components()
    
    # Create environment file
    create_env_file()
    
    # Create example files
    create_example_files()
    
    # Create test files
    create_test_files()
    
    # Install dependencies
    install_dependencies()
    
    # Create README
    create_readme()
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Add your GROQ_API_KEY to the .env file")
    print("2. Copy the actual GAIA component code to src/gaia_components/")
    print("3. Run: python examples/gaia_usage_example.py")
    print("4. Run tests: pytest tests/")


if __name__ == "__main__":
    main() 