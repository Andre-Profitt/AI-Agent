from src.tools.base_tool import Tool

from src.agents.advanced_agent_fsm import Agent

"""
Environment Setup Script for GAIA System
Handles environment configuration and validation
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Try to import dotenv, but don't fail if not available
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    def load_dotenv(*args, **kwargs):
        pass

logger = logging.getLogger(__name__)

class EnvironmentSetup:
    """Environment setup and validation for GAIA system"""
    
    def __init__(self):
        self.required_keys = [
            'GROQ_API_KEY'
        ]
        
        self.optional_keys = [
            'OPENAI_API_KEY',
            'ANTHROPIC_API_KEY',
            'PINECONE_API_KEY',
            'PINECONE_ENVIRONMENT',
            'SERPAPI_API_KEY',
            'WOLFRAM_ALPHA_APPID'
        ]
        
        self.directories = [
            'data/agent_memories',
            'data/tool_learning',
            'data/vector_store',
            'logs',
            'cache',
            'models',
            'config'
        ]
    
    def setup_environment(self) -> bool:
        """Setup environment variables and verify configuration"""
        print("🔧 Setting up GAIA environment...")
        
        # Load .env file
        env_path = Path('.env')
        if not env_path.exists():
            print("⚠️  No .env file found. Creating template...")
            self.create_env_template()
            print("❌ Please update .env file with your API keys and run setup again")
            return False
        
        if DOTENV_AVAILABLE:
            load_dotenv(env_path)
            print("✅ .env file loaded")
        else:
            print("⚠️  python-dotenv not available, using system environment")
        
        # Verify required API keys
        missing_keys = self.verify_api_keys()
        if missing_keys:
            print(f"❌ Missing required API keys: {', '.join(missing_keys)}")
            print("Please set them in your .env file")
            return False
        
        # Create necessary directories
        self.create_directories()
        
        # Configure logging
        self.configure_logging()
        
        # Verify optional components
        self.verify_optional_components()
        
        print("✅ Environment setup complete!")
        return True
    
    def create_env_template(self):
        """Create .env template file"""
        env_content = """# GAIA System Environment Configuration

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
"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("✅ .env template created")
    
    def verify_api_keys(self) -> List[str]:
        """Verify required API keys are set"""
        missing_keys = []
        
        for key in self.required_keys:
            if not os.getenv(key):
                missing_keys.append(key)
            else:
                print(f"✅ {key}: {'*' * (len(os.getenv(key)) - 4) + os.getenv(key)[-4:]}")
        
        # Check optional keys
        for key in self.optional_keys:
            if os.getenv(key):
                print(f"✅ {key}: {'*' * (len(os.getenv(key)) - 4) + os.getenv(key)[-4:]}")
            else:
                print(f"⚠️  {key}: Not set (optional)")
        
        return missing_keys
    
    def create_directories(self):
        """Create necessary directories"""
        print("\n📁 Creating directories...")
        
        for directory in self.directories:
            dir_path = Path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {directory}")
    
    def configure_logging(self):
        """Configure logging system"""
        print("\n📝 Configuring logging...")
        
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/gaia_agent.log'),
                logging.StreamHandler()
            ]
        )
        
        print(f"✅ Logging configured (level: {log_level})")
        print(f"✅ Log file: logs/gaia_agent.log")
    
    def verify_optional_components(self):
        """Verify optional components are available"""
        print("\n🔍 Verifying optional components...")
        
        components = {
            'prometheus_client': 'Prometheus monitoring',
            'opentelemetry_api': 'OpenTelemetry tracing',
            'chromadb': 'ChromaDB vector store',
            'sentence_transformers': 'Sentence transformers',
            'pinecone_client': 'Pinecone vector store',
            'psutil': 'System monitoring',
            'sympy': 'Symbolic mathematics',
            'spacy': 'NLP processing'
        }
        
        for module, description in components.items():
            try:
                __import__(module)
                print(f"✅ {description}: Available")
            except ImportError:
                print(f"⚠️  {description}: Not available")
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get comprehensive environment information"""
        info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'working_directory': str(Path.cwd()),
            'environment_variables': {}
        }
        
        # Environment variables (masked)
        for key in self.required_keys + self.optional_keys:
            value = os.getenv(key)
            if value:
                info['environment_variables'][key] = '*' * (len(value) - 4) + value[-4:]
            else:
                info['environment_variables'][key] = 'Not set'
        
        # Directory status
        info['directories'] = {}
        for directory in self.directories:
            dir_path = Path(directory)
            info['directories'][directory] = {
                'exists': dir_path.exists(),
                'writable': os.access(dir_path, os.W_OK) if dir_path.exists() else False
            }
        
        return info
    
    def validate_setup(self) -> bool:
        """Validate the complete setup"""
        print("\n🔍 Validating setup...")
        
        # Check if we can import GAIA components
        try:
            sys.path.append(str(Path.cwd()))
            from src.gaia_components.advanced_reasoning_engine import AdvancedReasoningEngine
            from src.gaia_components.enhanced_memory_system import EnhancedMemorySystem
            from src.gaia_components.adaptive_tool_system import AdaptiveToolSystem
            from src.gaia_components.multi_agent_orchestrator import MultiAgentOrchestrator
            print("✅ All GAIA components can be imported")
        except ImportError as e:
            print(f"❌ Import error: {e}")
            return False
        
        # Check if required directories exist and are writable
        for directory in self.directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                print(f"❌ Directory missing: {directory}")
                return False
            if not os.access(dir_path, os.W_OK):
                print(f"❌ Directory not writable: {directory}")
                return False
        
        print("✅ Setup validation passed")
        return True

def main():
    """Main setup function"""
    setup = EnvironmentSetup()
    
    if setup.setup_environment():
        if setup.validate_setup():
            print("\n🎉 GAIA System Setup Complete!")
            print("=" * 40)
            print("\nNext steps:")
            print("1. Run: python examples/gaia_usage_example.py")
            print("2. Check logs/ directory for any issues")
            print("3. See README_GAIA.md for more information")
            print("\nFor monitoring:")
            print("- Prometheus metrics: http://localhost:9090")
            print("- Health check: Use HealthCheckHandler.get_health_status()")
            print("\nHappy coding! 🚀")
        else:
            print("\n❌ Setup validation failed")
            sys.exit(1)
    else:
        print("\n❌ Environment setup failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 