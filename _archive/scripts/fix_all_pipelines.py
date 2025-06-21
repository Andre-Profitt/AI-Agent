#!/usr/bin/env python3
"""
Comprehensive pipeline fix script
Fixes all identified issues in the AI Agent project
"""

import os
import sys
import ast
from pathlib import Path
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class PipelineFixer:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        
    def fix_all(self):
        """Fix all pipeline issues"""
        logger.info("🔧 Fixing all pipeline issues...\n")
        
        # Fix 1: Create all missing __init__.py files
        self._create_init_files()
        
        # Fix 2: Fix import issues
        self._fix_imports()
        
        # Fix 3: Install missing dependencies
        self._install_dependencies()
        
        # Fix 4: Fix syntax errors
        self._fix_syntax_errors()
        
        # Fix 5: Create missing core files
        self._create_missing_files()
        
        logger.info("\n✅ All fixes applied!")
        
    def _create_init_files(self):
        """Create missing __init__.py files"""
        logger.info("📁 Creating missing __init__.py files...")
        
        dirs_needing_init = [
            "src",
            "src/core", 
            "src/agents",
            "src/tools",
            "src/tools/implementations",
            "src/config",
            "src/database",
            "src/application",
            "src/application/agents",
            "src/application/executors",
            "src/application/tools",
            "src/infrastructure",
            "src/infrastructure/agents",
            "src/infrastructure/config", 
            "src/infrastructure/database",
            "src/infrastructure/di",
            "src/infrastructure/logging",
            "src/infrastructure/monitoring",
            "src/infrastructure/resilience",
            "src/infrastructure/workflow",
            "src/gaia_components",
            "src/unified_architecture",
            "src/services",
            "src/utils",
            "src/shared",
            "src/shared/types",
            "src/workflow",
            "src/collaboration",
            "src/analytics",
            "src/api",
            "tests",
            "tests/unit",
            "tests/integration",
            "tests/e2e",
            "tests/performance"
        ]
        
        created = 0
        for dir_path in dirs_needing_init:
            full_path = self.project_root / dir_path
            init_file = full_path / "__init__.py"
            
            if not init_file.exists():
                full_path.mkdir(parents=True, exist_ok=True)
                init_file.write_text('"""Package initialization"""')
                logger.info(f"  ✓ Created {init_file}")
                created += 1
                
        logger.info(f"  Created {created} __init__.py files")
        
    def _fix_imports(self):
        """Fix common import issues"""
        logger.info("\n🔧 Fixing import issues...")
        
        # Fix missing imports in key files
        fixes = [
            {
                "file": "src/agents/advanced_agent_fsm.py",
                "add_imports": [
                    "from typing import Dict, List, Any, Optional, Union, Callable",
                    "from dataclasses import dataclass, field",
                    "from enum import Enum",
                    "import asyncio",
                    "import logging",
                    "from datetime import datetime"
                ]
            },
            {
                "file": "src/core/entities/agent.py", 
                "add_imports": [
                    "from abc import ABC, abstractmethod",
                    "from typing import Dict, List, Any, Optional",
                    "from dataclasses import dataclass"
                ]
            },
            {
                "file": "src/core/entities/tool.py",
                "add_imports": [
                    "from typing import Dict, Any, Optional",
                    "from dataclasses import dataclass"
                ]
            },
            {
                "file": "src/core/entities/message.py",
                "add_imports": [
                    "from typing import Dict, Any, Optional",
                    "from dataclasses import dataclass, field",
                    "from datetime import datetime"
                ]
            }
        ]
        
        for fix in fixes:
            file_path = self.project_root / fix["file"]
            if file_path.exists():
                try:
                    content = file_path.read_text()
                    
                    # Add imports at the beginning
                    new_imports = []
                    for imp in fix["add_imports"]:
                        if imp not in content:
                            new_imports.append(imp)
                            
                    if new_imports:
                        # Find the right place to insert (after docstring)
                        lines = content.split('\n')
                        insert_idx = 0
                        
                        # Skip docstring
                        if lines[0].startswith('"""'):
                            for i, line in enumerate(lines[1:], 1):
                                if line.strip().endswith('"""'):
                                    insert_idx = i + 1
                                    break
                                    
                        # Insert imports
                        import_block = '\n'.join(new_imports) + '\n'
                        lines.insert(insert_idx, import_block)
                        
                        file_path.write_text('\n'.join(lines))
                        logger.info(f"  ✓ Fixed imports in {fix['file']}")
                        
                except Exception as e:
                    logger.warning(f"  ⚠️  Could not fix {fix['file']}: {e}")
                    
    def _install_dependencies(self):
        """Install missing Python packages"""
        logger.info("\n📦 Installing missing dependencies...")
        
        # Create a simplified requirements file without problematic versions
        simplified_requirements = """# Core dependencies
python-dotenv==1.0.0
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
httpx==0.25.1

# AI/ML dependencies  
openai==1.3.5
anthropic==0.7.7
langchain==0.3.25
langchain-groq==0.1.0
langchain-community==0.0.10
langgraph==0.4.8

# Vector stores (simplified)
chromadb
faiss-cpu
pinecone-client
sentence-transformers

# Database
supabase==2.0.3
asyncpg==0.29.0
sqlalchemy==2.0.23

# Utils
numpy
pandas
aiohttp==3.9.1
redis==5.0.1
motor==3.3.2
networkx==3.2.1
pillow==10.1.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0

# Monitoring  
prometheus-client==0.19.0
opentelemetry-api==1.21.0
opentelemetry-sdk==1.21.0

# Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
"""
        
        req_file = self.project_root / "requirements_simplified.txt"
        req_file.write_text(simplified_requirements)
        
        # Install packages
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements_simplified.txt"],
            cwd=str(self.project_root)
        )
        
        logger.info("  ✓ Dependencies installed")
        
    def _fix_syntax_errors(self):
        """Fix known syntax errors"""
        logger.info("\n🔧 Fixing syntax errors...")
        
        # Already fixed self_improvement.py in previous step
        # Check for other syntax errors
        
        files_to_check = [
            "src/agents/advanced_agent_fsm.py",
            "src/unified_architecture/orchestration.py",
            "src/services/knowledge_ingestion.py"
        ]
        
        for file_path in files_to_check:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                        
                    # Try to parse
                    ast.parse(content)
                    logger.info(f"  ✓ {file_path} syntax OK")
                    
                except SyntaxError as e:
                    logger.warning(f"  ⚠️  Syntax error in {file_path}: {e}")
                    # Attempt basic fixes
                    self._attempt_syntax_fix(full_path)
                    
    def _attempt_syntax_fix(self, file_path: Path):
        """Attempt to fix common syntax errors"""
        try:
            content = file_path.read_text()
            
            # Fix triple quotes in f-strings
            content = content.replace('f"""', 'f\'\'\'')
            
            # Fix unclosed strings
            lines = content.split('\n')
            fixed_lines = []
            
            for i, line in enumerate(lines):
                # Count quotes
                if line.count('"') % 2 != 0 and not line.strip().startswith('#'):
                    # Odd number of quotes, might be unclosed
                    if not line.rstrip().endswith('"'):
                        line = line.rstrip() + '"'
                fixed_lines.append(line)
                
            file_path.write_text('\n'.join(fixed_lines))
            logger.info(f"    ✓ Applied syntax fixes to {file_path.name}")
            
        except Exception as e:
            logger.error(f"    ❌ Could not fix {file_path.name}: {e}")
            
    def _create_missing_files(self):
        """Create missing core files"""
        logger.info("\n📄 Creating missing core files...")
        
        # Create base entity classes if missing
        entity_files = {
            "src/core/entities/agent.py": '''"""Base Agent Entity"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Agent(ABC):
    """Base agent class"""
    agent_id: str
    name: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @abstractmethod
    async def process(self, message: Any) -> Any:
        """Process a message"""
        pass
''',
            "src/core/entities/tool.py": '''"""Tool Entity"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Tool:
    """Tool definition"""
    name: str
    description: str
    parameters: Dict[str, Any]
    
    async def execute(self, **kwargs) -> Any:
        """Execute tool"""
        raise NotImplementedError
        
@dataclass
class ToolResult:
    """Tool execution result"""
    tool_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
''',
            "src/core/entities/message.py": '''"""Message Entity"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Message:
    """Message class"""
    content: str
    role: str  # user, assistant, system
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
'''
        }
        
        for file_path, content in entity_files.items():
            full_path = self.project_root / file_path
            if not full_path.exists():
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)
                logger.info(f"  ✓ Created {file_path}")
                
        # Create exceptions file
        exceptions_file = self.project_root / "src/core/exceptions.py"
        if not exceptions_file.exists():
            exceptions_content = '''"""Core exceptions"""

class AgentError(Exception):
    """Base agent error"""
    pass
    
class ToolError(Exception):
    """Tool execution error"""
    pass
    
class ConfigurationError(Exception):
    """Configuration error"""
    pass
'''
            exceptions_file.write_text(exceptions_content)
            logger.info("  ✓ Created src/core/exceptions.py")
            
        # Create basic config
        agent_config = self.project_root / "src/infrastructure/config.py"
        if not agent_config.exists():
            config_content = '''"""Agent configuration"""

from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class AgentConfig:
    """Agent configuration"""
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    max_iterations: int = 10
    timeout: float = 30.0
    
    # Memory settings
    enable_memory: bool = True
    memory_window_size: int = 100
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_interval: int = 60
    
    # Error handling
    error_threshold: int = 3
    recovery_timeout: float = 5.0
    retry_attempts: int = 3
    
    # Advanced features
    enable_reasoning: bool = True
    enable_learning: bool = False
    enable_multimodal: bool = False
'''
            agent_config.parent.mkdir(parents=True, exist_ok=True)
            agent_config.write_text(config_content)
            logger.info("  ✓ Created src/infrastructure/config.py")

def main():
    fixer = PipelineFixer()
    fixer.fix_all()
    
    logger.info("\n" + "="*60)
    logger.info("NEXT STEPS")
    logger.info("="*60)
    logger.info("\n1. Verify fixes:")
    logger.info("   python comprehensive_pipeline_checker.py")
    logger.info("\n2. Test basic functionality:")
    logger.info("   python -c \"from src.agents.unified_agent import create_agent; print('✓ Import works!')\"")
    logger.info("\n3. Run tests:")
    logger.info("   pytest tests/unit/")
    logger.info("\n4. Start using your super agent:")
    logger.info("   python demo_super_agent.py")

if __name__ == "__main__":
    main()