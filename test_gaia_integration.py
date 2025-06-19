#!/usr/bin/env python3
"""
Test script for GAIA-Enhanced FSMReActAgent integration
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all GAIA components can be imported"""
    print("🧪 Testing imports...")
    
    try:
        # Test GAIA components
        from src.gaia_components.advanced_reasoning_engine import AdvancedReasoningEngine
        from src.gaia_components.enhanced_memory_system import EnhancedMemorySystem
        from src.gaia_components.adaptive_tool_system import AdaptiveToolSystem
        from src.gaia_components.multi_agent_orchestrator import MultiAgentGAIASystem
        
        print("   ✓ GAIA components imported successfully")
        
        # Test main agent
        from src.agents.advanced_agent_fsm import FSMReActAgent
        print("   ✓ FSMReActAgent imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return False


def test_agent_initialization():
    """Test agent initialization with mock tools"""
    print("\n🔧 Testing agent initialization...")
    
    try:
        from src.agents.advanced_agent_fsm import FSMReActAgent
        
        # Create mock tools
        class MockTool:
            def __init__(self, name: str):
                self.name = name
            
            async def __call__(self, **kwargs):
                return f"Mock result from {self.name}"
        
        tools = [
            MockTool("web_search"),
            MockTool("python_interpreter"),
            MockTool("calculator")
        ]
        
        # Initialize agent
        agent = FSMReActAgent(
            tools=tools,
            model_name="llama-3.3-70b-versatile"
        )
        
        print("   ✓ Agent initialized successfully")
        print(f"   ✓ {len(tools)} tools loaded")
        
        return agent
        
    except Exception as e:
        print(f"   ❌ Agent initialization failed: {e}")
        return None


async def test_basic_query(agent):
    """Test basic query execution"""
    print("\n🔍 Testing basic query...")
    
    try:
        # Test with a simple query
        result = await agent.run("What is 2 + 2?")
        
        print(f"   ✓ Query executed successfully")
        print(f"   ✓ Success: {result.get('success', False)}")
        print(f"   ✓ Answer: {result.get('answer', 'No answer')}")
        print(f"   ✓ Confidence: {result.get('confidence', 0):.2f}")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"   ❌ Query execution failed: {e}")
        return False


def test_gaia_components():
    """Test GAIA component functionality"""
    print("\n🧠 Testing GAIA components...")
    
    try:
        from src.gaia_components.enhanced_memory_system import EnhancedMemorySystem
        from src.gaia_components.adaptive_tool_system import AdaptiveToolSystem
        
        # Test memory system
        memory_system = EnhancedMemorySystem(
            embedding_model=None,
            persist_path=Path("data/agent_memories")
        )
        print("   ✓ Memory system initialized")
        
        # Test adaptive tool system
        class MockTool:
            def __init__(self, name: str):
                self.name = name
        
        tools = [MockTool("test_tool")]
        adaptive_tools = AdaptiveToolSystem(
            tools=tools,
            learning_path=Path("data/tool_learning")
        )
        print("   ✓ Adaptive tool system initialized")
        
        return True
        
    except Exception as e:
        print(f"   ❌ GAIA component test failed: {e}")
        return False


async def main():
    """Main test function"""
    print("🚀 GAIA-Enhanced FSMReActAgent Integration Test")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please check your installation.")
        return
    
    # Test GAIA components
    if not test_gaia_components():
        print("\n❌ GAIA component test failed.")
        return
    
    # Test agent initialization
    agent = test_agent_initialization()
    if not agent:
        print("\n❌ Agent initialization failed.")
        return
    
    # Test basic query (only if GROQ_API_KEY is set)
    if os.getenv("GROQ_API_KEY"):
        success = await test_basic_query(agent)
        if success:
            print("\n🎉 All tests passed! GAIA integration is working.")
        else:
            print("\n⚠️  Query test failed, but agent is initialized correctly.")
    else:
        print("\n⚠️  GROQ_API_KEY not set, skipping query test.")
        print("   Set GROQ_API_KEY in your .env file to test full functionality.")
    
    print("\n✅ Integration test completed!")


if __name__ == "__main__":
    asyncio.run(main()) 