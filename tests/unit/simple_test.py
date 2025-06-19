#!/usr/bin/env python3
"""
Simple Test Script for AI Agent Integration
This script tests basic functionality without requiring pytest or all dependencies.
Run with: python3 simple_test.py
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test basic imports"""
    print("🔍 Testing basic imports...")
    
    try:
        from src.embedding_manager import get_embedding_manager
        print("✅ Embedding manager import successful")
    except Exception as e:
        print(f"❌ Embedding manager import failed: {e}")
        return False
    
    try:
        from src.integration_hub import get_integration_hub
        print("✅ Integration hub import successful")
    except Exception as e:
        print(f"❌ Integration hub import failed: {e}")
        return False
    
    try:
        from src.config.integrations import integration_config
        print("✅ Config import successful")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    return True

def test_embedding_manager():
    """Test embedding manager functionality"""
    print("\n🔍 Testing embedding manager...")
    
    try:
        from src.embedding_manager import get_embedding_manager
        
        manager = get_embedding_manager()
        print(f"✅ Embedding manager created: method={manager.get_method()}, dimension={manager.get_dimension()}")
        
        # Test singleton behavior
        manager2 = get_embedding_manager()
        if manager is manager2:
            print("✅ Singleton behavior confirmed")
        else:
            print("❌ Singleton behavior failed")
            return False
        
        # Test embedding
        test_text = "Hello world"
        embedding = manager.embed(test_text)
        print(f"✅ Embedding created: length={len(embedding)}")
        
        # Test batch embedding
        texts = ["Text 1", "Text 2", "Text 3"]
        batch_embeddings = manager.embed_batch(texts)
        print(f"✅ Batch embedding created: count={len(batch_embeddings)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding manager test failed: {e}")
        return False

def test_integration_hub():
    """Test integration hub functionality"""
    print("\n🔍 Testing integration hub...")
    
    try:
        from src.integration_hub import get_integration_hub, tool_registry
        
        hub = get_integration_hub()
        print("✅ Integration hub created")
        
        # Test tool registry
        from src.tools.base_tool import BaseTool
        from pydantic import Field
        
        class TestTool(BaseTool):
            name: str = Field(default="test_tool", description="Tool name")
            description: str = Field(default="A test tool", description="Tool description")
            
            def _run(self, query: str) -> str:
                return f"Test result: {query}"
        
        tool = TestTool()
        tool_registry.register(tool)
        print("✅ Tool registered successfully")
        
        retrieved_tool = tool_registry.get("test_tool")
        if retrieved_tool is tool:
            print("✅ Tool retrieval successful")
        else:
            print("❌ Tool retrieval failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Integration hub test failed: {e}")
        return False

def test_config():
    """Test configuration"""
    print("\n🔍 Testing configuration...")
    
    try:
        from src.config.integrations import integration_config
        
        # Test basic config structure
        assert hasattr(integration_config, 'supabase')
        assert hasattr(integration_config, 'langchain')
        assert hasattr(integration_config, 'crewai')
        assert hasattr(integration_config, 'llamaindex')
        assert hasattr(integration_config, 'gaia')
        print("✅ Configuration structure valid")
        
        # Test validation
        is_valid, issues = integration_config.validate()
        print(f"✅ Configuration validation: valid={is_valid}, issues={len(issues)}")
        
        # Test to_dict
        config_dict = integration_config.to_dict()
        assert isinstance(config_dict, dict)
        print("✅ Configuration serialization successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_enhanced_components():
    """Test enhanced components"""
    print("\n🔍 Testing enhanced components...")
    
    components = [
        ("Database Enhanced", "src.database_enhanced", "initialize_supabase_enhanced"),
        ("Crew Enhanced", "src.crew_enhanced", "initialize_crew_enhanced"),
        ("LlamaIndex Enhanced", "src.llamaindex_enhanced", "create_gaia_knowledge_base"),
    ]
    
    all_passed = True
    
    for name, module, function in components:
        try:
            module_obj = __import__(module, fromlist=[function])
            func = getattr(module_obj, function)
            print(f"✅ {name} import successful")
        except Exception as e:
            print(f"⚠️ {name} import failed: {e}")
            all_passed = False
    
    return all_passed

def main():
    """Run all tests"""
    print("🚀 Starting AI Agent Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_imports),
        ("Embedding Manager", test_embedding_manager),
        ("Integration Hub", test_integration_hub),
        ("Configuration", test_config),
        ("Enhanced Components", test_enhanced_components),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your AI Agent integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 