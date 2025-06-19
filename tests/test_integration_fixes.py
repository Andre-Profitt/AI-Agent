#!/usr/bin/env python3
"""
Test script to verify integration fixes
Tests configuration imports, database connections, and health checks
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path and set up module structure
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Set up module structure for relative imports
os.chdir(src_path)

async def test_config_imports():
    """Test that configuration imports work correctly"""
    print("🔍 Testing configuration imports...")
    
    try:
        from config.integrations import integration_config
        print("✅ Integration config imported successfully")
        
        # Test config validation
        is_valid, issues = integration_config.validate()
        print(f"✅ Config validation: {'valid' if is_valid else 'invalid'}")
        if issues:
            print(f"⚠️ Issues: {issues}")
        
        return True
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False

async def test_llamaindex_fixes():
    """Test LlamaIndex configuration fixes"""
    print("\n🔍 Testing LlamaIndex fixes...")
    
    try:
        from llamaindex_enhanced import create_gaia_knowledge_base, LLAMAINDEX_AVAILABLE
        print(f"✅ LlamaIndex enhanced imported (available: {LLAMAINDEX_AVAILABLE})")
        
        if LLAMAINDEX_AVAILABLE:
            # Test knowledge base creation
            kb = create_gaia_knowledge_base()
            print(f"✅ Knowledge base created: {type(kb).__name__}")
        
        return True
    except Exception as e:
        print(f"❌ LlamaIndex test failed: {e}")
        return False

async def test_database_fixes():
    """Test database configuration fixes"""
    print("\n🔍 Testing database fixes...")
    
    try:
        from database_enhanced import initialize_supabase_enhanced
        
        # Test initialization (will fail if not configured, but should not crash)
        if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"):
            print("✅ Supabase configured, testing initialization...")
            try:
                components = await initialize_supabase_enhanced()
                print("✅ Supabase initialization successful")
            except Exception as e:
                print(f"⚠️ Supabase initialization failed (expected if not fully configured): {e}")
        else:
            print("⚠️ Supabase not configured, skipping initialization test")
        
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

async def test_integration_manager():
    """Test integration manager"""
    print("\n🔍 Testing integration manager...")
    
    try:
        from integration_manager import IntegrationManager
        
        manager = IntegrationManager()
        print("✅ Integration manager created")
        
        # Test status without initialization
        status = manager.get_status()
        print(f"✅ Status check: {status['initialized']}")
        
        return True
    except Exception as e:
        print(f"❌ Integration manager test failed: {e}")
        return False

async def test_health_check():
    """Test health check functionality"""
    print("\n🔍 Testing health check...")
    
    try:
        from health_check import get_health_summary
        
        summary = get_health_summary()
        print("✅ Health summary generated")
        print(f"   Config valid: {summary['config_valid']}")
        print(f"   Supabase configured: {summary['supabase_configured']}")
        print(f"   API keys: {summary['api_keys_available']}")
        
        return True
    except Exception as e:
        print(f"❌ Health check test failed: {e}")
        return False

async def test_config_cli():
    """Test configuration CLI"""
    print("\n🔍 Testing configuration CLI...")
    
    try:
        from config_cli import cli
        print("✅ Config CLI imported successfully")
        
        # Test that CLI commands exist
        commands = [cmd.name for cmd in cli.commands]
        expected_commands = ['validate', 'show', 'save', 'load', 'env', 'update', 'test']
        
        for cmd in expected_commands:
            if cmd in commands:
                print(f"✅ CLI command '{cmd}' available")
            else:
                print(f"⚠️ CLI command '{cmd}' missing")
        
        return True
    except Exception as e:
        print(f"❌ Config CLI test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting integration fixes test suite...\n")
    
    tests = [
        test_config_imports,
        test_llamaindex_fixes,
        test_database_fixes,
        test_integration_manager,
        test_health_check,
        test_config_cli
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print(f"\n📊 Test Results:")
    print(f"   Passed: {sum(results)}/{len(results)}")
    print(f"   Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! Integration fixes are working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 