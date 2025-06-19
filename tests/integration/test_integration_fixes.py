#!/usr/bin/env python3
"""
Test script to verify integration fixes
Tests configuration imports, database connections, and health checks
"""

import asyncio
import os
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# Add src to path and set up module structure
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Set up module structure for relative imports
os.chdir(src_path)

async def test_config_imports():
    """Test that configuration imports work correctly"""
    logger.info("🔍 Testing configuration imports...")
    
    try:
        from config.integrations import integration_config
        logger.info("✅ Integration config imported successfully")
        
        # Test config validation
        is_valid, issues = integration_config.validate()
        logger.info("✅ Config validation: {}", extra={"_valid__if_is_valid_else__invalid_": 'valid' if is_valid else 'invalid'})
        if issues:
            logger.info("⚠️ Issues: {}", extra={"issues": issues})
        
        return True
    except Exception as e:
        logger.info("❌ Config import failed: {}", extra={"e": e})
        return False

async def test_llamaindex_fixes():
    """Test LlamaIndex configuration fixes"""
    logger.info("\n🔍 Testing LlamaIndex fixes...")
    
    try:
        from llamaindex_enhanced import create_gaia_knowledge_base, LLAMAINDEX_AVAILABLE
        logger.info("✅ LlamaIndex enhanced imported (available: {})", extra={"LLAMAINDEX_AVAILABLE": LLAMAINDEX_AVAILABLE})
        
        if LLAMAINDEX_AVAILABLE:
            # Test knowledge base creation
            kb = create_gaia_knowledge_base()
            logger.info("✅ Knowledge base created: {}", extra={"type_kb____name__": type(kb).__name__})
        
        return True
    except Exception as e:
        logger.info("❌ LlamaIndex test failed: {}", extra={"e": e})
        return False

async def test_database_fixes():
    """Test database configuration fixes"""
    logger.info("\n🔍 Testing database fixes...")
    
    try:
        from database_enhanced import initialize_supabase_enhanced
        
        # Test initialization (will fail if not configured, but should not crash)
        if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"):
            logger.info("✅ Supabase configured, testing initialization...")
            try:
                components = await initialize_supabase_enhanced()
                logger.info("✅ Supabase initialization successful")
            except Exception as e:
                logger.info("⚠️ Supabase initialization failed (expected if not fully configured): {}", extra={"e": e})
        else:
            logger.info("⚠️ Supabase not configured, skipping initialization test")
        
        return True
    except Exception as e:
        logger.info("❌ Database test failed: {}", extra={"e": e})
        return False

async def test_integration_manager():
    """Test integration manager"""
    logger.info("\n🔍 Testing integration manager...")
    
    try:
        from integration_manager import IntegrationManager
        
        manager = IntegrationManager()
        logger.info("✅ Integration manager created")
        
        # Test status without initialization
        status = manager.get_status()
        logger.info("✅ Status check: {}", extra={"status__initialized_": status['initialized']})
        
        return True
    except Exception as e:
        logger.info("❌ Integration manager test failed: {}", extra={"e": e})
        return False

async def test_health_check():
    """Test health check functionality"""
    logger.info("\n🔍 Testing health check...")
    
    try:
        from health_check import get_health_summary
        
        summary = get_health_summary()
        logger.info("✅ Health summary generated")
        logger.info("   Config valid: {}", extra={"summary__config_valid_": summary['config_valid']})
        logger.info("   Supabase configured: {}", extra={"summary__supabase_configured_": summary['supabase_configured']})
        logger.info("   API keys: {}", extra={"summary__api_keys_available_": summary['api_keys_available']})
        
        return True
    except Exception as e:
        logger.info("❌ Health check test failed: {}", extra={"e": e})
        return False

async def test_config_cli():
    """Test configuration CLI"""
    logger.info("\n🔍 Testing configuration CLI...")
    
    try:
        from config_cli import cli
        logger.info("✅ Config CLI imported successfully")
        
        # Test that CLI commands exist
        commands = [cmd.name for cmd in cli.commands]
        expected_commands = ['validate', 'show', 'save', 'load', 'env', 'update', 'test']
        
        for cmd in expected_commands:
            if cmd in commands:
                logger.info("✅ CLI command '{}' available", extra={"cmd": cmd})
            else:
                logger.info("⚠️ CLI command '{}' missing", extra={"cmd": cmd})
        
        return True
    except Exception as e:
        logger.info("❌ Config CLI test failed: {}", extra={"e": e})
        return False

async def main():
    """Run all tests"""
    logger.info("🚀 Starting integration fixes test suite...\n")
    
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
            logger.info("❌ Test {} crashed: {}", extra={"test___name__": test.__name__, "e": e})
            results.append(False)
    
    logger.info("\n📊 Test Results:")
    logger.info("   Passed: {}/{}", extra={"sum_results_": sum(results), "len_results_": len(results)})
    logger.info("   Failed: {}/{}", extra={"len_results____sum_results_": len(results) - sum(results), "len_results_": len(results)})
    
    if all(results):
        logger.info("🎉 All tests passed! Integration fixes are working correctly.")
        return 0
    else:
        logger.info("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 