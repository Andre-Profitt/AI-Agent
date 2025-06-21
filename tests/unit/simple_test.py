#!/usr/bin/env python3
from agent import query
from benchmarks.cot_performance import total
from examples.parallel_execution_example import results
from setup_environment import components
from tests.unit.simple_test import all_passed
from tests.unit.simple_test import batch_embeddings
from tests.unit.simple_test import manager2
from tests.unit.simple_test import module_obj
from tests.unit.simple_test import passed
from tests.unit.simple_test import retrieved_tool
from tests.unit.simple_test import test_text

from src.api_server import manager
from src.config.integrations import is_valid
from src.config.settings import issues
from src.core.llamaindex_enhanced import integration_config
from src.database.models import status
from src.database.models import tool
from src.gaia_components.advanced_reasoning_engine import embedding
from src.gaia_components.production_vector_store import texts
from src.infrastructure.config_cli import config_dict
from src.infrastructure.embedding_manager import get_embedding_manager
from src.services.integration_hub import get_integration_hub
from src.services.integration_hub import test_name
from src.services.integration_hub import tests
from src.tools.registry import module
from src.tools_introspection import name

from src.tools.base_tool import Tool

from src.agents.advanced_agent_fsm import Agent
from src.shared.types.di_types import BaseTool
# TODO: Fix undefined variables: Path, all_passed, batch_embeddings, components, config_dict, e, embedding, function, integration_config, is_valid, issues, logging, manager, manager2, module, module_obj, name, passed, query, result, results, retrieved_tool, status, sys, test_func, test_name, test_text, tests, texts, tool_registry, total
from src.services.embedding_manager import get_embedding_manager
from src.services.integration_hub import get_integration_hub
from src.tools.base_tool import tool

# TODO: Fix undefined variables: Field, all_passed, batch_embeddings, components, config_dict, e, embedding, function, get_embedding_manager, get_integration_hub, integration_config, is_valid, issues, manager, manager2, module, module_obj, name, passed, query, result, results, retrieved_tool, status, test_func, test_name, test_text, tests, texts, tool, tool_registry, total

"""

from fastapi import status
from langchain.tools import BaseTool
Simple Test Script for AI Agent Integration
This script tests basic functionality without requiring pytest or all dependencies.
Run with: python3 simple_test.py
"""

import sys

from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test basic imports"""
    logger.info("🔍 Testing basic imports...")

    try:
        from src.embedding_manager import get_embedding_manager
        logger.info("✅ Embedding manager import successful")
    except Exception as e:
        logger.info("❌ Embedding manager import failed: {}", extra={"e": e})
        return False

    try:
        from src.integration_hub import get_integration_hub
        logger.info("✅ Integration hub import successful")
    except Exception as e:
        logger.info("❌ Integration hub import failed: {}", extra={"e": e})
        return False

    try:
        from src.config.integrations import integration_config
        logger.info("✅ Config import successful")
    except Exception as e:
        logger.info("❌ Config import failed: {}", extra={"e": e})
        return False

    return True

def test_embedding_manager():
    """Test embedding manager functionality"""
    logger.info("\n🔍 Testing embedding manager...")

    try:

        manager = get_embedding_manager()
        logger.info("✅ Embedding manager created: method={}, dimension={}", extra={"manager_get_method__": manager.get_method(), "manager_get_dimension__": manager.get_dimension()})

        # Test singleton behavior
        manager2 = get_embedding_manager()
        if manager is manager2:
            logger.info("✅ Singleton behavior confirmed")
        else:
            logger.info("❌ Singleton behavior failed")
            return False

        # Test embedding
        test_text = "Hello world"
        embedding = manager.embed(test_text)
        logger.info("✅ Embedding created: length={}", extra={"len_embedding_": len(embedding)})

        # Test batch embedding
        texts = ["Text 1", "Text 2", "Text 3"]
        batch_embeddings = manager.embed_batch(texts)
        logger.info("✅ Batch embedding created: count={}", extra={"len_batch_embeddings_": len(batch_embeddings)})

        return True

    except Exception as e:
        logger.info("❌ Embedding manager test failed: {}", extra={"e": e})
        return False

def test_integration_hub():
    """Test integration hub functionality"""
    logger.info("\n🔍 Testing integration hub...")

    try:
        from src.integration_hub import get_integration_hub, tool_registry

        hub = get_integration_hub()
        logger.info("✅ Integration hub created")

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
        logger.info("✅ Tool registered successfully")

        retrieved_tool = tool_registry.get("test_tool")
        if retrieved_tool is tool:
            logger.info("✅ Tool retrieval successful")
        else:
            logger.info("❌ Tool retrieval failed")
            return False

        return True

    except Exception as e:
        logger.info("❌ Integration hub test failed: {}", extra={"e": e})
        return False

def test_config():
    """Test configuration"""
    logger.info("\n🔍 Testing configuration...")

    try:

        # Test basic config structure
        assert hasattr(integration_config, 'supabase')
        assert hasattr(integration_config, 'langchain')
        assert hasattr(integration_config, 'crewai')
        assert hasattr(integration_config, 'llamaindex')
        assert hasattr(integration_config, 'gaia')
        logger.info("✅ Configuration structure valid")

        # Test validation
        is_valid, issues = integration_config.validate()
        logger.info("✅ Configuration validation: valid={}, issues={}", extra={"is_valid": is_valid, "len_issues_": len(issues)})

        # Test to_dict
        config_dict = integration_config.to_dict()
        assert isinstance(config_dict, dict)
        logger.info("✅ Configuration serialization successful")

        return True

    except Exception as e:
        logger.info("❌ Configuration test failed: {}", extra={"e": e})
        return False

def test_enhanced_components():
    """Test enhanced components"""
    logger.info("\n🔍 Testing enhanced components...")

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
            logger.info("✅ {} import successful", extra={"name": name})
        except Exception as e:
            logger.info("⚠️ {} import failed: {}", extra={"name": name, "e": e})
            all_passed = False

    return all_passed

def main():
    """Run all tests"""
    logger.info("🚀 Starting AI Agent Integration Tests")
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
            logger.info("❌ {} test crashed: {}", extra={"test_name": test_name, "e": e})
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    logger.info("📊 Test Results Summary:")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info("{}: {}", extra={"test_name": test_name, "status": status})
        if result:
            passed += 1

    logger.info("\nOverall: {}/{} tests passed", extra={"passed": passed, "total": total})

    if passed == total:
        logger.info("🎉 All tests passed! Your AI Agent integration is working correctly.")
    else:
        logger.info("⚠️ Some tests failed. Check the output above for details.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)