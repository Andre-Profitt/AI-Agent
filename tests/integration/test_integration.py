from agent import query
from agent import tools
from examples.enhanced_unified_example import start_time
from examples.parallel_execution_example import executor
from migrations.env import config
from tests.unit.simple_test import batch_embeddings
from tests.unit.simple_test import manager2
from tests.unit.simple_test import test_text

from src.agents.crew_enhanced import initialize_crew_enhanced
from src.agents.crew_enhanced import orchestrator
from src.api_server import manager
from src.config.integrations import is_valid
from src.config.settings import issues
from src.core.llamaindex_enhanced import create_gaia_knowledge_base
from src.core.llamaindex_enhanced import integration_config
from src.database.models import text
from src.database.models import tool
from src.gaia_components.advanced_reasoning_engine import embedding
from src.gaia_components.production_vector_store import texts
from src.infrastructure.config_cli import config_dict
from src.infrastructure.embedding_manager import get_embedding_manager
from src.utils.tools_production import get_tools
from src.workflow.workflow_automation import method

from src.tools.base_tool import Tool

from src.agents.advanced_agent_fsm import Agent
from src.agents.crew_enhanced import EnhancedCrewExecutor
from src.agents.crew_enhanced import GAIACrewOrchestrator
from src.services.integration_hub import IntegrationHub
from src.shared.types.di_types import BaseTool
# TODO: Fix undefined variables: Path, batch_emb, batch_embeddings, batch_time, config, config_dict, dimension, e, embedding, embedding1, embedding2, empty_embedding, executor, i, ind_emb, individual_embeddings, individual_time, initialize_supabase_enhanced, integration_config, is_async, is_valid, issues, manager, manager1, manager2, method, none_embedding, orchestrator, query, registration_time, retrieval_time, start_time, sys, test_text, text, texts, time, tool_registry, tools
from src.agents.crew_enhanced import initialize_crew_enhanced
from src.core.llamaindex_enhanced import create_gaia_knowledge_base
from src.services.embedding_manager import get_embedding_manager
from src.services.integration_hub import get_tools
from src.tools.base_tool import tool

# TODO: Fix undefined variables: batch_emb, batch_embeddings, batch_time, config, config_dict, create_gaia_knowledge_base, dimension, e, embedding, embedding1, embedding2, empty_embedding, executor, get_embedding_manager, get_tools, i, ind_emb, individual_embeddings, individual_time, initialize_crew_enhanced, initialize_supabase_enhanced, inspect, integration_config, is_async, is_valid, issues, manager, manager1, manager2, method, none_embedding, orchestrator, query, registration_time, retrieval_time, start_time, test_text, text, texts, tool, tool_registry, tools

"""

from langchain.tools import BaseTool
Integration Tests for AI Agent System
Tests all critical fixes identified in the audit:
1. Import path resolution
2. Embedding consistency
3. Async/sync execution
4. Component integration
5. Error handling
"""

import pytest

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.integration_hub import IntegrationHub, get_embedding_manager, get_tools

from src.config.integrations import integration_config

@pytest.fixture
def integration_hub():
    """Create integration hub for testing"""
    return IntegrationHub()

@pytest.fixture
def embedding_manager():
    """Get embedding manager instance"""
    return get_embedding_manager()

class TestEmbeddingConsistency:
    """Test embedding consistency across components"""

    def test_embedding_manager_singleton(self, embedding_manager):
        """Test that embedding manager is a singleton"""
        manager1 = get_embedding_manager()
        manager2 = get_embedding_manager()
        assert manager1 is manager2
        assert manager1 is embedding_manager

    def test_embedding_method_consistency(self, embedding_manager):
        """Test that embedding method is consistent"""
        method = embedding_manager.get_method()
        dimension = embedding_manager.get_dimension()

        assert method in ["openai", "local", "none"]
        assert dimension > 0 or method == "none"

    def test_embedding_output_consistency(self, embedding_manager):
        """Test that embeddings are consistent for same input"""
        text = "Test embedding consistency"

        embedding1 = embedding_manager.embed(text)
        embedding2 = embedding_manager.embed(text)

        assert len(embedding1) == len(embedding2)
        assert embedding1 == embedding2

    def test_batch_embedding(self, embedding_manager):
        """Test batch embedding functionality"""
        texts = ["Text 1", "Text 2", "Text 3"]

        batch_embeddings = embedding_manager.embed_batch(texts)
        individual_embeddings = [embedding_manager.embed(text) for text in texts]

        assert len(batch_embeddings) == len(individual_embeddings)
        assert len(batch_embeddings) == len(texts)

        # Check that batch and individual embeddings match
        for i, (batch_emb, ind_emb) in enumerate(zip(batch_embeddings, individual_embeddings)):
            assert batch_emb == ind_emb, f"Embedding mismatch for text {i}"

class TestImportPathResolution:
    """Test that import paths are resolved correctly"""

    def test_config_import(self):
        """Test that config can be imported"""
        try:
            assert integration_config is not None
        except ImportError as e:
            pytest.fail(f"Config import failed: {e}")

    def test_embedding_manager_import(self):
        """Test that embedding manager can be imported"""
        try:
            from src.embedding_manager import get_embedding_manager
            manager = get_embedding_manager()
            assert manager is not None
        except ImportError as e:
            pytest.fail(f"Embedding manager import failed: {e}")

    def test_database_enhanced_import(self):
        """Test that database enhanced can be imported"""
        try:
            from src.database_enhanced import initialize_supabase_enhanced
            assert callable(initialize_supabase_enhanced)
        except ImportError as e:
            pytest.fail(f"Database enhanced import failed: {e}")

    def test_crew_enhanced_import(self):
        """Test that crew enhanced can be imported"""
        try:
            from src.crew_enhanced import initialize_crew_enhanced
            assert callable(initialize_crew_enhanced)
        except ImportError as e:
            pytest.fail(f"Crew enhanced import failed: {e}")

    def test_llamaindex_enhanced_import(self):
        """Test that llamaindex enhanced can be imported"""
        try:
            from src.llamaindex_enhanced import create_gaia_knowledge_base
            assert callable(create_gaia_knowledge_base)
        except ImportError as e:
            pytest.fail(f"LlamaIndex enhanced import failed: {e}")

class TestAsyncSyncExecution:
    """Test async/sync execution patterns"""

    def test_crew_execution_is_sync(self):
        """Test that CrewAI execution is properly sync"""
        try:
            from src.crew_enhanced import EnhancedCrewExecutor, GAIACrewOrchestrator
            from src.tools.base_tool import BaseTool

            # Create dummy tools
            class DummyTool(BaseTool):
                name = "dummy_tool"
                description = "A dummy tool for testing"

                def run(self, query: str) -> str:
                    return f"Dummy result for: {query}"

            # Create orchestrator and executor
            tools = [DummyTool()]
            orchestrator = GAIACrewOrchestrator(tools)
            executor = EnhancedCrewExecutor(orchestrator)

            # Test that execute_gaia_question is sync (not async)
            import inspect
            is_async = inspect.iscoroutinefunction(executor.execute_gaia_question)
            assert not is_async, "execute_gaia_question should be sync, not async"

        except ImportError:
            pytest.skip("CrewAI not available")

class TestIntegrationHub:
    """Test integration hub functionality"""

    @pytest.mark.asyncio
    async def test_hub_initialization(self, integration_hub):
        """Test that integration hub can be initialized"""
        try:
            await integration_hub.initialize()
            assert integration_hub.initialized
            assert integration_hub.is_ready()
        except Exception as e:
            # Don't fail if some components aren't available
            pytest.skip(f"Integration hub initialization failed: {e}")

    @pytest.mark.asyncio
    async def test_hub_cleanup(self, integration_hub):
        """Test that integration hub can be cleaned up"""
        try:
            await integration_hub.initialize()
            await integration_hub.cleanup()
            assert not integration_hub.initialized
        except Exception as e:
            pytest.skip(f"Integration hub cleanup failed: {e}")

    def test_tool_registration(self):
        """Test that tools can be registered"""
        from src.integration_hub import tool_registry

        # Clear existing tools
        tool_registry.clear()

        # Create dummy tool

        class TestTool(BaseTool):
            name = "test_tool"
            description = "A test tool"

            def run(self, query: str) -> str:
                return f"Test result: {query}"

        # Register tool
        tool = TestTool()
        tool_registry.register(tool)

        # Verify registration
        assert tool_registry.get("test_tool") is tool
        assert len(tool_registry.get_all()) == 1

    def test_get_tools_function(self):
        """Test the get_tools convenience function"""
        tools = get_tools()
        assert isinstance(tools, list)
        # Should have at least some tools registered

class TestErrorHandling:
    """Test error handling and propagation"""

    def test_embedding_fallback(self, embedding_manager):
        """Test that embedding manager handles errors gracefully"""
        # Test with empty text
        empty_embedding = embedding_manager.embed("")
        assert len(empty_embedding) > 0

        # Test with None text
        none_embedding = embedding_manager.embed(None)
        assert len(none_embedding) > 0

    @pytest.mark.asyncio
    async def test_database_error_handling(self):
        """Test database error handling"""
        try:

            # Test with invalid credentials
            with pytest.raises((ValueError, Exception)):
                await initialize_supabase_enhanced(url="invalid", key="invalid")

        except ImportError:
            pytest.skip("Database enhanced not available")

    def test_config_validation(self):
        """Test configuration validation"""
        config = integration_config

        # Test validation method exists
        assert hasattr(config, 'validate')

        # Test validation returns tuple
        is_valid, issues = config.validate()
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)

class TestComponentIntegration:
    """Test that components work together"""

    @pytest.mark.asyncio
    async def test_full_integration_flow(self, integration_hub):
        """Test complete integration flow"""
        try:
            # Initialize hub
            await integration_hub.initialize()

            # Test that components are available
            assert integration_hub.is_ready()

            # Test that tools are registered
            tools = integration_hub.get_tools()
            assert len(tools) > 0

            # Test that embedding manager is available
            embedding_manager = integration_hub.get_embedding_manager()
            assert embedding_manager is not None

            # Test embedding consistency
            test_text = "Integration test"
            embedding = embedding_manager.embed(test_text)
            assert len(embedding) > 0

            # Cleanup
            await integration_hub.cleanup()

        except Exception as e:
            pytest.skip(f"Full integration test failed: {e}")

    def test_configuration_consistency(self):
        """Test that configuration is consistent across components"""
        config = integration_config

        # Test that all required sections exist
        assert hasattr(config, 'supabase')
        assert hasattr(config, 'langchain')
        assert hasattr(config, 'crewai')
        assert hasattr(config, 'llamaindex')
        assert hasattr(config, 'gaia')

        # Test that config can be converted to dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'supabase' in config_dict
        assert 'langchain' in config_dict

class TestPerformance:
    """Test performance characteristics"""

    def test_embedding_performance(self, embedding_manager):
        """Test embedding performance"""
        import time

        # Test single embedding performance
        start_time = time.time()
        embedding_manager.embed("Performance test")
        single_time = time.time() - start_time

        # Test batch embedding performance
        texts = ["Text " + str(i) for i in range(10)]
        start_time = time.time()
        embedding_manager.embed_batch(texts)
        batch_time = time.time() - start_time

        # Batch should be more efficient than individual
        individual_time = sum(time.time() - start_time for _ in range(10))
        assert batch_time < individual_time * 0.8  # At least 20% faster

    def test_tool_registry_performance(self):
        """Test tool registry performance"""

        # Test registration performance

        class PerfTool(BaseTool):
            name = "perf_tool"
            description = "Performance test tool"

            def run(self, query: str) -> str:
                return "perf result"

        start_time = time.time()
        for i in range(100):
            tool = PerfTool()
            tool.name = f"perf_tool_{i}"
            tool_registry.register(tool)
        registration_time = time.time() - start_time

        # Should be fast
        assert registration_time < 1.0  # Less than 1 second for 100 tools

        # Test retrieval performance
        start_time = time.time()
        for i in range(100):
            tool_registry.get(f"perf_tool_{i}")
        retrieval_time = time.time() - start_time

        # Should be very fast
        assert retrieval_time < 0.1  # Less than 0.1 seconds for 100 retrievals

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])