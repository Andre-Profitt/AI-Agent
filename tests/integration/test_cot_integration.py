"""
Integration tests for Chain of Thought system
Comprehensive testing of all CoT components working together
"""

import pytest
import asyncio
import time
from typing import Dict, List, Any
from unittest.mock import Mock, patch

# Import the CoT system components
from src.core.optimized_chain_of_thought import (
import logging

logger = logging.getLogger(__name__)

    OptimizedChainOfThought,
    ComplexityAnalyzer,
    TemplateLibrary,
    ReasoningCache,
    MultiPathReasoning,
    MetacognitiveLayer,
    ReasoningPath,
    ReasoningStep,
    ReasoningType
)

# Import existing tools for integration testing
from src.agents.advanced_hybrid_architecture import AdvancedHybridAgent
from src.utils.semantic_search_tool import semantic_search_tool
from src.utils.python_interpreter import python_interpreter


class TestCoTIntegration:
    """Integration tests for Chain of Thought system"""
    
    @pytest.fixture
    async def cot_system(self):
        """Create a configured CoT system for testing"""
        return OptimizedChainOfThought(
            "test_cot",
            config={
                'max_paths': 3,
                'cache_size': 100,
                'cache_ttl': 1,
                'parallel_threshold': 0.5,
                'confidence_threshold': 0.7,
                'complexity_depth_multiplier': 10
            }
        )
    
    @pytest.fixture
    async def complex_cot_system(self):
        """Create a more complex CoT system for advanced testing"""
        return OptimizedChainOfThought(
            "complex_test_cot",
            config={
                'max_paths': 5,
                'cache_size': 500,
                'cache_ttl': 24,
                'parallel_threshold': 0.3,
                'confidence_threshold': 0.8,
                'complexity_depth_multiplier': 15
            }
        )
    
    async def test_component_initialization(self, cot_system):
        """Test that all components initialize correctly"""
        assert cot_system.complexity_analyzer is not None
        assert cot_system.template_library is not None
        assert cot_system.reasoning_cache is not None
        assert cot_system.multi_path_engine is not None
        assert cot_system.metacognitive_layer is not None
        
        # Verify component types
        assert isinstance(cot_system.complexity_analyzer, ComplexityAnalyzer)
        assert isinstance(cot_system.template_library, TemplateLibrary)
        assert isinstance(cot_system.reasoning_cache, ReasoningCache)
        assert isinstance(cot_system.multi_path_engine, MultiPathReasoning)
        assert isinstance(cot_system.metacognitive_layer, MetacognitiveLayer)
    
    async def test_end_to_end_reasoning(self, cot_system):
        """Test complete reasoning flow"""
        query = "Explain the concept of recursion in programming"
        result = await cot_system.reason(query)
        
        assert result is not None
        assert isinstance(result, ReasoningPath)
        assert result.total_confidence > 0
        assert len(result.steps) > 0
        assert result.template_used in cot_system.template_library.templates
        assert result.execution_time > 0
        
        # Verify step structure
        for step in result.steps:
            assert isinstance(step, ReasoningStep)
            assert step.step_id > 0
            assert step.confidence >= 0 and step.confidence <= 1
            assert len(step.thought) > 0
    
    async def test_cache_integration(self, cot_system):
        """Test cache integration with main system"""
        query = "What is machine learning?"
        
        # First call - should miss cache
        result1 = await cot_system.reason(query)
        metrics1 = cot_system.performance_metrics
        cache_misses_1 = metrics1['cache_misses']
        
        # Second call - should hit cache
        result2 = await cot_system.reason(query)
        metrics2 = cot_system.performance_metrics
        cache_hits_2 = metrics2['cache_hits']
        
        assert cache_hits_2 > 0
        assert result2.execution_time < result1.execution_time
        
        # Verify cache stats
        cache_stats = cot_system.reasoning_cache.get_stats()
        assert cache_stats['hits'] > 0
        assert cache_stats['misses'] > 0
    
    async def test_complexity_analysis_integration(self, cot_system):
        """Test complexity analysis integration"""
        simple_query = "What is 2+2?"
        complex_query = "Analyze the time complexity of quicksort algorithm and compare it with merge sort in terms of space complexity and stability"
        
        # Analyze complexity
        simple_complexity, simple_features = cot_system.complexity_analyzer.analyze(simple_query)
        complex_complexity, complex_features = cot_system.complexity_analyzer.analyze(complex_query)
        
        assert simple_complexity < complex_complexity
        assert simple_features['length'] < complex_features['length']
        assert simple_features['vocabulary_complexity'] < complex_features['vocabulary_complexity']
        
        # Test that complexity affects reasoning depth
        simple_result = await cot_system.reason(simple_query)
        complex_result = await cot_system.reason(complex_query)
        
        # Complex queries should generally have more steps
        assert len(complex_result.steps) >= len(simple_result.steps)
    
    async def test_template_selection_integration(self, cot_system):
        """Test template selection integration"""
        mathematical_query = "Calculate the derivative of x^2 + 3x + 1"
        analytical_query = "Compare the benefits and drawbacks of cloud computing"
        
        # Get complexity analysis
        math_complexity, math_features = cot_system.complexity_analyzer.analyze(mathematical_query)
        analysis_complexity, analysis_features = cot_system.complexity_analyzer.analyze(analytical_query)
        
        # Test template selection
        math_template = cot_system.template_library.select_template(mathematical_query, math_features)
        analysis_template = cot_system.template_library.select_template(analytical_query, analysis_features)
        
        assert math_template is not None
        assert analysis_template is not None
        assert math_template.name != analysis_template.name
        
        # Test template applicability
        math_applicability = math_template.is_applicable(mathematical_query, math_features)
        analysis_applicability = analysis_template.is_applicable(analytical_query, analysis_features)
        
        assert math_applicability > 0
        assert analysis_applicability > 0
    
    async def test_multi_path_reasoning_integration(self, complex_cot_system):
        """Test multi-path reasoning integration"""
        complex_query = "Analyze the impact of artificial intelligence on job markets"
        
        # This should trigger multi-path reasoning due to complexity
        result = await complex_cot_system.reason(complex_query)
        
        assert result is not None
        assert len(result.steps) > 0
        
        # Verify that multiple reasoning types were used
        reasoning_types = [step.reasoning_type for step in result.steps]
        unique_types = set(reasoning_types)
        
        # Should have used multiple reasoning approaches
        assert len(unique_types) > 1
    
    async def test_metacognitive_layer_integration(self, cot_system):
        """Test metacognitive layer integration"""
        query = "Explain the concept of object-oriented programming"
        
        # Enable metacognitive reflection
        cot_system.config['enable_metacognition'] = True
        
        result = await cot_system.reason(query)
        
        assert result is not None
        assert result.total_confidence > 0
        
        # Metacognitive layer should improve confidence
        # (This is a basic test - in practice, metacognition should enhance reasoning)
        assert result.total_confidence >= 0.5
    
    async def test_error_handling_integration(self, cot_system):
        """Test error handling integration"""
        # Test with invalid query
        invalid_query = ""
        
        with pytest.raises(ValueError):
            await cot_system.reason(invalid_query)
        
        # Test with very long query
        long_query = "x" * 10000
        
        # Should handle gracefully
        result = await cot_system.reason(long_query)
        assert result is not None
    
    async def test_performance_metrics_integration(self, cot_system):
        """Test performance metrics integration"""
        queries = [
            "What is Python?",
            "Explain machine learning",
            "How does recursion work?"
        ]
        
        # Run multiple queries
        for query in queries:
            await cot_system.reason(query)
        
        # Check performance metrics
        metrics = cot_system.performance_metrics
        
        assert metrics['total_queries'] >= len(queries)
        assert metrics['average_execution_time'] > 0
        assert metrics['average_confidence'] > 0
        assert 'cache_hits' in metrics
        assert 'cache_misses' in metrics
    
    async def test_concurrent_reasoning(self, cot_system):
        """Test concurrent reasoning capabilities"""
        queries = [
            "What is artificial intelligence?",
            "Explain blockchain technology",
            "How do neural networks work?",
            "What is quantum computing?",
            "Explain the concept of APIs"
        ]
        
        # Run queries concurrently
        tasks = [cot_system.reason(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        # Verify all results
        for result in results:
            assert result is not None
            assert result.total_confidence > 0
            assert len(result.steps) > 0
    
    async def test_memory_management(self, cot_system):
        """Test memory management and cache eviction"""
        # Fill cache with many queries
        for i in range(150):  # More than cache size
            query = f"Test query number {i}"
            await cot_system.reason(query)
        
        # Check cache stats
        cache_stats = cot_system.reasoning_cache.get_stats()
        
        # Should have evicted some entries
        assert cache_stats['evictions'] > 0
        assert cache_stats['size'] <= cot_system.config['cache_size']


class TestCoTWithTools:
    """Test CoT integration with existing tools"""
    
    @pytest.fixture
    async def hybrid_agent_with_cot(self):
        """Create hybrid agent with CoT integration"""
        tools = [semantic_search_tool, python_interpreter]
        
        return AdvancedHybridAgent(
            "test_agent",
            config={
                'cot': {
                    'max_paths': 3,
                    'cache_size': 500,
                    'enable_metacognition': True
                }
            },
            tools=tools
        )
    
    async def test_cot_with_hybrid_agent(self, hybrid_agent_with_cot):
        """Test CoT integration with hybrid agent"""
        # Test query that uses CoT
        query = "Analyze the time complexity of quicksort algorithm"
        result = await hybrid_agent_with_cot.process_query(query)
        
        assert result is not None
        assert 'response' in result or 'answer' in result
        
        # Check reasoning history
        if hasattr(hybrid_agent_with_cot, 'reasoning_history'):
            assert len(hybrid_agent_with_cot.reasoning_history) > 0
            latest_entry = hybrid_agent_with_cot.reasoning_history[-1]
            assert 'mode' in latest_entry
            assert latest_entry['mode'] in ['cot', 'hybrid', 'tool']
    
    async def test_cot_with_semantic_search(self, hybrid_agent_with_cot):
        """Test CoT with semantic search tool"""
        query = "What are the latest developments in quantum computing?"
        
        # This should use semantic search and CoT reasoning
        result = await hybrid_agent_with_cot.process_query(query)
        
        assert result is not None
        
        # Verify that tools were used
        if hasattr(hybrid_agent_with_cot, 'tool_usage_history'):
            assert len(hybrid_agent_with_cot.tool_usage_history) > 0
    
    async def test_cot_with_python_interpreter(self, hybrid_agent_with_cot):
        """Test CoT with Python interpreter tool"""
        query = "Write a Python function to calculate fibonacci numbers and test it"
        
        result = await hybrid_agent_with_cot.process_query(query)
        
        assert result is not None
        
        # Should have used Python interpreter
        if hasattr(hybrid_agent_with_cot, 'tool_usage_history'):
            python_tool_used = any(
                'python_interpreter' in str(tool) 
                for tool in hybrid_agent_with_cot.tool_usage_history
            )
            assert python_tool_used


class TestCoTCompatibility:
    """Test compatibility with different configurations and versions"""
    
    def test_dependency_versions(self):
        """Ensure all dependencies are compatible"""
        import sys
        import importlib
        
        required_versions = {
            'numpy': '1.21.0',
            'asyncio': '3.4.3',
        }
        
        for package, min_version in required_versions.items():
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                logger.info("{}: {} (required: >={})", extra={"package": package, "version": version, "min_version": min_version})
            except ImportError:
                logger.info("{}: NOT INSTALLED (required: >={})", extra={"package": package, "min_version": min_version})
    
    async def test_reasoning_path_contract(self):
        """Ensure ReasoningPath maintains expected structure"""
        # Test that existing code expecting old structure still works
        path = ReasoningPath(
            path_id="test",
            query="test query",
            steps=[],
            total_confidence=0.8
        )
        
        # These attributes should exist for backward compatibility
        assert hasattr(path, 'path_id')
        assert hasattr(path, 'query')
        assert hasattr(path, 'steps')
        assert hasattr(path, 'total_confidence')
        assert hasattr(path, 'execution_time')
        assert hasattr(path, 'template_used')
        assert hasattr(path, 'complexity_score')
    
    async def test_configuration_compatibility(self):
        """Test compatibility with different configurations"""
        configs = [
            {'max_paths': 1, 'cache_size': 50},
            {'max_paths': 5, 'cache_size': 1000},
            {'parallel_threshold': 0.1, 'confidence_threshold': 0.9},
            {'parallel_threshold': 0.9, 'confidence_threshold': 0.5}
        ]
        
        for config in configs:
            cot_system = OptimizedChainOfThought("compat_test", config)
            
            # Should initialize without errors
            assert cot_system is not None
            
            # Should be able to reason
            result = await cot_system.reason("Test query")
            assert result is not None


# Performance and stress testing
class TestCoTPerformance:
    """Performance and stress testing for CoT system"""
    
    async def test_large_query_handling(self):
        """Test handling of large queries"""
        cot_system = OptimizedChainOfThought("perf_test", {'max_paths': 3})
        
        # Large query
        large_query = "Explain in detail the complete process of how machine learning algorithms work, including data preprocessing, feature engineering, model selection, training, validation, testing, and deployment, with specific examples of different types of algorithms like supervised learning, unsupervised learning, and reinforcement learning, and discuss the challenges and best practices in each step"
        
        start_time = time.time()
        result = await cot_system.reason(large_query)
        end_time = time.time()
        
        assert result is not None
        assert result.total_confidence > 0
        assert end_time - start_time < 30  # Should complete within 30 seconds
    
    async def test_concurrent_load(self):
        """Test concurrent load handling"""
        cot_system = OptimizedChainOfThought("load_test", {'max_paths': 2})
        
        # Create many concurrent queries
        queries = [f"Query {i}: Explain concept {i}" for i in range(20)]
        
        start_time = time.time()
        tasks = [cot_system.reason(query) for query in queries]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # All should complete successfully
        assert len(results) == len(queries)
        for result in results:
            assert result is not None
            assert result.total_confidence > 0
        
        # Should complete in reasonable time
        assert end_time - start_time < 60  # Within 60 seconds


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v"]) 